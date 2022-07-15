package devicepluginserver

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"encoding/json"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

const (
	alnairvGPUMemResource = "alnair/vgpu-memory"

	alnairGPUID      = "ai.centaurus.io/alnair-gpu-id"
	alnairID         = "ai.centaurus.io/alnair-id"
	physicalGPUcnt   = "ai.centaurus.io/physical-gpu-count"
	physicalGPUuuids = "ai.centaurus.io/physical-gpu-uuids"
	virtualGPUcnt    = "ai.centaurus.io/vitual-gpu-count"
	timestamp        = "ai.centaurus.io/scheduler-timestamp"
)

var (
	clientset *kubernetes.Clientset
	nodeName  string
)

func clientsetInit() (*kubernetes.Clientset, error) {
	// creates the in-cluster config

	var (
		config *rest.Config
		err    error
	)
	config, err = rest.InClusterConfig()
	if err != nil {
		nodeName, _ = os.Hostname()
		log.Printf("InClusterConfig failed, assume test env, retry env KUBECONFIG and use hostname %s as nodeName", nodeName)
		kubeconfigFile := os.Getenv("KUBECONFIG")
		if kubeconfigFile == "" {
			return nil, errors.New("environment variable KUBECONFIG is empty, please set with kube config file path")
		}
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfigFile) //use the default path for now, pass through arg later
	}
	if err != nil {
		return nil, err
	}
	// creates the clientset
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, err
	}
	return clientset, err

}

func PatchNode() error {
	// step 0: connect to cluster clientset
	var err error
	clientset, err = clientsetInit() // return clientset as a global variable, will be used in PatchPod() as well
	if err != nil {
		log.Printf("Error: patch node, failed to connect to kubernetes clientset: %v", err)
		return err
	}

	// step 1: get node name from environment variable passed down from the deployment yaml file
	if nodeName == "" {
		nodeName = os.Getenv("NODE_NAME")
		if nodeName == "" {
			log.Printf("Error: patch node, failed to get node name.")
			return errors.New("please pass node name as NODE_NAME env through downward API in yaml")
		}
	}

	//step 2: get physical device info
	cnt, uuids, sizes := GetPhysicalDeivces()
	patchData := map[string]interface{}{"metadata": map[string]map[string]string{"annotations": {
		physicalGPUcnt:   fmt.Sprintf("%d", cnt),
		physicalGPUuuids: fmt.Sprintf("%s", strings.Join(uuids, ",")),
		virtualGPUcnt:    fmt.Sprintf("%s", strings.Join(sizes, ",")),
	}}}
	playLoadBytes, _ := json.Marshal(patchData)
	_, err = clientset.CoreV1().Nodes().Patch(context.TODO(), nodeName, types.StrategicMergePatchType, playLoadBytes, metav1.PatchOptions{})
	if err != nil {
		log.Printf("Error: patch node, failed to patch node annotation: %v", err)
		return err
	}
	log.Println("successfully patch the static gpu info to the annotations of node", nodeName)
	return nil
}

func PatchPod(gpuIds string, alnairIds string, gpuCnt int) error {

	pod, _ := getPendingAlnairPod(gpuCnt)

	patchData := map[string]interface{}{"metadata": map[string]map[string]string{"annotations": {alnairGPUID: gpuIds, alnairID: alnairIds}}}
	playLoadBytes, _ := json.Marshal(patchData)
	namespace := pod.Namespace
	podName := pod.Name
	_, err := clientset.CoreV1().Pods(namespace).Patch(context.TODO(), podName, types.StrategicMergePatchType, playLoadBytes, metav1.PatchOptions{})
	if err != nil {
		log.Printf("Error: patch pod, failed to patch pod annotation: %v", err)
		return err
	}
	log.Printf("===patch done for current allocate request!")
	return nil
}

func getPendingAlnairPod(reqCnt int) (*v1.Pod, error) {
	//candidate pods: 1)status is pending and 2) on this node and 3)requested gpu number equal to allocate requests
	//if there are more than one candidates, select the one with earliest in queue timestamp, timestamp is added by scheduler
	pods, err := clientset.CoreV1().Pods("").List(context.TODO(), metav1.ListOptions{FieldSelector: "spec.nodeName=" + nodeName + ",status.phase=Pending"})
	if err != nil {
		log.Printf("Error: get pod failed on node %s: %v", nodeName, err)
		return nil, err
	}
	//only one pending pod
	if len(pods.Items) == 1 {
		log.Println("only one pending pod:", pods.Items[0].Name, "patch gpu id.")
		return &pods.Items[0], nil
	}
	candidates := &v1.PodList{}
	// find the one with same gpu request count and not gpu-id not exist yet
	// Note: gpu-id not exist is important, pending pods after patching/allocation will stay pending for a while will be listed when query
	for _, pod := range pods.Items {
		if pod.Annotations[alnairGPUID] != "" {
			log.Println(pod.Name, "is already patched with gpu id")
			continue
		}
		gpus := 0
		for _, ctnr := range pod.Spec.Containers {
			reqGPUs := ctnr.Resources.Limits[alnairvGPUMemResource]
			reqGPUQuantity, ok := reqGPUs.AsInt64()
			if ok {
				gpus += int(reqGPUQuantity)
			}
		}
		if gpus == reqCnt {
			candidates.Items = append(candidates.Items, pod)
		}
	}
	//only one pending pod request the same amount of gpus
	if len(candidates.Items) == 1 {
		log.Println("only one pending unpatched pod:", candidates.Items[0].Name, "request", reqCnt, "gpus, patch gpu id.")
		return &candidates.Items[0], nil
	}
	//find the earliest in queue pod among all the candidate pods
	var cur_ts string
	min_idx := 0
	min_ts := fmt.Sprintf("%d", time.Now().UnixNano())
	for idx, pod := range candidates.Items {
		cur_ts = pod.Annotations[timestamp]
		if cur_ts != "" {
			if cur_ts < min_ts {
				min_ts = cur_ts
				min_idx = idx
				log.Printf("rare cases, multiple candidates, update pod %s to the earliest pod, in queue time %s\n", pod.Name, cur_ts)
			}
		} else {
			log.Printf("warning: multiple candidates, no timestamp found, patching could be wrong")
		}
	}
	log.Printf("%d pods request %d vgpus on node %s, pending same time\n", len(candidates.Items), reqCnt, nodeName)
	return &candidates.Items[min_idx], nil
}

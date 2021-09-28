package collection

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strconv"

	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	framework "k8s.io/kubernetes/pkg/scheduler/framework"
)

type Data struct {
	NodeValue Node_static
	GPUValue  map[string]GPU_mem_usage
}

type Node_static struct {
	Model   string
	Count   int
	MemSize uint64
}

type GPU_mem_usage struct {
	mem_used uint64
	mem_free uint64
}

func (s *Data) Clone() framework.StateData {
	c := &Data{
		NodeValue: s.NodeValue,
		GPUValue:  s.GPUValue,
	}
	return c
}

func CollectValues(state *framework.CycleState, nodeName string) *framework.Status {
	///
	config, _ := clientcmd.BuildConfigFromFlags("", "/root/kube-dev/kube/config")
	///
	clientset, _ := kubernetes.NewForConfig(config)
	nodeList, _ := clientset.CoreV1().Nodes().List(context.TODO(), meta.ListOptions{})

	DOMAIN := "ai.centaurus.io"
	len_nodeList := len(nodeList.Items) - 1
	re := regexp.MustCompile(`[-]?\d[\d,]*[\.]?[\d{2}]*`)

	node_index := int(0)
	for i := int(0); i < len_nodeList; i++ {
		if nodeName == nodeList.Items[i].Name {
			node_index = i
		}
	}

	annotations := nodeList.Items[node_index].ObjectMeta.Annotations
	gpuStatic := annotations[DOMAIN+"/gpu-static"]
	gpuStatic_submatchall := re.FindAllString(gpuStatic, -1)
	len_s := len(gpuStatic)
	model := gpuStatic[25 : len_s-48]
	mem_size, _ := strconv.ParseUint(gpuStatic_submatchall[len(gpuStatic_submatchall)-3], 10, 64)
	count, _ := strconv.Atoi(gpuStatic_submatchall[0])
	node_value := Node_static{
		Model:   model,
		Count:   count,
		MemSize: mem_size,
	}

	var node_GPU_value = make(map[string]GPU_mem_usage)
	for id := int(0); id < count; id++ {
		cur_usage := annotations[DOMAIN+"/gpu-"+strconv.Itoa(id)]
		cur_usage_submatchall := re.FindAllString(cur_usage, -1)
		mem_used, err := strconv.ParseUint(cur_usage_submatchall[0], 10, 64)
		mem_free, err := strconv.ParseUint(cur_usage_submatchall[1], 10, 64)
		if err != nil {
			// handle error
			fmt.Println(err)
			os.Exit(2)
		}
		node_GPU_value["gpu-"+strconv.Itoa(id)] = GPU_mem_usage{
			mem_used: mem_used,
			mem_free: mem_free,
		}
	}
	data := Data{
		NodeValue: node_value,
		GPUValue:  node_GPU_value,
	}
	state.Lock()
	state.Write("Anno", &data)
	state.Unlock()
	return framework.NewStatus(framework.Success, "")
}

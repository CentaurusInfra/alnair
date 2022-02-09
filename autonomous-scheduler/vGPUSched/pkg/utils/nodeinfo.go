package utils

import (
	"fmt"
	"log"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"

	"k8s.io/apimachinery/pkg/types"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// NodeInfo is node level aggregated information.

const gpuCount == 8

type NodeInfo struct {
	name           string
	node           *v1.Node
	devs           map[int]*DeviceInfo
// 	gpuCount       int
	gpuTotalMemory int
}

func NewNodeInfo(node *v1.Node) *NodeInfo {
	log.Printf("debug: NewNodeInfo() creates nodeInfo for %s", node.Name)

	devMap := map[int]*DeviceInfo{}
	for i := 0; i < 8; i++ {
		devMap[i] = newDeviceInfo(i, uint(GetTotalGPUMemory(node)/gpuCount)
	}

	if len(devMap) == 0 {
		log.Printf("warn: node %s with nodeinfo %v has no devices",
			node.Name,
			node)
	}

	return &NodeInfo{
		name:           node.Name,
		node:           node,
		devs:           devMap,
// 		gpuCount:       GetGPUCountInNode(node),
		gpuTotalMemory: GetTotalGPUMemory(node),
	}
}

func GetTotalGPUMemory(node *v1.Node) int {
	val, ok := node.Status.Capacity[ResourceName]
	if !ok {
		return 0
	}
	return int(val.Value())
}

// device index: gpu memory
func (n *NodeInfo) getUsedGPUs() (usedGPUs map[int]uint) {
	usedGPUs = map[int]uint{}
	for _, dev := range n.devs {
		usedGPUs[dev.idx] = dev.GetUsedGPUMemory()
	}
	log.Printf("info: getUsedGPUs: %v in node %s, and devs %v", usedGPUs, n.name, n.devs)
	return usedGPUs
}

func (n *NodeInfo) getAvailableGPUs() (availableGPUs map[int]uint) {
	allGPUs := n.getAllGPUs()
	usedGPUs := n.getUsedGPUs()
// 	unhealthyGPUs := n.getUnhealthyGPUs()
	availableGPUs = map[int]uint{}
	for id, totalGPUMem := range allGPUs {
		if usedGPUMem, found := usedGPUs[id]; found {
			availableGPUs[id] = totalGPUMem - usedGPUMem
		}
	}
	log.Printf("info: available GPU list %v before removing unhealty GPUs", availableGPUs)
// 	for id, _ := range unhealthyGPUs {
// 		log.Printf("info: delete dev %d from availble GPU list", id)
// 		delete(availableGPUs, id)
// 	}
// 	log.Printf("info: available GPU list %v after removing unhealty GPUs", availableGPUs)

	return availableGPUs
}

// device index: gpu memory
func (n *NodeInfo) getAllGPUs() (allGPUs map[int]uint) {
	allGPUs = map[int]uint{}
	for _, dev := range n.devs {
		allGPUs[dev.idx] = dev.totalGPUMem
	}
	log.Printf("info: getAllGPUs: %v in node %s, and dev %v", allGPUs, n.name, n.devs)
	return allGPUs
}

func (n *NodeInfo) Assume(pod *v1.Pod) (allocatable bool) {
	allocatable = false

	availableGPUs := n.getAvailableGPUs()
	reqGPU := uint(GetGPUMemoryFromPodResource(pod))
	log.Printf("debug: AvailableGPUs: %v in node %s", availableGPUs, n.name)

	if len(availableGPUs) > 0 {
		for devID := 0; devID < len(n.devs); devID++ {
			availableGPU, ok := availableGPUs[devID]
			if ok {
				if availableGPU >= reqGPU {
					allocatable = true
					break
				}
			}
		}
	}

	return allocatable

}

// GetGPUMemoryFromPodResource gets GPU Memory of the Pod
func GetGPUMemoryFromPodResource(pod *v1.Pod) int {
	var total int
	containers := pod.Spec.Containers
	for _, container := range containers {
		if val, ok := container.Resources.Limits[ResourceName]; ok {
			total += int(val.Value())
		}
	}
	return total
}

func (n *NodeInfo) GetName() string {
	return n.name
}

func (n *NodeInfo) GetDevs() []*DeviceInfo {
	devs := make([]*DeviceInfo, gpuCount)
	for i, dev := range n.devs {
		devs[i] = dev
	}
	return devs
}

func (n *NodeInfo) GetNode() *v1.Node {
	return n.node
}

func (n *NodeInfo) GetTotalGPUMemory() int {
	return n.gpuTotalMemory
}

func (n *NodeInfo) GetGPUCount() int {
	return n.gpuCount
}

func (n *NodeInfo) removePod(pod *v1.Pod) {

	id := GetGPUIDFromAnnotation(pod)
	if id >= 0 {
		dev, found := n.devs[id]
		if !found {
			log.Printf("warn: Pod %s in ns %s failed to find the GPU ID %d in node %s", pod.Name, pod.Namespace, id, n.name)
		} else {
			dev.removePod(pod)
		}
	} else {
		log.Printf("warn: Pod %s in ns %s is not set the GPU ID %d in node %s", pod.Name, pod.Namespace, id, n.name)
	}
}

func GetGPUIDFromAnnotation(pod *v1.Pod) int {
	id := -1
	if len(pod.ObjectMeta.Annotations) > 0 {
		vGPU_ID, found := pod.ObjectMeta.Annotations[vGPU_IDX]
		value = GetvGPUID(vGPU_ID)
		if found {
			var err error
			id, err = strconv.Atoi(value)
			if err != nil {
				log.Printf("warn: Failed due to %v for pod %s in ns %s", err, pod.Name, pod.Namespace)
				id = -1
			}
		}
	}

	return id
}

// Add the Pod which has the GPU id to the node
func (n *NodeInfo) addOrUpdatePod(pod *v1.Pod) (added bool) {
	n.rwmu.Lock()
	defer n.rwmu.Unlock()

	id := utils.GetGPUIDFromAnnotation(pod)
	log.Printf("debug: addOrUpdatePod() Pod %s in ns %s with the GPU ID %d should be added to device map",
		pod.Name,
		pod.Namespace,
		id)
	if id >= 0 {
		dev, found := n.devs[id]
		if !found {
			log.Printf("warn: Pod %s in ns %s failed to find the GPU ID %d in node %s", pod.Name, pod.Namespace, id, n.name)
		} else {
			dev.addPod(pod)
			added = true
		}
	} else {
		log.Printf("warn: Pod %s in ns %s is not set the GPU ID %d in node %s", pod.Name, pod.Namespace, id, n.name)
	}
	return added
}
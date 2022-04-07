package utils

import (
	v1 "k8s.io/api/core/v1"
	"log"
	"strconv"
	"strings"
)

// NodeInfos is node level aggregated information.
type NodeInfos struct {
	name     string
	node     *v1.Node
	devs     map[int]*DeviceInfos
	gpuCount int
	// 	gpuTotalMemory int
}

// get Physical gpuCount from node annotation
func GetPhysicalGPUCountFromNodeAnno(node *v1.Node) int {
	val := -1

	if len(node.ObjectMeta.Annotations) > 0 {
		vet, found := node.ObjectMeta.Annotations["ai.centaurus.io/physical-gpu-count"]
		if found {
			var err error
			val, err = strconv.Atoi(vet)
			if err != nil {
				log.Printf("warn: Failed obtain node annotation due to %v for node %s", err, node.Name)
				val = -1
			}
		}
	}
	return val
}

func GetVirtualGPUCountFromNodeAnno(node *v1.Node) int {
	val := -1
	if len(node.ObjectMeta.Annotations) > 0 {
		vet, found := node.ObjectMeta.Annotations["ai.centaurus.io/virtual-gpu-count"]
		vGPU_count := strings.Split(vet, ",")
		if found {
			var err error
			val, err = strconv.Atoi(vGPU_count[0])
			if err != nil {
				log.Printf("warn: Failed obtain node annotation due to %v for node %s", err, node.Name)
				val = -1
			}
		}
	}
	return val
}

func NewNodeInfos(node *v1.Node) *NodeInfos {
	gpuCount := GetPhysicalGPUCountFromNodeAnno(node)
	if gpuCount == -1 {
		log.Printf("debug: cannot get Physical GPU count from node annotation")
	}

	log.Printf("debug: NewNodeInfos() creates nodeInfos for %s", node.Name)
	devMap := map[int]*DeviceInfos{}
	for i := 0; i < int(gpuCount); i++ {
		devMap[i] = NewDeviceInfos(i, uint(GetVirtualGPUCountFromNodeAnno(node)))
	}

	if len(devMap) == 0 {
		log.Printf("warn: node %s with nodeinfos %v has no devices",
			node.Name,
			node)
	}

	return &NodeInfos{
		name:     node.Name,
		node:     node,
		devs:     devMap,
		gpuCount: gpuCount,
		// 		gpuTotalMemory: GetTotalGPUMemory(node),
	}
}

func (n *NodeInfos) GetDevs() []*DeviceInfos {
	devs := make([]*DeviceInfos, n.gpuCount)
	for i, dev := range n.devs {
		devs[i] = dev
	}
	return devs
}

func (n *NodeInfos) GetNode() *v1.Node {
	return n.node
}

// device index: gpu memory
func (n *NodeInfos) getAllGPUs() (allGPUs map[int]uint) {
	allGPUs = map[int]uint{}
	for _, dev := range n.devs {
		allGPUs[dev.idx] = dev.totalGPUMem
	}
	log.Printf("info: getAllGPUs: %v in node %s, and dev %v", allGPUs, n.name, n.devs)
	return allGPUs
}

func (n *NodeInfos) getUsedGPUs() (usedGPUs map[int]uint) {
	usedGPUs = map[int]uint{}
	for _, dev := range n.devs {
		usedGPUs[dev.idx] = dev.GetUsedGPUMemory()
	}
	log.Printf("info: getUsedGPUs: %v in node %s, and devs %v", usedGPUs, n.name, n.devs)
	return usedGPUs
}

func (n *NodeInfos) getAvailableGPUs() (availableGPUs map[int]uint) {
	allGPUs := n.getAllGPUs()
	usedGPUs := n.getUsedGPUs()
	availableGPUs = map[int]uint{}
	for id, totalGPUMem := range allGPUs {
		if usedGPUMem, found := usedGPUs[id]; found {
			availableGPUs[id] = totalGPUMem - usedGPUMem
		}
	}
	log.Printf("info: available GPU list %v", availableGPUs)

	return availableGPUs
}

func IsGPUsharingPod(pod *v1.Pod) bool {
	return GetGPUMemoryFromPodResource(pod) > 0
}

// GetGPUMemoryFromPodResource gets GPU Memory of the Pod
func GetGPUMemoryFromPodResource(pod *v1.Pod) int {
	var total int
	containers := pod.Spec.Containers
	for _, container := range containers {
		if val, ok := container.Resources.Limits["alnair/vgpu-memory"]; ok {
			total += int(val.Value())
		}
	}
	return total
}

func (n *NodeInfos) Assume(pod *v1.Pod) (allocatable bool) {
	allocatable = false

	availableGPUs := n.getAvailableGPUs()
	log.Printf("debug: AvailableGPUs: %v in node %s", availableGPUs, n.name)
	reqGPU := uint(GetGPUMemoryFromPodResource(pod))

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

// get assumed timestamp from the pod
func getAssumeTimeFromPodAnnotation(pod *v1.Pod) (assumeTime uint64) {
	if assumeTimeStr, ok := pod.ObjectMeta.Annotations["scheduler-timestamp"]; ok {
		u64, err := strconv.ParseUint(assumeTimeStr, 10, 64)
		if err != nil {
			log.Printf("Failed to parse assume Timestamp %s due to %v", assumeTimeStr, err)
		} else {
			assumeTime = u64
		}
	}
	return assumeTime
}

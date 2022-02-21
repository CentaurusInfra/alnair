package utils

import (
	"log"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
    "strings"
    "sort"
)

// DeviceInfos is pod level aggregated information

type DeviceInfos struct {
	idx    int
	podMap map[types.UID]*v1.Pod
// 	usedGPUMem  uint
	totalGPUMem uint
}

func NewDeviceInfos(index int, totalGPUMem uint) *DeviceInfos {
	return &DeviceInfos{
		idx:         index,
		totalGPUMem: totalGPUMem,
		podMap:      map[types.UID]*v1.Pod{},
	}
}



func (d *DeviceInfos) GetUsedGPUMemory() (gpuMem uint) {
	log.Printf("debug: GetUsedGPUMemory() podMap %v, and its address is %p", d.podMap, d)
// 	d.rwmu.RLock()
// 	defer d.rwmu.RUnlock()
	for _, pod := range d.podMap {
		if pod.Status.Phase == v1.PodRunning || pod.Status.Phase == v1.PodPending{
			log.Printf("debug: the pod %s in ns %s is counted as used due to its status in %s", pod.Name, pod.Namespace, pod.Status.Phase)
			gpuMem += GetGPUMemoryFromPodAnnotation(pod)
	    }
	}
	return gpuMem
}

// func (d *DeviceInfos) GetUsedGPUMemory() (gpuMem uint) {
// 	log.Printf("debug: GetUsedGPUMemory() podMap %+v, and its address is %p", d.podMap, d)
// 	for _, pod := range d.podMap {
// // 		if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
// // 			log.Printf("debug: skip the pod %s in ns %s due to its status is %s", pod.Name, pod.Namespace, pod.Status.Phase)
// // 			continue
// // 		}
// 		gpuMem += GetGPUMemoryFromPodAnnotation(pod)
// 		log.Printf("debug: skip the pod %s in ns %s due to its status is %s", pod.Name, pod.Namespace, pod.Status.Phase)
// 	}
// 	return gpuMem
// }

// GetGPUMemoryFromPodAnnotation gets the GPU Memory of the pod
func GetGPUMemoryFromPodAnnotation(pod *v1.Pod) (gpuMemory uint) {
	if len(pod.ObjectMeta.Annotations) > 0 {
		vet, found := pod.ObjectMeta.Annotations["ai.centaurus.io/alnair-gpu-id"]
		vGPU_ID := strings.Split(vet,",")
		idx := GetvGPUIDX(vGPU_ID)
		if found {
			s := len(idx)
			if s < 0 {
				s = 0
			}
			gpuMemory += uint(s)
		}
	}
	log.Printf("debug: pod %s in ns %s with status %v has vGPU Mem %d",
		pod.Name,
		pod.Namespace,
		pod.Status.Phase,
		gpuMemory)
	return gpuMemory
}

// get vGPU index
func GetvGPUIDX(vGPU_ID []string) []string {
	var ret []string
	sort.Strings(vGPU_ID)
	for _, sid := range vGPU_ID {
		id := strings.SplitN(sid, "_", 2)[1]
		if len(ret) == 0 || id != ret[len(ret)-1]{
			ret = append(ret, id)
		}
	}
	return ret
}
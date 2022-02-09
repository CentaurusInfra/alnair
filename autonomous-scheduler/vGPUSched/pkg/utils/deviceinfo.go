package utils

import (
	"log"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"strconv"
	"time"
)

// DeviceInfo is pod level aggregated information
// Here I assume a running pod has annotation like vGPU_IDX: GPU_XXX_XXX_0, GPU_XXX_XXX_1

type DeviceInfo struct {
	idx    int
	podMap map[types.UID]*v1.Pod
	// usedGPUMem  uint
	totalGPUMem uint
}

func newDeviceInfo(index int, totalGPUMem uint) *DeviceInfo {
	return &DeviceInfo{
		idx:         index,
		totalGPUMem: totalGPUMem,
		podMap:      map[types.UID]*v1.Pod{},
	}
}

func (d *DeviceInfo) GetPods() []*v1.Pod {
	pods := []*v1.Pod{}
	for _, pod := range d.podMap {
		pods = append(pods, pod)
	}
	return pods
}

func (d *DeviceInfo) GetTotalGPUMemory() uint {
	return d.totalGPUMem
}

func (d *DeviceInfo) GetUsedGPUMemory() (gpuMem uint) {
	log.Printf("debug: GetUsedGPUMemory() podMap %v, and its address is %p", d.podMap, d)
// 	d.rwmu.RLock()
// 	defer d.rwmu.RUnlock()
	for _, pod := range d.podMap {
		if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
			log.Printf("debug: skip the pod %s in ns %s due to its status is %s", pod.Name, pod.Namespace, pod.Status.Phase)
			continue
		}
		// gpuMem += utils.GetGPUMemoryFromPodEnv(pod)
		gpuMem += GetGPUMemoryFromPodAnnotation(pod)
	}
	return gpuMem
}

// GetGPUMemoryFromPodAnnotation gets the GPU Memory of the pod
func GetGPUMemoryFromPodAnnotation(pod *v1.Pod) (gpuMemory uint) {
	if len(pod.ObjectMeta.Annotations) > 0 {
		vGPU_ID, found := pod.ObjectMeta.Annotations[vGPU_IDX] //need to confirm the annotation format
		value := GetvGPUID(vGPU_ID) //according to vGPU_IDX format, convert it to a number
		if found {
			s, _ := len(value)
			if s < 0 {
				s = 0
			}
			gpuMemory += uint(s)
		}
	}
	log.Printf("debug: pod %s in ns %s with status %v has GPU Mem %d",
		pod.Name,
		pod.Namespace,
		pod.Status.Phase,
		gpuMemory)
	return gpuMemory
}

// need to confirm with the annotation format
func GetvGPUID(vGPU_ID []string) []string {
    var ret []string
    for _, sid := range vGPU_ID {
        id := string.SplitN(sid, "_", 2)[1]
        if len(ret) == 0 || id != ret[len(ret)-1]{
            ret = append(ret, id)
        }
    }
    return ret
}

// get assumed timestamp
func getAssumeTimeFromPodAnnotation(pod *v1.Pod) (assumeTime uint64) {
	if assumeTimeStr, ok := pod.ObjectMeta.Annotations[ResourceAssumeTime]; ok {
		u64, err := strconv.ParseUint(assumeTimeStr, 10, 64)
		if err != nil {
			log.Warningf("Failed to parse assume Timestamp %s due to %v", assumeTimeStr, err)
		} else {
			assumeTime = u64
		}
	}

	return assumeTime
}

// get GPU ID from pod annotation
func GetGPUIDFromPodAnnotation(pod *v1.Pod) (id int) {
    var err error
    id := -1
    if len(pod.ObjectMeta.Annotations) > 0 {
        value, found := pod.ObjectMeta.Annotations[ResourceIndex] // upon to annotation definition
        if found {
			id, err = strconv.Atoi(value)
			if err != nil {
			    log.Printf("warn: Failed due to %v for pod %s in ns %s", err, pod.Name, pod.Namespace)
				id = -1
			}
		}
	}
	return id
}

func (d *DeviceInfo) addPod(pod *v1.Pod) {
	log.Printf("debug: dev.addPod() Pod %s in ns %s with the GPU ID %d will be added to device map",
		pod.Name,
		pod.Namespace,
		d.idx)
	d.podMap[pod.UID] = pod
	log.Printf("debug: dev.addPod() after updated is %v, and its address is %p",
		d.podMap,
		d)
}

func (d *DeviceInfo) removePod(pod *v1.Pod) {
	log.Printf("debug: dev.removePod() Pod %s in ns %s with the GPU ID %d will be removed from device map",
		pod.Name,
		pod.Namespace,
		d.idx)
	delete(d.podMap, pod.UID)
	log.Printf("debug: dev.removePod() after updated is %v, and its address is %p",
		d.podMap,
		d)
}

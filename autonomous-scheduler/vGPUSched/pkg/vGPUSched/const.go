package vGPUSched

import (
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1beta1"
)

const (
    ResourceName            = "alnair/vgpu-mem"

    ResourceIndex           = "Alnair_GPU_MEM_IDX"
	AssignedFlag            = "Alnair_GPU_MEM_ASSIGNED"
	ResourceAssumeTime      = "Alnair_GPU_MEM_ASSUME_TIME"
)
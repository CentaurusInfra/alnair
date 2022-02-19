# Alnair Device Plugin (Kubernetes)
Alnair device plugin uses the Kubernetes extension mechanism [Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/) to enable the K8s managed containers to access Nvidia GPUs. It depends on the Nvidia docker runtime [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Compared with the [Nvidia Device Plugin](https://github.com/NVIDIA/k8s-device-plugin), Alnair amis to provide fractional GPU, GPU limits enforcement and GPU resources isolation among containers.

The GPU limits enforcement is done using the [LD_PRELOAD](https://osterlund.xyz/posts/2018-03-12-interceptiong-functions-c.html) mechanism. The specific library intercepted is the [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html). We choose to intercept at this level in order to support all CUDA applications as well as to have a stable public API.

The CUDA driver API interpose library can be found [here](https://github.com/CentaurusInfra/alnair/tree/main/intercept-lib)

## High level architecture diagram:
<img src="./docs/images/alnair-device-plugin.jpg">

## Quick Start

### Prerequisites
* Provision a Kubernetes cluster with at least one GPU node.
* Install Nvidia docker runtime [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) and configure it as the default docker runtime.
* A kubeconfig with enough permissions. 


### Steps

Please refer to the [CUDA interpose library](https://github.com/CentaurusInfra/alnair/tree/main/intercept-lib#steps)

# Alnair Device Plugin (Kubernetes)
Alnair device plugin uses the Kubernetes extension mechanism [Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/) to enable the K8s managed containers to access Nvidia GPUs. It depends on the Nvidia docker runtime [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Compared with the [Nvidia Device Plugin](https://github.com/NVIDIA/k8s-device-plugin), Alnair amis to provide fractional GPU, GPU limits enforcement and GPU resources isolation among containers.

The GPU limits enforcement is done using the [LD_PRELOAD](https://osterlund.xyz/posts/2018-03-12-interceptiong-functions-c.html) mechanism. The specific library intercepted is the [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html). We choose to intercept at this level in order to support all CUDA applications as well as to have a stable public API. 

## High level architecture diagram:
<img src="./docs/images/alnair-device-plugin.jpg">

## Quick Start

### Prerequisites
* Provision a Kubernetes cluster with at least one GPU node.
* Install Nvidia docker runtime [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) and configure it as the default docker runtime.
* A kubeconfig with enough permissions. 


### Steps

1. Clone repo
```bash
git clone https://github.com/CentaurusInfra/alnair.git
cd alnair/alnair-device-plugin

```
2. Run the device plugin
```bash
sudo go run cmd/alnair-device-plugin/main.go
```
3. Open another terminal and create a sample pod with 2GiB GPU memory limit

```yaml
cat pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
    - name: cuda-container
      image: nvidia/cuda:9.0-devel
      resources:
        limits:
          alnair/vgpu-mem: 2 
      command: ["sleep", "3600"]
```
```bash
kubectl create -f pod.yaml
```
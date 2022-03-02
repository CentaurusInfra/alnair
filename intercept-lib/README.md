# CUDA Interpose Library
This shared library provides the key functionality for [Alnair Device Plugin](https://github.com/CentaurusInfra/alnair/tree/main/alnair-device-plugin) to control the GPU memory usage and the GPU compute usage within a container. Using CUDA interception, it sits between the user applications and the system CUDA libraries. 

The specific technique used for CUDA interception is the [LD_PRELOAD](https://osterlund.xyz/posts/2018-03-12-interceptiong-functions-c.html) mechanism. Memory allocation and kernel launch functions within the [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html) are intercepted. We don't intercept the CUDA runtime API because deep learning frameworks may use CUDA driver API directly.

This library is used in a way that is totally transparent to the end user. When a container (requesting GPU resources) is launched by the kubelet, the [Alnair Device Plugin](https://github.com/CentaurusInfra/alnair/tree/main/alnair-device-plugin) will silently mount the library into the container, and set the LD_PRELOAD environment variable to the mounted location. 

Once the user program first issues the cuInit() call, the library will register the container with the vGPU registration server, which has been deployed together with the Alnair device plugin. The registration server sets up the 
appropriate workspace that are necessary for the library to run correctly, where host process IDs, resource limits, and etc are saved.

After the registration and remaining initializations, the CUDA driver API calls issued by the user applications are intercepted by the library and GPU resource limits are enforced as required. The GPU memory resource limit is a hard limit. The GPU compute resource limits are enforced to track the GPU utilization as reported by nvidia-smi. 

## High level architecture diagram:
<img src="https://github.com/CentaurusInfra/alnair/blob/main/alnair-device-plugin/docs/images/alnair-device-plugin.jpg?raw=true">

## Quick Start

### Prerequisites
* Build needs to be done on a GPU node.
* The Nvidia driver and CUDA toolkit must be installed first.

### Steps

1. Build
```bash
make
```

2. Manual Deploy on the GPU node
```bash
cd <path to alnair source code>
sudo su
mkdir -p /opt/alnair
cp intercept-lib/build/lib/libcuinterpose.so /opt/alnair/libcuinterpose.so
cd alnair-device-plugin
go run cmd/alnair-device-plugin/main.go &
go run cmd/vgpu-server/main.go
```

3. Test pod
```yaml
cat pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
    - name: cuda-container
      image: hxhp/elastic-horovod-demo:1.0
      resources:
        limits:
          alnair/vgpu-memory: 2
          alnair/vgpu-compute: 50 
      command: ["python", "/examples/tensorflow2_mnist.py"]
```
```bash
kubectl create -f pod.yaml
```

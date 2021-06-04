# Continuous GPU usage profiling


## Introduction

## System Diagram
<img  src="https://github.com/CentaurusInfra/alnair/blob/main/profiling/images/System%20Diagram.png" width="700" height="300">

## Quick Start
   
### 1. Prerequisite: 
 - K8s cluster with Nvidia GPU plugin installed
 - GPU node with Nvidia driver installed
 - GPU node with nvidia container runtime installed, docker default runtime is set to nvidia-container-runtime ([installation guide](https://github.com/NVIDIA/nvidia-container-runtime))

Detailed cluster installation guide can be refered [here](https://github.com/CentaurusInfra/alnair/blob/main/profiling/k8s-clusters/README.md).

### 2. Install profiler

To use profiler in a Kubernetes cluster, only two yaml files [prometheus-complete.yaml](https://github.com/CentaurusInfra/alnair/blob/main/profiling/prometheus-service/prometheus-complete.yaml), [profiler-dcgm-daemonset.yaml](https://github.com/CentaurusInfra/alnair/blob/main/profiling/profiler/profiler-dcgm-daemonset.yaml) need to be applied.
1. Install prometheus service with ```kubectl apply -f prometheus-complete.yaml```
2. Install profiler daemon set with ```kubectl apply -f profiler-dcgm-daemonset.yaml```

**Note**: Install the prometheus service first, since profiler needs to connect to Prometheus server and read metrics data.

### 3. View profiler results

Profiler results are written into cluster nodes' annotations. With ```kubectl describe node <your-node-name> | grep ai.centaurus.io```, results like the followings can be seen. By default, profiler will update annotations every 30 seconds, if memory utilization pattern changes.

<img  src="https://github.com/CentaurusInfra/alnair/blob/main/profiling/images/annotation_results.png" width="700" height="160">

In addition, GPU metrics can be viewed through Prometheus's web UI. In the ```prometheus-complete.yaml```, prometheus' container port is mapped to the host port for the sake of simiplicity. http://prometheus-node-ip:9090, the node-ip is the IP address of the node where Prometheus server is deployed. 

<img  src="https://github.com/CentaurusInfra/alnair/blob/main/profiling/images/prometheus_UI.png" width="1000" height="500">

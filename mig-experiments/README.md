# Project: Multiple Instance GPU (MIG) 

### Background information
  
MIG is a new feature on Nvidia Ampere-architecture GPU. It provides an ability to partition one GPU hardware into multiple intances of 'virtual' gpus, so users can share one GPU hardware among multiple programs. More information can be found here:  
  https://www.nvidia.com/en-us/technologies/multi-instance-gpu/

To enable MIG in K8s, there are several components to be installed:  
   1. GPU node feature discover module:  
      kubectl apply -k https://github.com/kubernetes-sigs/node-feature-discovery/deployment/overlays/default?ref=v0.11.2
   2. GPU operator
       https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/gpu-operator-mig.html#install-gpu-operator-mig  
         
         
How does MIG work in kubernetes:  
    https://developer.nvidia.com/blog/getting-kubernetes-ready-for-the-a100-gpu-with-multi-instance-gpu/  
  
Advanced training topics in GPU:  
  

### CUDA code samples :
advance features requires SMS="80"
https://github.com/nvidia/cuda-samples

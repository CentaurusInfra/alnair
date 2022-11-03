# Project: Multiple Instance GPU (MIG) 

### Background information
  
MIG is a new feature on Nvidia Ampere-architecture GPU. It is a function to partition one GPU hardware into multiple intances of 'virtual' GPUs, so users can share one GPU hardware among multiple programs. More information can be found here:  
  https://www.nvidia.com/en-us/technologies/multi-instance-gpu/

To enable MIG in K8s, there are several components to be installed:  
   1. GPU node feature discover module:  
      kubectl apply -k https://github.com/kubernetes-sigs/node-feature-discovery/deployment/overlays/default?ref=v0.11.2
   2. GPU operator  
      https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/gpu-operator-mig.html#install-gpu-operator-mig  
         
         
How does MIG work in kubernetes? Here are two pieces of information:
   .  https://developer.nvidia.com/blog/getting-kubernetes-ready-for-the-a100-gpu-with-multi-instance-gpu/  
   .  https://info.nvidia.com/rs/156-OFN-742/images/How_to_easily_use_GPUs_with_Kubernetes.pdf
  
Advanced training topics in GPU:  
  

### CUDA code samples :
advance features requires SMS="80"
https://github.com/nvidia/cuda-samples

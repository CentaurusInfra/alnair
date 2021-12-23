# Scheduler 

Simply put, the Scheduler handles 1) updating a new job's ```Spec.ReplicaSpec.TargetReplicas``` when it is submitted and 2) updating *every* job's ```Spec.ReplicaSpec.TargetReplicas``` when jobs are reshuffled every set period of time 

## Goals  

The Scheduler should be able to make intelligent decisions on which nodes and which GPUs to assign to every DL job that is submitted to the cluster. 

A DLT job has a few key aspects that make this decision non-trivial and require more than the base Kubernetes scheduler. These are: 
- They exhibit slowdowns when trained across nodes, likely due to network bandwith (varying depending on model and input size)
- Certain model architectures may work better with certain GPUs. For example, a computer vision model may be 10x faster than an LSTM model on one GPU but only 2x faster on another GPU. We describe this as "GPU-job affinity".  
- DLT jobs are typically very long such that the time it takes for profiling to be performed through dry runs may be worth it in the long run to find a better configuration overall
- Long DLT job times may mean that resources are disproportionately locked out and be "unfair" 

## Requirements 

The requirements for a Kubernetes operator would be: 
- Be able to maintain a global view of all (running or waiting) jobs submitted to the cluster, so that GPU-job affinity is maximized and waiting jobs are not starved of GPU 
- Have access to node and GPU information to make intelligent scheduling 

## Algorithms 

Testing of the efficacy of a certain algorithm for reshuffling and assigning to a new GPU is beneficial can be performed with the Python Synthetic Cluster Simulator. 
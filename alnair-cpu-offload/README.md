# Project: CPU Offload

### Background information
  
AI jobs (training and inferencing) in the cloud are normally handled by the collaboration of CPU and GPU to get the best performance. There are 4 major tasks in the AI workflow:  
  1.  preparation  
  2.  data feeding  
  3.  model computing  
  4.  result retrieving/reporting.  

CPU is responsible for task 1. 2. 4. GPU only focuses on task 3: computing.  

Computing technology has been improved dramatically over years. Taking Nvidia GPU as an example, the computing capacity is increased 450x from G80 to A100. Meanwhile, the memory onboard of GPU is creased by 50x. In order to run GPU with full capacity, the data transfering from system memory into GPU memory becomes a critical task. On a system with multiple A100s, CPU has a heavy duty to prepare the data in time for all GPUs.

If we can reduce the workload of data transferring on CPU, it not only removes the potential bottleneck of the whole AI process, but also increases the parallelism and resourc sharing possibility.

There are some new technologies in hardware and software to support a possible GPU-centered system architecture, which includes NVMe SSD, gpudirect. THis project will explore a novel solution to provide a GPU-centered AI-training System (GAS) to offload CPU in the process. GAS tries to achieve the following goals:
1.  Efficiency improvements by resource sharing in the cloud
2.  Performance improvements in generic AI training and inference applications by high parallelism

### Goal: Offloading CPU by GPU-centered AI system

1. NVMe-SSD for training data storage
2. store and read kernels from NVMe SSD as a share lib storage
3. sharing NVMe-SSD between GPUs on one node
4. gpudirect data among GPU in the cloud

### Project Planning
1. Stage I (architecture research)  
  . Explore the research papers for side-band SSD accessible.  
  . Setup GPU + SSD environment  
  . Develop a test program to analyze the architecture behaviors  
2. Stage II (performance research)  
  . Implement a set of API to support benchmark  
  . Implement an application to demo the capability  
  . Benchmark the performance of the new architecture with applications of native operations  
3. Stage III (Production implementation)  
  . Explore the API for complex applications like TF or Ptorch  
  . Define the CPU-Offload application field  
  . Implement API and infrastructure  

### sub tasks : 

1. Read the papers about the current work in NVMe-SSD in HPC  
  <a id="1">[1]</a> 
  Zaid Qureshi (2022) BaM: A Case for Enabling Fine-grain High Throughput GPU-Orchestrated Access to Storage  
  <a id="2">[2]</a> 
  Shweta Pandey (2022) GPM: Leveraging Persistent Memory from a GPU  
  <a id="3">[3]</a> 
  Samyam Rajbhandari (2021) ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning    
3. Setup the development/experiment environment on v100   
  a. SATA NVMe drive  
  b. M2 NVMe drive  
  c. customized M2. NVMe drive to get 26Gbps performance  
3. Explore gpu-SSD direct API  
  a. gpudirect API from the paper (1)  
  b. GPM API from the paper (2)  
4. CPU-Offload package  
  a. a lib to contain supporting functions.   
  b. working example to support gpudirect in cuda  
  c. a working example to do AI training in Cuda completely with SSD  
  d. a working example to benchmark the performance  
5. Performance Probe and improvements

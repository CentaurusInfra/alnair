# Project: CPU Offload

### Background information
  
AI jobs (training and inferencing) in the cloud are normally handled by the collaboration of CPU and GPU to get the best performance. There are 4 major tasks in the AI workflow: I. preparation II. data feeding III. model computing IV. result retrieving/reporting. CPU is responsible for task I. II. IV. GPU only focuses on task III: computing. Computing technology has been improved dramatically over years. Taking Nvidia GPU as an example, the computing capacity is increased 450x from G80 to A100. Meanwhile, the memory onboard of GPU is creased by 50x. In order to run GPU with full capacity, the data transfering from system memory into GPU memory becomes a critical task. On a system with multiple A100s, CPU has a heavy duty to prepare the data in time for all GPUs.

If we can reduce the workload of data transferring on CPU, it not only removes the potential bottleneck of the whole AI process, but also increases the parallelism and resourc sharing possibility.

There are some new technologies to support a possible GPU-centered system architecture, which includes NVMe SSD, gpudirect. THis project will explore a novel solution to provide a GPU-centered system to offload CPU.

### Goal: Offloading CPU by GPU-centered AI system

1. NVMe-SSD for training data storage
2. store and read kernels from NVMe SSD as a share lib storage
3. sharing NVMe-SSD between GPUs on one node
4. gpudirect data among GPU in the cloud

### sub tasks : 

1. Read the papers about the current work in NVMe-SSD in HPC
2. Setup the development/experiment environment on v100
  a. sata NVMe drive
  b. M2 NVMe drive
  c. customized M2. NVMe drive to get 26Gbps performance
3. Explore gpu-SSD direct API
  a. gpudirect API from the paper ()
  b. gpm AP from the paper ()
4. CPU-Offload package 
  a. a working example to support gpudirect in cuda
  b. a working example to do AI training in Cuda completely with SSD
  c. a working example to benchmark the performance

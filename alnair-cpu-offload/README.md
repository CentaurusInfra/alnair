# Project: CPU Offload

### Background information
  
AI jobs (training and inferencing) in the cloud are normally handled by the collaboration of CPU and GPU to get the best performance. There are 4 major tasks in the AI workflow: I. preparation II. data feeding III. model computing IV. result retrieving/reporting. CPU is responsible for task I. II. IV. GPU only focuses on task III: computing. Computing technology has been improved dramatically over years. Taking Nvidia GPU as an example, the computing capacity is increased 450x from G80 to A100. Meanwhile, the memory onboard of GPU is creased by 50x. In order to run GPU with full capacity, the data transfering from system memory into GPU memory becomes a critical task. On a system with multiple A100s, CPU has a heavy duty to prepare the data in time for all GPUs.


### Goal: Building an intelligent platform to improve AI workloads efficiency



###  : Building an intelligent platform to improve AI workloads efficiency



# Alnair Function and Performance Test

## GPU sharing
### GPU Sharing Definition: run two or more programs on one GPU card.
### Test 1
Compare the job completion time of two idential gpu programs on Alnair sharing, Kubeshare sharing, MPS, and baremetal GPU. 

Settings: 
 - Same GPU card
 - Each program with 50% GPU compute limit and half size of GPU memory

Metrics:
 - Job completion time
   1. Job 1 completion time on Nvidia GPU (single job, on physical GPU, no vGPU)
   2. Job 1 completion time on vGPU=100 (single job, use vGPU, but give 100% util limit, i.e., no limit, check overhead)
   3. Job 1 completion time on vGPU= 10, 30, 50, 70 (single job, grant different util limit, check when slow down happens)
   4. Job 1 completion time on vGPU=50, and launch Job 1 with vGPU=50 at the same time (two jobs, check Job 1 completion time with different interference)
   5. Job 1 completion time on vGPU=50, and lanuch Job 2 with vGPU=50 at the same time (two jobs, check Job 1 completion time with different interference)
   6. Job 1 completion time on vGPU=50, and lanuch Job 3 with vGPU=50 at the same time (two jobs, check Job 1 completion time with different interference)
 - GPU utilization, (and nvprof timeline)


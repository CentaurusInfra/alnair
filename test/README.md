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
 - GPU utilization, (and nvprof timeline)

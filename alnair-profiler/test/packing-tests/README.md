# Example Scripts for Single-GPU
## Measuring Interference of Job Packing Using MPS 

Currently, these scripts are ran directly on the server using [Nvidia MPS](https://github.com/CentaurusInfra/alnair/tree/main/fine-grained-sharing). 

For base time, the jobs are all ran individually with user-specified parameters (batch_size in particular as another variable aside from the script). 

The purpose of these scripts is to investigate packing interference for DLT jobs. This is to see what type of DLT jobs are most viable to be eligible for packing for the
scheduler. Currently, the scripts print out the average time/10_steps and the interference measurement would simply be recorded by hand. The pickle function for further 
analysis is not used and the only measurement currently is time/10_steps, but it can be easily modified to measure others such as time/unit = time/10_steps * batch_size.

For comparison, jobs are initially ran individually and recorded for short epochs, and then the same jobs are ran for longer amounts of time (by modifying epochs) and other 
jobs are ran alongside with all_small_tests.sh; the outputs of each individual job in all_small_tests.sh are recorded and compared to its individual times. 

Example: Measuring interference of various scripts on pyt-cf-rn50-pack.py -e 4 -b 16

First SSH into the pod in two windows. 

Run script on its own to see individual time: 

```pyt-cf-rn50-pack.py -e 4 -b 16```

After it finishes and its time is recorded, run it for indefinite amount of time on one window. 

```pyt-cf-rn50-pack.py -e 100 -b 16``` 

On the second window, run every other script. 

```./all-small-tests.sh```

Record the times on the second window, and exit out of the long process with ^C. 

## Possible Issues

Data is not all loaded into memory so data transfer from disk to GPU is an additional factor that may affect results.

PyTorch's and Tensorflow's memory constraints on their processes ([PyTorch](https://pytorch.org/docs/stable/generated/torch.cuda.set_per_process_memory_fraction.html) and [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/compat/v1/GPUOptions)) 
can be used but much of it depends on the specified batch size anyways. Additionally, the two functions are not exact due to memory used to initialize GPU libraries such as CUDA and cuDNN.

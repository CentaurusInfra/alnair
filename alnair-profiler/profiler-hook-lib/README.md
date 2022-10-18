# Profiler CUDA Interpose 

This library is used to profile AI applications' performance by tapping CUDA driver level API.


## Quick Start

### Prerequisites
* Build needs to be done on a GPU node.
* The Nvidia driver and CUDA toolkit must be installed first.

### Steps

1. Build
```bash
git clone https://github.com/CentaurusInfra/alnair.git
cd alnair/alnair-profiler/profiler-hook-lib
make

ls build/lib
total 116
drwxrwxr-x 2 xxxx xxxx   4096 Sep 14 10:52 ./
drwxrwxr-x 5 xxxx xxxx   4096 Aug 31 21:15 ../
-rwxrwxr-x 1 xxxx xxxx 109992 Sep 14 10:52 libpfinterpose.so*
```

timeline log field 'kind' definition:  
0: Host -> Host  
1: Host -> Device  
2: Device -> Host  
3: Device -> Device  
4: Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing  


2. testing  
This test requires to setup a kubernetes cluster.  

```bash
kubeclt apply -f test/profiler-interpose-test.yaml

ls -l test
total 36
drwxrwxr-x 2 xxxx xxxx  4096 Aug 31 21:15 intercept-cuda-11.3-demo
-rw-rw-r-- 1 xxxx xxxx   258 Sep  6 13:46 metrics.log
-rw-rw-r-- 1 xxxx xxxx 19081 Aug 31 21:15 pyt-test1.py
-rw-rw-r-- 1 xxxx xxxx  1494 Aug 31 21:15 single_node_swin_trans.yaml
-rw-rw-r-- 1 xxxx xxxx  2751 Sep  6 13:46 timeline.log


cat ./test/metrics.log
name:mem_used,count:452984832
name:cuInit,count:3,burst:4349618
name:cuMemAlloc,count:125,burst:176696
name:cuMemFree,count:8,burst:4
name:cuLaunchKernel,count:523848,burst:1436854
name:cuMemcpyH2D,count:6,burst:0
name:cuMemcpyD2H,count:6,burst:0
name:cuGetProcAddress,count:0,burst:0
name:cuMemcpyH2D_ASYNC,count:0,burst:0
name:cudaMemcpy,count:0,burst:0
name:cudaMemcpyAsync,count:1072,burst:803485


cat ./test/timeline.log
name:cuInit, start:1663174503143895116, burst:4336641, bytes:0, kind:-1
name:cuMemAlloc, start:1663174512512732618, burst:1192, bytes:2097152, kind:-1
name:cudaMemcpyAsync, start:1663174512513972203, burst:140285180914600, bytes:3456, kind:1
name:cudaMemcpyAsync, start:1663174512514104262, burst:140285180914600, bytes:128, kind:1
name:cudaMemcpyAsync, start:1663174512514132684, burst:140285180914600, bytes:128, kind:1
name:cudaMemcpyAsync, start:1663174512514154613, burst:140285180914600, bytes:128, kind:1
name:cudaMemcpyAsync, start:1663174512514172355, burst:140285180914600, bytes:128, kind:1
name:cudaMemcpyAsync, start:1663174512514189929, burst:140285180914600, bytes:8, kind:1
name:cudaMemcpyAsync, start:1663174512514216777, burst:140285180914600, bytes:36864, kind:1


...

3. a sample yaml to test the interpose lib (profiler-interpose-test.yaml)

...
apiVersion: v1
kind: Pod
metadata:
  name: interpose-test
spec:
  containers:
  - name: pytorch
    image: centaurusinfra/pytorch-testing
    env:
    - name: PFLOG
      value: /log/pflog    
    - name: LD_PRELOAD
      value: /lib/interpose/libpfinterpose.so    
    command: ["sh", "-c", "python /workspace/test/pyt-cf-rn50-pack.py"]
    volumeMounts:
    - mountPath: /dev/shm
      name: dshm
    - mountPath: /nfs_3/data/cityscapes/
      name: dataset 
    - mountPath: /lib/interpose/
      name: intercept       
    - mountPath: /workspace/test
      name: test       
    - name: pflog
      mountPath: /log/pflog      
  volumes:
  - name: dshm  # this is to trick pytorch when large size of shared memory is needed
    emptyDir:
      medium: Memory
  - name: dataset
    hostPath:
      path: /nfs_3/data/cityscapes/
      type: Directory
  - name: intercept
    hostPath:
      path: /home/xxx/dev/alnair/alnair-profiler/profiler-hook-lib/build/lib
      type: Directory
  - name: test
    hostPath:
      path: /home/xxx/dev/alnair/alnair-profiler/profiler-hook-lib/test
      type: Directory      
  - name: pflog
    hostPath:
      path: /var/lib/alnair/workspace/         
  restartPolicy: OnFailure
  ...
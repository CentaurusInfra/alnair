# Profiler CUDA Interpose 

This library is used to profile AI applications' performance by tapping CUDA driver level API.


## Quick Start

### Prerequisites
* Build needs to be done on a GPU node.
* The Nvidia driver and CUDA toolkit must be installed first.

### Steps

1. Build
```bash
cd alnair/alnair-profiler/profiler-hook-lib
make

ls build/lib
```

the output file (libpfinterpose.so) is in build/lib.

2. testing (pyt_test1.py)
```bash
cd alnair/alnair-profiler/profiler-hook-lib
PFLOG=./test

LD_PRELOAD=./build/lib/libpfinterpose.so python ./test/pyt_test1.py

ls -l test
total 36
drwxrwxr-x 2 steven steven  4096 Aug 31 21:15 intercept-cuda-11.3-demo
-rw-rw-r-- 1 steven steven   258 Sep  6 13:46 metrics.log
-rw-rw-r-- 1 steven steven 19081 Aug 31 21:15 pyt-test1.py
-rw-rw-r-- 1 steven steven  1494 Aug 31 21:15 single_node_swin_trans.yaml
-rw-rw-r-- 1 steven steven  2751 Sep  6 13:46 timeline.log


cat ./test/metrics.log
name:cuInit
count:4
burst:4213891
name:cuMemAlloc
count:124
burst:174295
name:cuMemFree
count:14
burst:4
name:cuLaunchKernel
count:186804
burst:1414970
name:cuMemcpyH2D
count:4
burst:134
name:cuMemcpyD2H
count:0
burst:0
name:cuGetProcAddress
count:0
burst:0

cat ./test/timeline.log
name:cuInit start:1662497122263345556 burst:4213887
name:cuInit start:1662497126572571216 burst:2
name:cuInit start:1662497161782326866 burst:1
name:cuInit start:1662497161970781648 burst:1
name:cuMemcpyH2D start:1662497162147795473 burst:23
name:cuMemcpyH2D start:1662497162269146413 burst:35
name:cuMemcpyH2D start:1662497162281073995 burst:59
name:cuMemcpyH2D start:1662497162285972153 burst:17
...

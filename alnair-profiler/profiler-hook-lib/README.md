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
name:cuInit, start:1662678141402204280, burst:4270333, bytes:0
name:cuInit, start:1662678145746315995, burst:1, bytes:0
name:cuInit, start:1662678179159774310, burst:1, bytes:0
name:cuInit, start:1662678179385822246, burst:2, bytes:0
name:cuMemcpyH2D, start:1662678179542395214, burst:18, bytes:112
name:cuMemcpyH2D, start:1662678179638013798, burst:32, bytes:112
name:cuMemcpyH2D, start:1662678179645375933, burst:35, bytes:112
name:cuMemcpyH2D, start:1662678179650053504, burst:21, bytes:112
name:cuInit, start:1662678379333727700, burst:4304661, bytes:0
name:cuInit, start:1662678383715128389, burst:1, bytes:0
name:cuMemAlloc, start:1662678413097898406, burst:1878, bytes:2097152
name:cuMemAlloc, start:1662678415949164683, burst:1723, bytes:2097152
name:cuMemAlloc, start:1662678415954001022, burst:1714, bytes:20971520
name:cuMemAlloc, start:1662678415958147512, burst:1595, bytes:2097152


cat ./test/timeline.log
name:mem_used,count:421527552
name:cuInit,count:4,burst:6202325
name:cuMemAlloc,count:124,burst:143212
name:cuMemFree,count:14,burst:2
name:cuLaunchKernel,count:537826,burst:4159812
name:cuMemcpyH2D,count:4,burst:92
name:cuMemcpyD2H,count:4,burst:0
name:cuGetProcAddress,count:0,burst:0

...

# CUDA Interpose Library
CUDA version 11.3 and above changes the API to load driver functions when it is called. The hook lib need be changed in order to intercept the driver calls. Here is the demo
codes to enable LD_PRELOAD.

## Quick Start

### Prerequisites
* Build needs to be done on a GPU node.
* The Nvidia driver and CUDA (11.4 and above) toolkit must be installed first.

### Steps

1. Build hook lib
```nvcc -shared -lcuda --compiler-options '-fPIC' hook.cpp -o hook.so
```

2. Build test program
```bash
nvcc -lcuda test.cu -o demo
```

3. Run test
```bash
export LD_PRELOAD=./hook.so
./demo
```

4. Expected result 
>./demo \
testing cudaMalloc... \
cuInitc_hook is called \
cuMemAllocc_hook is called

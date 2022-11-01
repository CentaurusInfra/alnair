

# Build NCCL
## Pre-requirements
```
Requirements
Mellanox ConnectX 6 HCA and Mellanox Quantum IB switch with SHARP support
Nvidia CUDA Toolkit
Mellanox OFED >= 5.0-2.x
Nvidia NCCL >= 2.7.3.
Mellanox HPC-X >= 2.6.2
(Note: HPC-X contains Mellanox SHARP library and latest stable NCCL SHARP plugin)
GPUDirectRDMA driver
More details on GPUDirect RDMA https://www.mellanox.com/products/GPUDirect-RDMA
```


## trouble shooting

### CFLAGS
```
export CFLAGS="-I<...> -I<...>"
```
### LDFLAGS
```
export LDFLAGS="-L<...> -L<...>"
```
# Build nccl-test
## test
```
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```
## trouble shooting
### fatal error: nccl.h: No such file or directory
```
make -C src build BUILDDIR=/home/huide/proj/nvidia/nccl-tests/build
make[1]: Entering directory '/home/huide/proj/nvidia/nccl-tests/src'
Compiling /home/huide/proj/nvidia/nccl-tests/build/verifiable/verifiable.o
../verifiable/verifiable.cu:4:10: fatal error: nccl.h: No such file or directory
 #include <nccl.h>
          ^~~~~~~~
```
#### NCCL_HOME env var
using absolute path below:
```
make NCCL_HOME=/home/huide/proj/nvidia/nccl/build
```
**enable MPI**
```
make MPI=1 NCCL_HOME=/home/dario/nccl/build MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
```

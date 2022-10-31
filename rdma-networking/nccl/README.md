

# Build NCCL

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

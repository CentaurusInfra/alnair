# rdma-over-container

* draft project so far.
* Before enable FreeFlow, the current job is to have RDMA/Pytroch enabled iamge for "distributed" AI training tasks.

* `bin/vRouter.sh` is for triggering running container on the RDMA enabled machine.

# Dev Tools
## conda
https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
```
bash Anaconda-latest-Linux-x86_64.sh
```
### bash env:
bash install will set shell environment through `$HOME/.bashrc`, make sure the `$HOME/.profile` includes below settings:
```
# if running bash
if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
  . "$HOME/.bashrc"
    fi  
fi

# set PATH so cuda bin if it exists
CUDAPATH="/usr/local/cuda"
if [ -d "$CUDAPATH/bin" ] ; then
    PATH="$PATH:$CUDAPATH/bin"
fi
```

###  creat conda env
```
conda activate base
conda create --name targetEnv python=3.8
```

## python3.9 on host

### apt installation
```
sudo add-apt-repository ppa:deadsnakes/ppa 
sudo apt update 
sudo apt install python3.9 
```

## CUDA
### locla mode
```
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.run
sudo sh cuda_11.4.0_470.42.01_linux.run
```
## rdma user lib
```
git clone git://github.com/rdma-core

follow the README.md
```


# Driver prepareation
## other
https://github.com/CentaurusInfra/alnair/wiki/GPU-node-preparation#gpu-node-preparation

## cudnn
If you hit GPG keyring problem and can solve it smoothly, here's a workaround to install tarball directly:

cudnn download web: https://developer.nvidia.com/cudnn You need an account before download.
```
 tar xvf cudnn*.tgz
 cd cudnn-linux-x86_64.../
 sudo cp inluce/*.h /usr/local/cuda/include/
 sudo cp lib/libcudnn* /usr/local/cuda/lib64/
 sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```
### Version verify:
```
$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
or 
$ cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

```

## RDMA OFED  
Kernel modules and RDMA tools included. 
#### driver download
https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
#### UBU20.04 LTS
https://content.mellanox.com/ofed/MLNX_OFED-4.9-5.1.0.0/MLNX_OFED_LINUX-4.9-5.1.0.0-ubuntu20.04-x86_64.tgz

```
tar xzf 
sudo mlnxofedinstall --all
```

#### driver installation
https://docs.nvidia.com/networking/display/MLNXOFEDv461000/Installing+Mellanox+OFED#InstallingMellanoxOFED-InstallationScript
#### rdma-core (user space tools)
This should be included in OFED installation package, in case missing, `rdma-core` can be built separately. 

# Libraries
## pytorch (binary)
https://pytorch.org/get-started/locally/
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## openmpi
### apt install (cli ref)
```
sudo apt install openmpi-bin openmpi-common
```
The following will be installed when installing `openmpi-bin`:
```
 libfabric1 libhwloc-plugins libhwloc5 libopenmpi2 libpsm-infinipath1 ocl-icd-libopencl1 openmpi-bin openmpi-common
 
```
### conda install (cli ref)
```
conda activate <your env>
conda install -c conda-forge openmpi<=[version info]>
```
### Build from source:
```
git clone git://github.com/...
cd ompi
git checkout tags/<tag_name> -b <branch_name>
git submodule update --init --recursive
./autogen.pl
mkdir build && cd build
../configure --prefix=/usr/local --with-cuda --enable-mpi-thread-multiple
```
or with more debug info
```
../configure --prefix=/usr/local --with-cuda --enable-mpi-thread-multiple --enable-debug 
../configure --prefix=/usr/local --with-cuda --enable-mpi-thread-multiple --enable-debug --enable-mem-debug --enable-event-debug ?
```
```
make -j32 all
sudo make install
```
#### in my case:
```
mkdir build && cd build && ../configure --prefix=/usr/local --with-cuda --enable-mpi-thread-multiple --enable-debug --enable-mem-debug --enable-event-debug  && make -j 32 all && sudo make install 
```
## pytorch with openmpi (build ompi first)
if runing in conda, suggest to create a new env with `conda create --name targetEnv python=3.8` 

```
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# read references section for setuptools installation first
# conda install numpy pyyaml mkl cmake cffi setuptools 
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# version need to fit your goal
conda install -c pytorch magma-cuda102 
```
##### Notes: for submodules and specific version
`git clone --recursive git://github.com/pytorch...`

`git clone --recursive --branch v1.7.0 http://github.com/pytorch/pytorch`

```
git clone git://github.com/pytorch...
cd pytorch
git checkout tags/v1.8.2 -b b1.8.2
git submodule sync
git submodule update --init --recursive --jobs 0

python setup.py install 
```

### torchvision
```
git clone github.com:pytorch/vision.git
cd vision
git checkout tags/v0.9.2 -b b0.9.2
python setup.py install
```


on your filesystem directly:
```
pip3 install  numpy pyyaml mkl setuptools cmake cffi

```
if you hit below when `pip3 install cmake`:
```
    ModuleNotFoundError: No module named 'skbuild'
```
try:
```
pip3 install --upgrade pip
```
some potential packages may need:
```
apt install python3.9-distutil
```
# References
## Driver
https://github.com/mjiUST/driver_cuda_cudnn

## dependencies reference:
```
python >= 3.8 
pytorch==1.7.1+cu101
torchvision == 0.8.2 + cu101
tensorboard == 2. 4.0
numpy == 1.21.0
opencv-python == 4.6.0
pillow==9.2.0
```
## know issues in `setuptools`
### setuptools has no 'version' attribute
```
AttributeError: module 'setuptools._distutils' has no attribute 'version'
```
soltuion:
```
either use the nightly-release of PyTorch, or otherwise downgrade setup tools to setuptools version 59.5.0.
I install even smaller version.
```
### ninja build stoped: submodule failed
big hole here, took me some time . 
```
git clean -xdf
python setup.py clean
git submodule sync
git submodule deinit -f .
git submodule update --init --recursive
python setup.py install
```

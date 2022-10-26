
## OFED issues:

### Failed to install mlnx-ofed-kernel-dkms DEB

#### error message
```
Removing old packages...
Uninstalling the previous version of MLNX_OFED_LINUX
Installing new packages
Installing ofed-scripts-4.9...
Installing mlnx-ofed-kernel-utils-4.9...
Installing mlnx-ofed-kernel-dkms-4.9...
Failed to install mlnx-ofed-kernel-dkms DEB
Collecting debug info...
See /tmp/MLNX_OFED_LINUX.3031997.logs/mlnx-ofed-kernel-dkms.debinstall.log
```

#### kernel on target host
```
(p27) huide@titan35:~/proj/rdma/OFED/MLNX_OFED_LINUX-4.9-5.1.0.0-ubuntu20.04-x86_64$ uname -sr
Linux 5.15.0-48-generic
```

#### log analyzing:

`/var/lib/dkms/mlnx-ofed-kernel/4.9/build/make.log`
```
...
CONFIG_MLX5_ESWITCH not support kernel version 5.6 or higher (current: 5.15.0-48-generic)
...
```


## OFED references
- driver download
https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
- release notes
https://docs.nvidia.com/networking/display/MLNXOFEDv543580/General+Support

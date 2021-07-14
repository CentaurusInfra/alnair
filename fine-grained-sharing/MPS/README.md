# Multi-Process Service 

Nvidia Reference [Doc](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)

AWS reference MPS-based GPU plugin [blog](https://aws.amazon.com/blogs/opensource/virtual-gpu-device-plugin-for-inference-workload-in-kubernetes/), [github](https://github.com/awslabs/aws-virtual-gpu-device-plugin)
## Overall 
- One user at a time
- Multiple applications (clients)  from the same user
- No. of applications <= 48
- Use ```CUDA_MPS_ACTIVE_THREAD_PERCENTAGE``` to limit the resource usage, memory limits are configured from user scripts 
- Support 64 bits application only
- One application fails could affect others

## MPS commands
- Enable MPS for one single GPU
```
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d
```
**Note: don't add sudo for the last command. MPS is user sensitive, if server and client application are not launched from the same user, GPUs are not found.**
- Disable MPS for one single GPU
```
echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c 0
```
- Verify MPS daemon is running
```ps -ef | grep mps```

- Enable MPS for multiple GPUs (4 GPUs example)
```
sudo nvidia-smi -i 0,1,2,3 -c EXCLUSIVE_PROCESS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
export CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-cuda-mps-control -d
```
- Disable MPS for multiple GPUs
```
echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -i 0,1,2,3 -c 0
```
**One MPS server per GPU will be lauched. In the above example, 4 mps-server processes can be observed by ```nvidia-smi```**
## MPS for standalone applications
After MPS is enabled on the host, just lauch the applicaiton with no extra configuration, e.g. ```python3 XXX.py```

**Note: make sure the user launched MPS is the same who runs the application**

## MPS for containerized applications
TO access the MPS server on the host, docker run requires **"--ipc=host"** option. So a complete command could be as following

```sudo docker run --ipc=host --rm --gpus device=0 --name=dlt -d centaurusinfra/dlt-job```

Other options like mount the mps folder ```-v /tmp/nvidia-mps:/tmp/nvidia-mps```, or ```--runtime=nvidia``` are not necessary at least for TitanX GPU.
Assume docker default daemon is already configured to nvidia in /etc/docker/daemon.json file
```
{
    "default-runtime":"nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}

```
**Note: since docker container(process) by default is run as root, to make MPS server visible inside container, the MPS daemon must be launched by root. Or run container as the user launches MPS**

## Memory limits configuration within ML Framework
- Tensorflow

```tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])```
- Pytorch

```torch.cude.set_memory_per_process_fraction(0.5)```
## Common issues
- No GPU found, control.log show "User did not send valid credentials"

Reason: the user started nvidia-cuda-mps-control daemon is not the same who launched docker container.
If the docker is lauched by default (root), to connect to MPS server from the container, the mps service must be launched by "root".
- Tensorflow cannot find GPU, could be some cuda libs not installed, e.g. libcudnn8, or need to link some newer version to a specific version Tensorflow requires
- Cannot launch pytorch scripts twice, error: address is already in use
Reason: os.environ['MASTER_PORT'] = 'XXXX', one port per process.
Use different ports to run multiple scripts.
- Processes interference causes MPS server freeze, could use profiler detects gpu/memory abnormal utilization.


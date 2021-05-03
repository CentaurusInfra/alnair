# DCGM(data center GPU management)
A Nvidia GPU monitoring tool
Quick Start refer to Nvidia's [github](https://github.com/NVIDIA/gpu-monitoring-tools) and [documents](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/dcgm-exporter.html)

dcgm-export container can be easily lauched by the following commands


```sudo docker run --name=dcgm --cap-add SYS_ADMIN -d -e DCGM_EXPORTER_INTERVAL=100 --gpus all --rm -p 9400:9400 nvidia/dcgm-exporter:2.0.13-2.1.1-ubuntu18.04```


Environment variables can be used to congifure DCGM, like ```DCGM_EXPORTER_INTERVAL``` is configured to 100ms in the above. More configurations can be found by ```dcgm-exporter --help```.
Evnironment variables configuration can be verified by
```docker exec dcgm printenv```

Once the container is launched, metrics can be scraped by
```curl http://localhost:9400/metrics```

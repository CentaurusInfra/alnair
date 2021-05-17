# Device-level GPU metrics collector
DCGM (data center GPU management) is a Nvidia GPU monitoring tool.
Quick Start refer to Nvidia's [github](https://github.com/NVIDIA/gpu-monitoring-tools) and [documents](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/dcgm-exporter.html)

dcgm-export container can be easily lauched by the following commands. Make sure docker and Nvidia driver version is up-to-date on the host.

```sudo apt-get install -y nvidia-container-toolkit```

```sudo docker run --name=dcgm --cap-add SYS_ADMIN -d -e DCGM_EXPORTER_INTERVAL=100 --gpus all --rm -p 9400:9400 nvidia/dcgm-exporter:2.0.13-2.1.1-ubuntu18.04```


Environment variables can be used to congifure DCGM, like ```DCGM_EXPORTER_INTERVAL``` is configured to 100ms in the above. More configurations can be found by ```dcgm-exporter --help```.
Evnironment variables configuration can be verified by
```docker exec dcgm printenv```

Once the container is launched, metrics can be scraped by
```curl http://localhost:9400/metrics```

Default exported metrics are configured in ```/etc/dcgm-exporter/default-counters.csv``` in the container. Custom metrics file can be passed by ```-f``` option. Complete avaliable metrics/fields can be found [here](https://docs.nvidia.com/datacenter/dcgm/1.6/dcgm-api/group__dcgmFieldIdentifiers.html).

If deploy in a Kubernetes cluster, [here](https://github.com/CentaurusInfra/AI-SIG/blob/main/dcgm-gpu-monitoring/dcgm-pod.yaml) is a basic yaml file, and [this](https://developer.nvidia.com/blog/monitoring-gpus-in-kubernetes-with-dcgm/) Nvidia blog on monitoring GPUS in Kubernetes with DCGM providers more information.

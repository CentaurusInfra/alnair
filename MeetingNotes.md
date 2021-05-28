# AI-SIG Weekly Meeting Notes
## 2021-05-26
- Kubecon US proposal submission
  - Title: GPU Profiling and Elastic Allocation for AI jobs in Kubernetes ([link](https://docs.google.com/document/d/1hrNXUYZDlMz6518pqLD0KawGo_3FjoCG0-rvuE4tvpw/edit)) 
- Profiler
  - Bug fix on annotation removal functions
  - Compare multiple GPUs and single GPU utilization for the same workload, based on Tensorflow's all reduce distributed strategy
- Elastic Training
  - Implement GPU allocator for the elastic training framework
  - Setup demo environment
- Big Model
  - Further discuss big model training solutions: data/model/pipeline parallelism, compression, specific hardware

## 2021-05-19
- Profiler
  - Check in module's 1st version, profiler and dcgm are deployed together in the same pod for easy Pod IP based query in Prometheus
  - Discuss parameter based memory utilization estimation before job execution
- Elastic Training
  - Design horovod operator and GPU allocator with kubebuilder
- Big Model
  - Big Model status, why, what, how, who, ([slides](https://github.com/CentaurusInfra/AI-SIG/blob/main/reference/BigModels.pdf))
## 2021-05-12
- Profiler
  - Demostrate basic functions: profiler updates node annotations (GPU job-type) based on workloads memory utilization
- Elastic Training
  - Use headless service to discover worker IP
  - Raise issues of horovod: deployment randomly crashes when scale down
## 2021-05-05
- Profiler/Telemetry
	- Continue to analyze deep learning workloads memory usage behavior.
	  - Confirmed both Pytorch and Tensorflow's memory usage has cyclic behavior associated with Epoch, for both small dataset (CIFAR10) and big dataset (ImageNet)
	  - Step level seasonality is not observable based on current 1Hz sampling rate, 3-4 steps happend in 1 second with current GeForce RTX GPU
	- Complete basic framework of data collection, analysis and results tagging in Kubernetes.

- Elastic Training
	- Demonstrate horovod deployment in K8s, worker communication through sshkey secretes.
	- Design horovod based elastic training job CRD and operator.

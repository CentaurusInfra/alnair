# AI-SIG Weekly Meeting Notes
## 2021-06-23
- Investigate current AI platform with different focus, e.g., IDE(SageMaker), libarary(Tensorflow), pararell framework(Horovod), distributed computing(Ray), Data processing(Spark), Kubernetes Eco-system(Kubeflow, Volcano). Position Alnair.
- Connect with TU Wien on cloud resource usage prediction research, focus on transformer and LSTM algoritms for time series forecasting
- Setup GPU cluster in fw-corp network
- Implement e2e test in elastic-training framework to test cluster launch results
- Preliminary test on Nvidia Multi-Process Service (MPS) for GPU sharing
- Setup AWS account for GPU instances/services and SageMaker trial
- Attend CVPR and Ray summit
## 2021-06-16
- Setup Jenkins CI flow ([project link](https://jenkins.alkaidcloud.io/job/alnair/)), kubebuilder test investigation
- Sort out next release features
  - Resource sharing within one GPU (GPU virtualization, MPS, CUDA call intercept)
  - GPU allocation algorithms (learning capability, continuous optimization)
  - Close loop: profiling-allocation-sharing
  - previous release improvement (profiler, elastic-training framework) 
    - profiler: process level monitoring, usage prediction
    - elastic-training framework: GPU type/topology awareness, multiple ML framework support (e.g pytorch)
- Discuss spot instance features in public clouds, possible GPU application
- CVPR video preparation (Futurewei AI activities intro)
- Submit proposal for Edge AI to OSS
- Attend IAP workshop
## 2021-06-09
- Pick project name **Alnair**(the brightest star in constellation Grus) and tag the first release 
- Cluster reconfigure within the team
- Dissucss global scheduler design, k8s scheduler modification (queue removal), synchronize batch scheduling to reduce latency.
- Next release feature discussion (fine-grained sharing, automomous scheduler, continuous gpu allocation optimization)
- CI setup requirements (Jenkins)
## 2021-06-02
- Profiler
  - Profiling different scenarios, local and remote dataset, different types and numbers of GPUs
  - Fix Prometheus scraping issue (context deadline exceed), due to scrape duration too long
- Elastic Training
  - Demo elastic training framework, including increase/decrease GPU counts for a job when more or less resources avaliable, fault tolerance when some GPU is not avaliable, etc.
  - Fix GPU resource release issue after job is done.
- Big Model
  - Discuss PanGu and transformer principle 
  - Investigate model paralellism in PyTorch and TorchElastic
## 2021-05-26
- KubeCon 2021 NA proposal submission
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

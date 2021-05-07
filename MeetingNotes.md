# AI-SIG Weekly Meeting Notes
## 2021-05-05
- Profiler/Telemetry
	- Continue to analyze deep learning workloads memory usage behavior.
	  - Confirmed both Pytorch and Tensorflow's memory usage has cyclic behavior associated with Epoch, for both small dataset (CIFAR10) and big dataset (ImageNet)
	  - Step level seasonality is not observable based on current 1Hz sampling rate, 3-4 steps happend in 1 second with current GeForce RTX GPU
	- Complete basic framework of data collection, analysis and results tagging in Kubernetes.

- Elastic Training
	- Demonstrate horovod deployment in K8s, worker communication through sshkey secretes.
	- Design horovod based elastic training job CRD and operator.

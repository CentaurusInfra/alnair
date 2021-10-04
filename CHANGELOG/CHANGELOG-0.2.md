# Release Summary

This is Release v0.2. It includes improvements on Profiler and Elastic Framework, and the following new components:

- Autonomous Scheduler
- Alnair Device Plugin


# Key Features and Improvements:
## Profiler (imporvemnts)
  - CPU metrics collection
    - Take advantage of cAdvisor, pod-level metrics, e.g. CPU and Memory utilization, disk io and network utilization are collected every second.
  - GPU metrics mapping
    - The process IDs on GPUs are extracted using nvml library and they are mapped to Pod name. Therefore the GPU utilization resolution is improved from node level to pod level.
  - Resource utilization aggregation 
    - The pods created by kubernetes jobs and job-like CRDs are auto deleted after job is complete, the annotations on pods will be lost. By using Pod's owner reference, Profiler automatic aggreates the Pods/workers' information to they owner (Jobs/CRDs). The max utilization of each metrics are recorded in the annotations at Job level. This is for future job execution efficiency analysis. 

## Elastic Training Framework (improvements)
  - GPU allocation auto reduction
    - Due to race condition, when the number of GPUS set by TargetReplica is not avaliable at scheduling phase, elastic horovod job controller will auto scale down the size of StatefulSet by one.
  - PodGroup Integration 
    - Leverage coscheduling plugin, create and assign PodGroup for each elastic horovod job, and launch the workers(StatefulSet) as a PodGroup, i.e., the workers run in an all-or-nothing manner to avoid resource starvation.  
  - Improve scaling speed
    - Updated podManagementPolicy field in the StatefulSet from default (OrderedReady) to Parallel to reduce scaling-up/down time. All Pods can be launched or terminated in parallel. No need to wait for predecessor to become Running and Ready or completely terminated.

## Autonomous Scheduler (new)
  - An utilization-driven scheduler: UtilSched
    - UtilSched is a customized kubernetes scheduler based on the k8s scheduling framework. It optimizes the scheduling strategy of AI workloads (with GPUs invoked), by aggregating the real-time GPU metrics that the above Profiler module extracts. By leveraging the APIs provided by the k8s scheduling framework, UtilSched works as a plugin that optimizes the scheduling decisions by only being invoked at several extension points (such as the Filter and Score process) without interrupting the core of k8s scheduling.
  - Co-scheduling feature
    - Co-scheduling feature is to ensure the atomicity of a group of pods being scheduled together. The default k8s scheduler, under certain scenes (e.g. race conditions), cannot schedule the pods of a batch workload that are spawned by a statefulset or deployment, which causes a heavy waste of resource. By introducing a PodGroup CRD, a batch scheduling will be masked failed once a pod in the PodGroup failed. A controller is designed to reconcile PodGroup status and help recover from abnormal cases. The co-scheduling feature is based on the k8s scheduling framework. Cooperating with the above elastic-training-framework module, it alleviates the potential race conditions in an elastic scale down/up process.
 
## Alnair Device Plugin (new)
  - Kubernetes Device Plugin for Nvidia GPU
  - Support Fractional GPU resources request and allocation

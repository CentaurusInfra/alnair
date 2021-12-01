# Autonomous scheduler

## Initial GPU Allocator 

When an ElasticHorovodJob is first submitted by the user, the [Initial GPU allocator](../elastic-training/controllers/scheduler.go) will calculate the most suitable number of GPUs the job should use. Afterwards, the implemented scheduler endpoints will be invoked as per the Kubernetes [scheduling-framework](https://github.com/kubernetes/enhancements/blob/master/keps/sig-scheduling/20180409-scheduling-framework.md). 

## Coscheduling 

Coscheduling is used to ensure atomicity in worker pods — that is, if one or more of the pods spawned by a StatefulSet, Deployment, or other similar Resources cannot pass through due to race conditions or other issues, the scheduler will reject the other pods that belong to the same PodGroup. In other words, it is an all-or-nothing approach. 

This coscheduling implementation is based on the Kubernetes-SIGs [scheduler-plugins](https://github.com/kubernetes-sigs/scheduler-plugins) repository. 

## UtilSched

UtilSched is a utilization-aware scheduler built upon the Scheduling Framework. As a sub-project of [Alnair](https://github.com/CentaurusInfra/alnair), it can cooperate with the [profilling module](https://github.com/CentaurusInfra/alnair/tree/main/profiling).
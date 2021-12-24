# High Level Design 

## Background and Motivation

"Modern distributed machine learning (ML) training workloads benefit significantly from leveraging GPUs. However, significant contention ensues when multiple such workloads are run atop a shared cluster of GPUs. A key question is how to fairly apportion GPUs across workloads. We find that established cluster scheduling disciplines are a poor fit because of ML workloads’ unique attributes: ML jobs have long-running tasks that need to be gang-scheduled, and their performance is sensitive to tasks’ relative placement" - Mahajan et. al, *Themis: Fair and Efficient GPU Cluster Scheduling* 

Thus, a cluster manager must be able to acertain a specific GPU configuration (across nodes) for a given workload, which we will call ```UnifiedJob```.

## Scheduler

The Scheduler is a Kubernetes [Controller](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) that watches over deep learning jobs and enforces decisions by updating ```UnifiedJob.Spec.ReplicaSpec.TargetReplicas```. While its loop function is triggered by a change in the UnifiedJob object itself, it is able to maintain a global view of the cluster resources as well as UnifiedJobs. 

The Scheduler is explored more in [scheduler.md](scheduler.md) and the code is located in [controllers/unified_scheduler.go](../../controllers/unified_scheduler.go). 

## Unified Job 

The Scheduler should be able to create global decisions on any workload despite it's job type (JobType), whether it be a single-node job that can be run natively through [Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) or a cross-node solution that requires a special CRD and controller such as [PyTorch Elastic](https://pytorch.org/docs/stable/distributed.elastic.html). 

Thus, we propose the ```UnifiedJob``` [Custom Resource Defintion](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) (CRD). This is a CRD that allows for flexible JobTypes (provided a controller for that JobType), and allows for the scheduler to simply look at the UnifiedJob scheduler. 

The UnifiedJob CRD is explored more in [unifiedjob-CRD.md](unifiedjob-CRD.md), and the Golang API is located in [api/v1alpha1/unifiedjob_types](../../api/v1alpha1/unifiedjob_types.go) and the Kubernetes CRD declaration can be found in [config/crd/bases/ai.centauruscloud.io_unifiedjobs.yaml](../../config/crd/bases/ai.centauruscloud.io_unifiedjobs.yaml). 

## Controller

The Kubernetes Controller for a UnifiedJob must be able to run the job based on the ```UnifiedJob.Spec.ReplicaSpec.TargetReplicas``` that is updated by the Scheduler. This Controller must be able to manage the job's resources accordingly while acocunting for the large range of JobTypes that are possible. 

The UnifiedJob controller is explored more in [operator.md](operator.md) and the code is located in [controllers/unifiedjob_controller.go](../../controllers/unifiedjob_controller.go). 
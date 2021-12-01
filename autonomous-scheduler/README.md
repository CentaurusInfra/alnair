# Autonomous scheduler

## Introduction
To achieve the [Alnair project goals](https://github.com/CentaurusInfra/alnair), we need to customize the Kubernetes default scheduler so that the proposed platform with higher efficiency and intelligence can be implemented on top of the Kubernetes infrastructure:

1) Cooperating with the [Alnair/elastic-training module](https://github.com/CentaurusInfra/alnair/tree/main/elastic-training), the scheduler alleviates the race condition in a scale up/down process of elastic training. The feature is mainly implemented via the co-scheduling plugin. 

2) Cooperating with the [Alnair/fine-grained-sharing module](https://github.com/CentaurusInfra/alnair/tree/main/fine-grained-sharing), the scheduler allocates the GPU resource requests according to ''amount of GPU memory in MiB'' rather than ''number of GPUs''. Plus, the scheduling policy needs to take ''the topology of GPUs'' into the consideration, and this feature is mainly implemented via the NodeAffinity plugin. 

3) Cooperating with the [Alnair/profilling module](https://github.com/CentaurusInfra/alnair/tree/main/profiling), the scheduler assigns the GPU-related Pods to suitable Nodes according to the current cluster utilization. This feature is mainly implemented via the UtilSched plugin. 

The proposed autonomous scheduler are based on [the scheduling-framework](https://github.com/kubernetes/enhancements/blob/master/keps/sig-scheduling/20180409-scheduling-framework.md) of Kubernetes. The APIs of scheduling-framework allow most scheduling features to be implemented as plugins, while keeping the scheduling core more maintainable. As shown in the diagram, the framework defines a few extension points in both the scheduling cycle and the binding cycle. Our design of plugins are registered and invoked at one or more extension points to change the scheduling decisions, respectively. 

![Diagram](./UtilSched/img/diagram.png)

### Initial GPU Allocator 

Initial GPU allocator is a component in our [elastic-training module](../elastic-training/controllers/scheduler.go). Once an ElasticHorovodJob is first submitted by the user, the initial GPU allocator will calculate the most suitable number of GPUs the job should use. Then, the implemented scheduler endpoints will be invoked as per the Kubernetes scheduling-framework. 

### Co-Scheduling Plugin

Currently, through the default scheduler of Kubernetes, we cannot ensure a group of pods can be scheduled altogether. Under some scenes, it would waste resources since the whole application cannot work with only partial Pods' running, like Spark jobs, TensorFlow jobs, and so on. 

In our co-scheduling plugin, the atomicity is ensured in worker pods — that is, if one or more of the pods spawned by a StatefulSet, Deployment, or other similar Resources cannot pass through due to race conditions or other issues, the scheduler will reject the other pods that belong to the same PodGroup. In other words, it is an all-or-nothing approach. 

This co-scheduling implementation is based on the Kubernetes-SIGs [scheduler-plugins](https://github.dcom/kubernetes-sigs/scheduler-plugins) repository. 

### NodeAffinity Plugin

This plugin enables scheduling decisions based on worker node hardware topology. Ideally, the scheduler should not try to place a pod onto a node where certain topology policy is violated (e.g., a single NUMA node is requested). Currently, because the default scheduler of Kubernetes is not topology-aware, once the topology affinity error happens and pod fails, the ReplicaSet will create another pod, repeat the error again, and waste CPU time constantly.  

The proposed NodeAffinity plugin leverages the NodeResourceTopology CRD instance corresponding to the nodes to obtain the resource topology information to make a topology-aware scheduling decision. This co-scheduling implementation is based on the Kubernetes-SIGs [scheduler-plugins](https://github.dcom/kubernetes-sigs/scheduler-plugins) repository. 

### UtilSched Plugin

As a sub-project of [Alnair](https://github.com/CentaurusInfra/alnair), it can cooperate with the [profilling module](https://github.com/CentaurusInfra/alnair/tree/main/profiling) to schedule GPU tasks according to the current cluster utilization. The goal is to make the Kubernetes scheduler aware of the gap between resource allocation and actual resource utilization and pack pods more efficiently. 

As discussed in the [profilling module](https://github.com/CentaurusInfra/alnair/tree/main/profiling), profiler results are written into cluster nodes' annotations. With ```kubectl describe node <your-node-name> | grep ai.centaurus.io```, we can find the metrics it collected including GPU architecture, GPU memory size, GPU utilization rate, I/O bandwidth, and so on. By default, profiler will update annotations every 30 seconds.

With the above collected utilization metrics, the UtilSched plugin will aggregate them via a customized score function which includes three components as shown in the equation. The basic score indicates a static measure of a specific node. The utilization score measures the real-time utilization rate of nodes. The allocated score aggregates the metrics information from the dry-run process of a Pod.  

![Diagram](./UtilSched/img/score_function.png)

## Install

### Create a Kubernetes Cluster

Firstly you need to have a Kubernetes cluster, and a kubectl command-line tool must be configured to communicate with the cluster. The Kubernetes version must equal to or greater than v1.20.0. To check the version, use kubectl version --short.

### Install and Use as Single Scheduler

Although there are two ways to install the scheduler-plugin artifacts: as a second scheduler and as a single scheduler. We here only provide the single scheduler installation procedure which needs some manual steps. 

The major advantage of using a unified scheduler is to keep the resource conflicting free. As the new image is built on top of the default scheduler in Kubernetes, you won't lose any scheduling capability. Instead, a lot of extra out-of-box functionalities can be obtained. 

The following steps are based on a Kubernetes cluster created by Kind. Plus, the configuration yaml file takes the co-scheduling plugin as an example.  

1. Log into the control plane node 
   
    ```bash
    sudo docker exec -it $(sudo docker ps | grep control-plane | awk '{print $1}') bash
    ```
  
1. Backup `kube-scheduler.yaml`

   ```bash
   cp /etc/kubernetes/manifests/kube-scheduler.yaml /etc/kubernetes/kube-scheduler.yaml
   ```

1. Create `/etc/kubernetes/sched-cc.yaml`

    ```yaml
    apiVersion: kubescheduler.config.k8s.io/v1beta1
    kind: KubeSchedulerConfiguration
    leaderElection:
      # (Optional) Change true to false if you are not running a HA control-plane.
      leaderElect: true
    clientConnection:
      kubeconfig: /etc/kubernetes/scheduler.conf
    profiles:
    - schedulerName: default-scheduler
      plugins:
        queueSort:
          enabled:
          - name: Coscheduling
          disabled:
          - name: "*"
        preFilter:
          enabled:
          - name: Coscheduling
        postFilter:
          enabled:
          - name: Coscheduling
        permit:
          enabled:
          - name: Coscheduling
        reserve:
          enabled:
          - name: Coscheduling
        postBind:
          enabled:
          - name: Coscheduling
      # pluginConfig is needed for coscheduling plugin to manipulate PodGroup CR objects.
      pluginConfig:
      - name: Coscheduling
        args:
          kubeConfigPath: /etc/kubernetes/scheduler.conf
    ```

    It is noticed that 1) queueSort, permit and unreserve must be enabled in coscheduling, and 2) preFilter is an optional feature according to the actual situation of users. 

1. The proposed plugins introduced CRD to optimize their design and implementation. And hence we need an extra step to:

    - apply extra RBAC privileges to user `system:kube-scheduler` so that the scheduler binary is
      able to manipulate the custom resource objects
    - install a controller binary managing the custom resource objects

    Next, we apply the compiled yaml located at [manifests/install/all-in-one.yaml](/plugins/manifests/install/all-in-one.yaml).

    ```bash
    $ kubectl apply -f plugins/manifests/install/all-in-one.yaml
    ```

    After this step, a deployment called `scheduler-plugins-controller` is expected to run in
    namespace `scheduler-plugins`:

    ```bash
    $ kubectl get deploy -n scheduler-plugins
    NAME                           READY   UP-TO-DATE   AVAILABLE   AGE
    scheduler-plugins-controller   1/1     1            1           19h
    ```

1. Install the CRDs your workloads depend on.

    You can refer to each folder under [manifests](plugins/manifests) to obtain the CRD yaml for each
    plugin. Here we install coscheduling CRD:

    ```bash
    $ kubectl apply -f manifests/coscheduling/crd.yaml
    ```

1. Modify `/etc/kubernetes/manifests/kube-scheduler.yaml` to run scheduler-plugins. Generally, we need to make a couple of changes:
    
    - pass in the composed scheduler-config file via argument `--config`
    - (optional) remove duplicated CLI parameters (e.g., `--leader-elect`), as they may have been defined in the config file
    - replace vanilla Kubernetes scheduler image with scheduler-plugin image
    - mount the scheduler-config file to be readable when scheduler starting
    
    Now, you can verify that kube-scheduler pod is running properly with a correct image: `k8s.gcr.io/scheduler-plugins/kube-scheduler:v0.20.10`

    ```bash
    $ kubectl get pod -n kube-system | grep kube-scheduler
    kube-scheduler-kind-control-plane            1/1     Running   0          3m27s
 
    $ kubectl get pods -l component=kube-scheduler -n kube-system -o=jsonpath="{.items[0].spec.containers[0].image}{'\n'}"
    k8s.gcr.io/scheduler-plugins/kube-scheduler:v0.20.10
    ```
   
## Demo

Here is the demo for testing the co-scheduling plugin. The procedure for testing the other proposed plugins will be added soon.  
1. Create a PodGroup custom object called `pg1`:

    ```yaml
    # podgroup.yaml
    apiVersion: scheduling.sigs.k8s.io/v1alpha1
    kind: PodGroup
    metadata:
      name: pg1
    spec:
      scheduleTimeoutSeconds: 10
      minMember: 3
    ```

    ```bash
    $ kubectl apply -f podgroup.yaml
    ```

1. Create a deployment labelled `pod-group.scheduling.sigs.k8s.io: pg1` to associated with PodGroup
   `pg1` created in the previous step.

    ```yaml
    # deploy.yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: pause
    spec:
      replicas: 2
      selector:
        matchLabels:
          app: pause
      template:
        metadata:
          labels:
            app: pause
            pod-group.scheduling.sigs.k8s.io: pg1
        spec:
          containers:
          - name: pause
            image: k8s.gcr.io/pause:3.2
    ```
   
1. As PodGroup `pg1` requires at least 3 pods to be scheduled all-together, and there are only 2 Pods
   so far, so it's expected to observer they are pending:

    All nginx pods are expected to be `Pending` as they cannot be co-scheduled altogether.

    ```bash
    $ kubectl get pod
    NAME                     READY   STATUS    RESTARTS   AGE
    pause-58f7d7db67-7sqgp   0/1     Pending   0          9s
    pause-58f7d7db67-jbmfv   0/1     Pending   0          9s
   ```

1. Now let's delete the deployment to re-create it with replicas=3, so as to qualify for `minMember`
   (i.e., 3) of the associated PodGroup:

    ```bash
    $ kubectl delete -f deploy.yaml && sed 's/replicas: 2/replicas: 3/' deploy.yaml | kubectl apply -f -
    deployment.apps "pause" deleted
    deployment.apps/pause created
    ```

    And wait for a couple of seconds, it's expected to see all Pods get into running state:

    ```bash
    $ kubectl get pod
    NAME                     READY   STATUS    RESTARTS   AGE
    pause-64f5c9ccf4-kprg7   1/1     Running   0          8s
    pause-64f5c9ccf4-tc8lx   1/1     Running   0          8s
    pause-64f5c9ccf4-xrgkw   1/1     Running   0          8s
    ```

1. You can also get the PodGroup's spec via:

    ```bash
    $ kubectl get podgroup pg1 -o yaml
    apiVersion: scheduling.sigs.k8s.io/v1alpha1
    kind: PodGroup
    metadata:
      annotations:
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"scheduling.sigs.k8s.io/v1alpha1","kind":"PodGroup","metadata":{"annotations":{},"name":"pg1","namespace":"default"}, "spec":{"minMember":3,"scheduleTimeoutSeconds":10}}
      creationTimestamp: "2021-08-17T19:20:08Z"
      generation: 1
      managedFields:
      ...
      name: pg1
      namespace: default
      resourceVersion: "135603"
      selfLink: /apis/scheduling.sigs.k8s.io/v1alpha1/namespaces/default/podgroups/pg1
      uid: b4ac3562-54ab-4c1e-89bb-541a81c6acce
    spec:
      minMember: 3
      scheduleTimeoutSeconds: 10
    status:
      failed: 0
      phase: Running
      running: 3
      scheduleStartTime: "2021-08-17T19:20:58Z"
      scheduled: 3
      succeeded: 0
    ```
   
## Community, discussion, contribution, and support

Learn how to engage with the Kubernetes community on the [community page](http://kubernetes.io/community/).

You can reach the maintainers of this project at:

- [Slack](https://kubernetes.slack.com/messages/sig-scheduling)
- [Mailing List](https://groups.google.com/forum/#!forum/kubernetes-sig-scheduling)

You can find an [instruction how to build and run out-of-tree plugin here](doc/develop.md) .

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
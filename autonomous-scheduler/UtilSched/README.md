# UtilSched: A Cluster-`Util`ization-Driven K8S `Sched`uler 

UtilSched is a kubernetes scheduler based on [scheduling-framework](https://github.com/kubernetes/enhancements/blob/master/keps/sig-scheduling/20180409-scheduling-framework.md). As a sub-project of [Alnair](https://github.com/CentaurusInfra/alnair), it can cooperate with the [profilling module](https://github.com/CentaurusInfra/alnair/tree/main/profiling) to schedule GPU tasks according to the current cluster utilization. 

The later version will also support: 
1) Cooperating with the [elastic-training module](https://github.com/CentaurusInfra/alnair/tree/main/elastic-training) to alleviate race conditions in a scale up/down process. 
2) Cooperating with the [fine-grained-sharing module](https://github.com/CentaurusInfra/alnair/tree/main/fine-grained-sharing) to schedule GPU tasks with granularity less than 1. 

### Diagram of K8S scheduling-framework and our design
![Diagram](./img/diagram.png)
----

## Quick Start
- Make sure kubernetes cluster version is `1.17+`
- Deploy utilsched scheduler
```shell
kubectl apply -f https://raw.githubusercontent.com/YHDING23/UtilSched/master/deploy/utilsched.yaml
```
- Check the scheduler status
```shell
kubectl get pods -n kube-system 
```
## Develop from UtilSched
- Compile:
```shell
make local
```
- Build the docker image:
```shell
make build
```
- Clean the Build file
```shell
make clean
```







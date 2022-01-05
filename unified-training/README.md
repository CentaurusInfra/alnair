# Elastic Training Framework
The elastic training framework is built upon [CentaurusCloud](https://www.centauruscloud.io/) and 
[Elastic Horovod](https://horovod.readthedocs.io/en/stable/elastic_include.html). 
It is able to run on any Kubernetes compatible platforms. 

The framework exposes K8s CRDs (HorovodJobs, ElasticHorovodJobs) as API extensions to end users. It provides the ability to run
deep learning training jobs elastically within the cluster. 

It consists of two components: an ElasticHorovodJob controller and a GPU allocator. The ElasticHorovodJob controller
watches the creation of CRD objects and launches elastic horovod jobs accordingly. The GPU allocator dynamically allocates
the pool of GPUs available in the cluster to the training jobs. 

Key benefits of this framework include fault tolerance, higher GPU utilization and lower job completion time.

## Quick start 

### Prerequisites
* Provision a Centaurus or Kubernetes cluster with GPU nodes.
* Install official [Nvidia device plugin](https://github.com/NVIDIA/k8s-device-plugin).
* A kubeconfig with enough permissions. (We are working on deploying it to run within a pod) 
* Generate an ssh key pair, for exapmle, 
```bash
ssh-keygen -t rsa -b 2048
``` 

### Steps

1. Clone repo and create CRDs
```bash
git clone git@github.com:CentaurusInfra/AI-SIG.git
cd alnair/elastic-training
kubectl apply -f config/crd/bases/ai.centauruscloud.io_elastichorovodjobs.yaml
kubectl create configmap ai-horovod-discover-hosts --from-file=scripts/discover_hosts.sh
kubectl create secret generic horovod-sshkeys --from-file=id_rsa=/path/to/id_rsa --from-file=authorized_keys=/path/to/id_rsa.pub
```
2. Build and run the operators
```bash
make manager
bin/manager
```
3. Open another terminal and create a sample CRD
```bash
kubectl apply -f config/samples/ai_v1alpha1_elastichorovodjob.yaml
```
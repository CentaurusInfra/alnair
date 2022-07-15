# K-V-Store

This project aims to develop a K8s-based storage caching system for speeding up big data-driven Deep Leaning training and inference jobs.

## System Architecture and Main Components
- Global Manager (GM): acts as the core of the cache cluster. Specifically, it
    - handles Connection, Job Registration, Heartbeat and Cache Missing requests from Client
    - interacts with MongoDB servers to manipulate client and job meta information
    - pulls data from S3 to Redis Cluster
    - performs data persistence by periodically flushing data into NFS storage
- AlnairPod: is a [CRD](./alnairpod-operator) resource associated with a ConfigMap and a Pod, in which the latter hosts a Client container and a DLJob container.
    - Client Container: communicates with GM and share data location in Redis with DLJob. This container is automatically created when deploying an AlnairPod, and act as a daemon process of bridging DLJobs and Cache Cluster.
    - DLJob Container: is the container where DL training or inference jobs are running. Users are required to subclass our [AlnairJobDataset](./examples/lib/AlnairJobDataset.py) class and initialize the [AlnairJobDataLoader](./examples/lib/AlnairJobDataLoader.py) class, as what they do when defining Pytorch custom Dataset.
- Redis Cluster: is a key-value caching system deployed as a K8s StatefulSet.
    - Content hashing is executed when generating keys. 
    - When data eviction is enabled, GM automatically dumps data that are likely to be evicted into NFS. 
    - Redis Cluster supports two working modes: direct mode and proxy mode, handling cases in that clients are from the same or different K8s cluster than the Cache Cluster, respectively.
- HA MongoDB: is a MongoDB server cluster deployed as a K8s StatefulSet. The database contains a [Client](./src/manager/mongo-schemas/client.json) and a [Job](./src/manager/mongo-schemas/job.json) collections.

![arch](./docs/images/arch.png)

## Cache Cluster Setup
### Prerequisites:
- 2+ Ubuntu machines
- A Kubernetes cluster with kubeadm,kubectl,kubelet (v1.18.0-00) installed
- AWS S3 Credential

In the below example, we'll run the Cache Cluster on Worker nodes and AlnairPods on the Master node.

### Step 1. Redis Statefulset (3 replicas)
Redis configurations are saved in [config.yaml](./cache-cluster/cacher/Redis/cluster/config.yaml). Please change and remember the `masterauth` and `requirepass` fields and uncomment the `bind=0.0.0.0`. For data eviction, consider to configure `maxmemory-policy`, `maxmemory` and `maxmemory-samples`. To run more instances, change the `replicas` field in the [cluster.yaml]((./cache-cluster/cacher/Redis/cluster/cluster.yaml)) file.

```bash
cd cache-cluster/cacher/Redis/cluster
sh setup.sh init direct # enter `yes` when you see the prompt
```

If clients are from a different K8s cluster than the Cache Cluster, please set the `enable_proxy` knob to true in the [config.yaml](./cache-cluster/manager/configmap.yaml). This will enable the envoy-proxy mode. Then, execute:
```bash
kubectl apply -f envoy-proxy-config.yaml
kubectl apply -f envoy-proxy-deploy.yaml
```
Envoy-proxy service is exposed through NodePort with targetPort 30001. To get the node IP envoy-proxy is using:
```bash
kubectl get pods -o wide | grep envoy-proxy | awk '{ print $7 }'
```

### Step 2. HA MongoDB Statefulset
```bash
# please ensure you create diectory /mnt/data on all nodes
cd cache-cluster/mongodb
sh setup.sh init
```

### Step 3. Global Manager (GM) Deployment
Edit the [configmap.yaml](cache-cluster/manager/configmap.yaml) to define configurations of the manager, mongodb, and redis_proxy. To enable data persistence, execute [nfs.sh](cache-cluster/manager/nfs.sh) to set up NFS server on all worker nodes. Then execute:
```bash
cd cache-cluster/manager/
kubectl apply -f .
```
### Step 4. Deploy AlnairPod CRD
Follow the [README](./alnairpod-operator/README.md) file to deploy AlnairPod operator and run its controller in your cluster.

## AlnairPod Development and Deployment
### Step 1. Write Deep Learning Job
All datasets that represent a map from keys to data samples should subclass
the [AlnairJobDataset](./examples/lib/AlnairJobDataset.py) class. All subclasses should overwrite:
- meth:`__convert__`: supporting pre-processing loaded data. Data are saved as key-value map before calling this method. You are responsible for reshaping the dict to desired array.
- meth:`__getitem__`: supporting fetching a data sample for a given index. Subclasses could also optionally overwrite
- meth:`__len__`: returning the size of the dataset

To traverse the created dataset, initialize an object of [AlnairJobDataLoader](./examples/lib/AlnairJobDataLoader.py) as the normal way of creating a PyTorch Dataloader. 

More examples can be found [here](./examples/). Please create a Docker image for the application before moving to the next step.
### Step 2. Example: Create an Imagenet AlnirPod
The below shows an example alnairpod secret and ImageNet AlnairPod.
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: alnairpod-client-secret
  namespace: default
type: Opaque
stringData:
  client.conf: |
    [aws_s3]
    aws_access_key_id=<your key id>
    aws_secret_access_key=<your key>
    region_name=<your region name>
---
apiVersion: alnair.com/v1alpha1
kind: AlnairPod
metadata:
  name: imagenet
  namespace: default
spec:
  secret:  # ensure you have created the Secret that contains 
    name: alnairpod-client-secret
  jobs: # all fields supported by a regular Pod container are supported here.
  - name: job
    image: <your repository>/imagenet:latest
    command: ["python3", "main.py"]
    datasource: # required field
      name: ImageNet-Mini
      bucket: <your s3 bucket name>
      keys: # prefix is supported
      - imagenet-mini/train
      - imagenet-mini/val
    configurations:
      usecache: true # use Cache Cluster(true) or S3(false)
      maxmemory: 0 # unlimited
      durabilityindisk: 1440
      lazyloading: true # lazy loading saves memory
    tty: true
    stdin: true
```
DL Job will automatically start once data are loaded from S3 to Redis. The framework checks and copies data only if it's unavailable in the Cache Cluster or modified since last use.
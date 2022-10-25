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
    - DLJob Container: is the container where DL training or inference jobs are running. Users are required to subclass our AlnairJobDataset class and initialize the AlnairJobDataLoader, as what they do when defining Pytorch custom Dataset.
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
```bash
kubectl taint nodes $(hostname | awk '{print tolower($0)}') node-role.kubernetes.io/master-

kubectl label node --overwrite $(hostname) alnair=Client
workers=$(kubectl get nodes --no-headers=true --selector=kubernetes.io/hostname!=$(hostname) | awk '{print $1}')
kubectl label node --overwrite $workers alnair=CacheCluster
```
Clone the repository
```bash
mkdir alnair-k-v-store
cd alnair-k-v-store
git init
git remote add -f origin https://github.com/CentaurusInfra/alnair.git
git config core.sparsecheckout true
echo "storage-caching/k-v-store" >> .git/info/sparse-checkout
git pull origin main
cd alnair-k-v-store/storage-caching/k-v-store
```

### Step 1. Redis Statefulset
Redis configurations are saved in [config.yaml](./cache-cluster/cacher/Redis/cluster/config.yaml). To run more servers (default to 3), change the `replicas` field in the [cluster.yaml]((./cache-cluster/cacher/Redis/cluster/cluster.yaml)) file.

```bash
cd cache-cluster/cacher/Redis/cluster
sh setup.sh init # enter `yes` when you see the prompt
```

(Optional) If clients are from a different K8s cluster than the Cache Cluster, please set the `enable_proxy` knob to true in the [config.yaml](./cache-cluster/manager/configmap.yaml). This will enable the envoy-proxy mode. Then, execute:
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
Edit the [configmap.yaml](cache-cluster/manager/configmap.yaml) to define configurations of the manager and mongodb. To enable data persistence, execute [nfs.sh](cache-cluster/manager/nfs.sh) to set up NFS server on all worker nodes.
```bash
cd cache-cluster/manager/
sh nfs.sh <CIDR> # example: sh nfs.sh 192.168.41.0/24

# deploy GM
kubectl apply -f .
```

## AlnairPod Development and Deployment
### Step 1. Create CRD 
```bash
cd alnairpod-operator
make install run
```

### Step 2. Write Deep Learning Job
Install the [alnairjob](https://pypi.org/project/alnairjob/) library using pip.
```bash
pip3 install alnairjob
```
All datasets that represent a map from keys to data samples should subclass
the AlnairJobDataset class. All subclasses should overwrite:
- meth:`__convert__`: supporting pre-processing loaded data. Data are saved as key-value map before calling this method. You are responsible for reshaping the dict to desired array.
- meth:`__getitem__`: supporting fetching a data sample for a given index.
- meth:`__len__`: returning the size of the dataset

To traverse the created dataset, initialize an object of AlnairJobDataLoader as the normal way of creating a PyTorch Dataloader. 

### Step 3. Example: Create an Imagenet AlnirPod
The Imagenet-Mini dataset is availabel on [Kaggle](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000).

Please refer [ImageNetMiniDataset](./examples/imagenet/src/ImageNetMiniDataset.py) for Dataset implementation. Then, in your main program:
```python
from AlnairJob import AlnairJobDataset, AlnairJobDataLoader

val_dataset = ImageNetDataset(keys=['imagenet-mini/val'], transform=transform)
...
val_loader = AlnairJobDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False num_workers=args.workers, pin_memory=True, sampler=val_sampler)
...
```

The below shows an example alnairpod secret and ImageNet AlnairPod. Make sure you have wrapped your DL Job in a Docker image.

```yaml
---
apiVersion: v1
kind: Secret
metadata:
  name: client-secret
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
    name: client-secret
  jobs: # all fields supported by a regular Pod container are supported here.
  - name: job
    image: centaurusinfra/imagenet:latest
    command: ["python3", "main.py"]
    datasource: # required field
      name: ImageNet-Mini
      bucket: <your s3 bucket name>
      keys: # prefix is supported, must be valid object keys/prefix in S3
      - imagenet-mini/train
      - imagenet-mini/val
    configurations:
      usecache: true # use Cache Cluster(true) or S3(false)
      maxmemory: 0 # unlimited
      durabilityindisk: 1440
      lazyloading: true # lazy loading mode saves memory
    tty: true
    stdin: true
```
You should see the following resources after execution:
```text
NAME                                     READY   STATUS    RESTARTS   AGE
pod/alnairpod-manager-75b69c6d44-dxbcq   1/1     Running   0          4m28s
pod/alnairpod-manager-75b69c6d44-m9jhv   1/1     Running   0          4m28s
pod/alnairpod-manager-75b69c6d44-w76d6   1/1     Running   0          4m28s
pod/imagenet-mini0                       2/2     Running   0          58s
pod/mongo-0                              1/1     Running   0          7m42s
pod/mongo-1                              1/1     Running   0          7m40s
pod/redis-cluster-0                      1/1     Running   0          20m
pod/redis-cluster-1                      1/1     Running   0          20m
pod/redis-cluster-2                      1/1     Running   0          20m

NAME                        TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)           AGE
service/alnairpod-manager   NodePort    10.97.125.143    <none>        50051:32200/TCP   4m28s
service/kubernetes          ClusterIP   10.96.0.1        <none>        443/TCP           33m
service/mongo               NodePort    10.96.65.198     <none>        27017:30017/TCP   7m42s
service/redis-cluster       ClusterIP   10.111.134.223   <none>        6379/TCP          20m

NAME                                READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/alnairpod-manager   3/3     3            3           4m28s

NAME                                           DESIRED   CURRENT   READY   AGE
replicaset.apps/alnairpod-manager-75b69c6d44   3         3         3       4m28s

NAME                             READY   AGE
statefulset.apps/mongo           2/2     7m42s
statefulset.apps/redis-cluster   3/3     20m
```
The Imagenet job is ready to execute once data are loaded from S3 to Redis. You should see Client starts sending Heartbeat message to GM:
```bash
$ kubectl logs -f imagenet-mini0 --container client
2022-07-15 23:53:44,578 - __main__ - INFO - connect to server
2022-07-15 23:53:44,590 - __main__ - INFO - waiting for data preparation
2022-07-15 23:57:08,081 - __main__ - INFO - receiving registration response stream
2022-07-15 23:57:08,081 - __main__ - INFO - registered job job, assigned jobId is alnair-client/ImageNet-Mini
2022-07-15 23:57:08,082 - __main__ - INFO - send heartbeat
```
Note that the GM checks and copies data only if they are unavailable in the Cache Cluster or modified since last use. Therefore, the time in the first execution includes the time of downloading data.

Experiment results:
![](./docs/images/exp1.png)

Experiment results indicate Alnair significantly outperforms S3.
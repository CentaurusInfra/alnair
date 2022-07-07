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
### Step 1. Redis Statefulset (3 replicas)
Redis configurations are saved in [config.yaml](./cache-cluster/cacher/Redis/cluster/config.yaml). Please change and remember the `masterauth` and `requirepass` fields and uncomment the `bind=0.0.0.0`. For data eviction, consider to configure `maxmemory-policy`, `maxmemory` and `maxmemory-samples`. To run more instances, change the `replicas` field in the [cluster.yaml]((./cache-cluster/cacher/Redis/cluster/cluster.yaml)) file.

Edit the [secret.yaml](cache-cluster/cacher/Redis/cluster/secret.yaml) to set passwords. Then execute:

```bash
cd cache-cluster/cacher/Redis/cluster
sh setup.sh init # enter `yes` when you see the prompt
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

# execute the follwing commands in the mongodb console:
rs.initiate()
var cfg = rs.conf()
cfg.members[0].host="mongo-0.mongo:27017"
rs.reconfig(cfg)
rs.add("mongo-1.mongo:27017")
rs.status()
db.createUser(
{
    user: "alnair",
    pwd: "alnair",
    roles: [
            { role: "userAdminAnyDatabase", db: "admin" },
            { role: "readWriteAnyDatabase", db: "admin" },
            { role: "dbAdminAnyDatabase", db: "admin" },
            { role: "clusterAdmin", db: "admin" }
        ]
})
exit()

# get the Cluser-IP and targetPort of mongodb service
kubectl get svc mongo
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
- meth:`__preprocess__`: supporting pre-processing loaded data. Data are saved as key-value map before calling this method. You are responsible for reshaping the dict to desired format.
- meth:`__getitem__`: supporting fetching a data sample for a given index. Subclasses could also optionally overwrite
- meth:`__len__`: returning the size of the dataset

To traverse the created dataset, initialize an object of [AlnairJobDataLoader](./examples/lib/AlnairJobDataLoader.py) as the normal way of creating a PyTorch Dataloader. 

More examples can be found [here](./examples/). Please create a Docker image for the application before moving to the next step.
### Step 2. Create the AlnirPod
The below shows an example of the definition of an ImageNet AlnairPod.
```yaml
apiVersion: alnair.com/v1alpha1
kind: AlnairPod
metadata:
  name: imagenet
  namespace: default
spec:
  secret:  # ensure you have created the Secret
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
    configurations: # optional
      usecache: true
      maxmemory: 0  # unlimited memory
      durabilityindisk: 1440
    tty: true
    stdin: true
```
Once the AlnairPod is deployed, you will see resources similar to:
```shell
zhuangwei@fw0015254:~/alnair/storage-caching$ kubectl get all
NAME                                     READY   STATUS    RESTARTS     AGE
pod/alnairpod-manager-6fc86ccfdd-2j8km   1/1     Running   0            25h
pod/alnairpod-manager-6fc86ccfdd-bt8xt   1/1     Running   0            25h
pod/alnairpod-manager-6fc86ccfdd-dvksg   1/1     Running   0            25h
pod/envoy-redis-proxy-7ff5cb5996-5ljf9   1/1     Running   0            8d
pod/envoy-redis-proxy-7ff5cb5996-xzxms   1/1     Running   0            8d
pod/imagenet                             2/2     Running   0            16h
pod/mongo-0                              1/1     Running   0            14d
pod/mongo-1                              1/1     Running   1 (8d ago)   14d
pod/redis-cluster-0                      1/1     Running   0            8d
pod/redis-cluster-1                      1/1     Running   0            8d
pod/redis-cluster-2                      1/1     Running   0            8d

NAME                        TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)           AGE
service/alnairpod-manager   NodePort    10.104.1.114     <none>        50051:32200/TCP   25h
service/envoy-redis-proxy   NodePort    10.97.232.136    <none>        6379:30079/TCP    8d
service/kubernetes          ClusterIP   10.96.0.1        <none>        443/TCP           24d
service/mongo               NodePort    10.110.131.82    <none>        27017:30017/TCP   14d
service/redis-cluster       ClusterIP   10.108.234.231   <none>        6379/TCP          8d

NAME                                READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/alnairpod-manager   3/3     3            3           25h
deployment.apps/envoy-redis-proxy   2/2     2            2           8d

NAME                                           DESIRED   CURRENT   READY   AGE
replicaset.apps/alnairpod-manager-6fc86ccfdd   3         3         3       25h
replicaset.apps/envoy-redis-proxy-7ff5cb5996   2         2         2       8d

NAME                             READY   AGE
statefulset.apps/mongo           2/2     14d
statefulset.apps/redis-cluster   3/3     8d
```
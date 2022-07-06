# Alluxio based file system caching for K8s workloads

## Alluxio Cluster Setup

### I. Master and Worker node preparation

Just follow below commands to setup the Alluxio on master first. The workers will be setup as part of steps during preparation of the master.
This guide walks thruogh the option to setup Alluxio on Kubernetes using the Helm Charts. If you want to setup Alluxio on Bare-Metal cluster, that guide will be linked here.

#### 1) Install Helm on Ubuntu if you don't have it:
On Ubuntu / Debian:
Option 1: Using `apt`
```
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
sudo apt-get install apt-transport-https --yes
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

Option 2: Using `snap`
```
sudo snap install helm --classic
```

#### 2) Execute below commands to create the "disk volume" first, that Alluxio can use for any persisted data:
```
cd $HOME/alluxio-2.8.0/integration/kubernetes/singleMaster-localJournal
helm uninstall alluxio
kubectl create -f alluxio-master-journal-pv.yaml
```

#### 3) Install Alluxio using Helm now:
```
helm upgrade --install alluxio --debug --values my-alluxio-values.yaml -f config.yaml -f alluxio-configmap.yaml --set journal.format.runFormat=true --set alluxio.master.hostname=_Hostname of your master_ --set alluxio.zookeeper.enabled=true --set alluxio.zookeeper.address=[ _IP of your master_ ]  alluxio-charts/alluxio  | tee  helm.out   
```

Now Alluxio should have been deployed. One way is to wait for a minute and then check for the pods to be running in `alluxio` namespace, like this:
```
kubectl get pods -n alluxio

NAMESPACE      NAME                                   READY   STATUS    RESTARTS   AGE     IP              NODE         NOMINATED NODE   READINESS GATES
alluxio        alluxio-master-0                       2/2     Running   0          5d18h   10.244.12.253   edgeml2gpu   <none>           <none>
alluxio        alluxio-worker-7p42v                   2/2     Running   0          5d18h   10.244.16.103   titan34      <none>           <none>
alluxio        alluxio-worker-g9bc4                   2/2     Running   0          5d18h   10.244.12.254   edgeml2gpu   <none>           <none>
```

#### 4) Scaling of the Alluxio Cluster:
##### 4a) Details of Scaling the Master:

The Master is a StatefulSet - provides STATEFUL ordering, uniqueness and unique Volume per pod, because each replica will have own "state".
You CAN scale master for HA configuration, in one of the few ways mentioned below. Just like below (this is an optional step):

Option 1:
Just increment the `count` in config.yaml
OR
```
kubectl scale --replicas=2 statefulset/alluxio-master
```

AND
```
helm upgrade alluxio -f config.yaml alluxio-charts/alluxio
```

##### 4b) Details of Scaling the Workers:

The "alluxio-worker" is a DaemonSet. As such, you can NOT directly scale workers. It runs one pod per node. Just add nodes to scale Alluxio cluster.

#### 5) Health & Monitoring of the Alluxio Cluster:

In order to monitor the cluster, we can utilize Alluxio Dashboard. To do that we'd need to forward the Dashboard port (default port 19999, can be changed in config.yaml), like this, on any Ubuntu node in the network:
```
kubectl port-forward --address 0.0.0.0 pods/alluxio-master-0 8080:19999 &
```

To test this and view healt metrics, browse to `http://10.145.41.31:8080/metrics`.
To view healt metrics, browse to `http://10.145.41.31:8080/overview`.

#### 6) How To "get into" master:

Just execute this command:
```
kubectl exec -ti alluxio-master-0 -c alluxio-master -- /bin/bash
```
#### 7) How To Alluxio Cluster's Functionalities?:

One option is to persist the data for yourself, mount that as volume in your pod, and deploy your training pod. We'd get to describe that later, but there's an easier way.

Get into the master by following above step, and then just run below command. This will execute built in self tests that came with Alluxio. All tests need to pass, or you can show errors using the --debug flag:
```
		alluxio runTests
```

#### 8) How To Verify and Repair Persistence of Data In the Cache:

The checkConsistency command compares Alluxio and under storage metadata for a given path. If the path is a directory, the entire subtree will be compared. The command returns a message listing each inconsistent file or directory. The system administrator should reconcile the differences of these files at their discretion. To avoid metadata inconsistencies between Alluxio and under storages, design your systems to modify files and directories through Alluxio and avoid directly modifying the under storage.

If the -r option is used, the checkConsistency command will repair all inconsistent files and directories under the given path. If an inconsistent file or directory exists only in under storage, its metadata will be added to Alluxio. If an inconsistent file exists in Alluxio and its data is fully present in Alluxio, its metadata will be loaded to Alluxio again.

If the -t <thread count> option is specified, the provided number of threads will be used when repairing consistency. Defaults to the number of CPU cores available,

This option has no effect if -r is not specified

_List each inconsistent file or directory:_

```
alluxio fs checkConsistency /
```

_To repair the inconsistent files or derectories:_

```
alluxio fs -r checkConsistency <path of file or directory>
```

ALso note that the checksum command outputs the md5 value of a file in Alluxio.

For example, checksum can be used to verify the contents of a file stored in Alluxio.

```
alluxio fs checksum /futurewei-data/datasets/coco_dataset

md5sum: bf0513403ff54711966f39b058e059a3
md5 data
MD5 (data) = bf0513403ff54711966f39b058e059a3
```

## II. PV/PVC Configurations

As discussed in step 2 above, Alluxio first needs a Persistent Volume to load data from and persist data into. There are actually two volumes to describe.
Volume "alluxio-journal-alluxio-master" is used for journaling, mounting and persisting your training big-data.
Volume "alluxio-worker-domain-socket" is used as a "short circuit" to utilize local filesystem APIs to access files that are persisted on the same node, thereby circumventing or "short circuiting" the network protocols to access the same files. Networked access is required for files that are on other alluxio workers. This is where data locality is important. _(I am adding more details on short circuit, concept of domain sockets in Unix and the journal soon.)_

```
# Name the file alluxio-master-journal-pv.yaml
kind: PersistentVolume
apiVersion: v1
metadata:
  name: alluxio-journal-0
  labels:
    type: local
spec:
  storageClassName: standard
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: /mnt/fuse3
  claimRef:
    name: alluxio-journal-alluxio-master-0
    namespace: alluxio

---
# Name the file alluxio-master-journal-pv.yaml
kind: PersistentVolume
apiVersion: v1
metadata:
  name: alluxio-worker-0
  labels:
    type: local
spec:
  storageClassName: standard
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: /mnt/fuse3
  claimRef:
    name: alluxio-worker-domain-socket
    namespace: alluxio
```

## Pod yaml exmaple with Alluxio


## Sample Results on data loading speed with and without Alluxio
1. within CPU cluster (cpu workloads)
2. cross CPU and GPU network (gpu workloads)

## Notes
1. user data/path isolation
2. network bottleneck


# Steps to Start Alluxio Cluster

To get Alluxio Data Orchestration Cluster started from scratch, just follow below commands / steps:

1) [One-time step] Create alluxio-user on all workers for ssh commands.
   Since k8s CRD API doesn't fully work, this is the custom user to execute certain commands

	`sudo useradd -s /bin/bash -U -m -b /home alluxio-user # On all alluxio workers`

*All below steps are automated*

2) Enable password-less ssh for alluxio-user on all k8s workers

	`sshpass -p ${ALLUXIO_PASS} ssh-copy-id -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -f alluxio-user@${WORKER}`

${ALLUXIO_PASS} is the environment variable that stores clear-text password of the `alluxio-user`. In real deployment the code uses Kubernetes secret `alnair-cache-operator-secret` from file `alnair-cache-crd-operator-secret.yml`. If you want to manually set this password in an experimental deployment, please ping nparekh@futurewei.com (Nikunj Parekh).

3) Create journal dirs on all workers and masters and specify correct ownerships and permissions for them

	```
	mkdir -p /mnt/fuse/{alluxio-journal/datasets,domain-socket}
	chmod -R 0777 /mnt/fuse
	```

4) Clone file-system caching code and Create Persistent Volume that'd work with Alluxio
Create or go to the dir whenever you want the code to live, such as ${HONME}/code/. Let's call it <alnair-clone-dir>.
Clone code and checkout my branch (or keep in the main branch once code is released).
	```
	mkdir -p <alnair-clone-dir>
	cd <alnair-clone-dir>
	git clone https://github.com/CentaurusInfra/alnair/
	git checkout alluxio-data-orchestration
	```
	
Delete any existing Alluxio deployment and volume

	```
	helm uninstall alluxio
	cd <alnair-clone-dir>/alnair/storage-caching/file-system/alluxio-integration/alluxio-2.8.1/singleMaster-localJournal
	kubectl delete alluxio-master-journal-pv.yaml
	```

6) Add new Alluxio Helm Chart

	```
	helm repo remove alluxio-charts
	helm repo add alluxio-charts https://alluxio-charts.storage.googleapis.com/openSource/2.8.1
	```

7) Create ClusterRole, ClusterRoleBinding to let Operator, Master etc all use operations to query, delete, create, list pods, deployments, jobs, nodes

	```
	kubectl create -f alnair-cache-crd-operator-rbac.yml
	kubectl create -f alluxio-master-journal-pv.yaml
	```

8) Deploy Alluxio

	```
	helm upgrade --install alluxio --debug --values my-alluxio-values.yaml -f config.yaml -f alluxio-configmap.yaml  --set fuse.enabled=true --set fuse.clientEnabled=true --set alluxio.master.hostname=\`hostname\` --set alluxio.worker.ramdisk.size=100G --set alluxio.worker.ramdisk.size=50Gi --set alluxio.worker.tieredstore.level0.dirs.quota=50Gi   alluxio-charts/alluxio  2>&1>helm.out
	```

Note that we can finetune this later such that the resource deployments are only observed within certain namespace instead of at cluster level, by creating Role and RoleBinding instead. However, querying workers / k8s nodes will continue to require Cluster level RBAC.


# (Older instructions)
## New helpful Utilities to work with your datasets

### The `dataorch-host-data` program will allow you to host the data. Below is how that program works:
```
cd ~/data-orchestration/storage-caching/file-system/futurewei-tools
./dataorch-host-data -h
         Futurewei Data Orchestrator v0.1

         Usage: ./dataorch-host-data <node> <path> [data_type: datasets | deployment] [namespace] [debug: 0 | 1]

                Node is one of the Data Orcheatration worker nodes and should allow ssh without password
                Path is the directory or file to be copied into Data Orchestration
                data_type is the type of data that helps decide storage location in the Data Orchestration system
                  Two data_types are currently supported: datasets or deployment
```
### Use it like this:

Suppose you want to host into Alluxio in-memory cache, the dataset from some folder "~/my-awesome-datasets/some_smaller_coco_dataset_dir/data", which happens to be not on the master node where you are loggedin, but some worker node, say "fw0013512", then here's how you will host that data:

```
./dataorch-host-data fw0013512 ~/my-awesome-datasets/some_smaller_coco_dataset_dir datasets default 1
Argument 0 is the program name
Argument 1, fw0013512 is the node / machine name where data is currently available, needs to be an Alluxio worker
Argument 2, ~/alluxio-2.7.4/webui/master/build/, is the origin / source path to the file or directory of your data
Argument 3, "datasets", is the type of data. It can be either "datasets" or "deployment". This type is used to organize data correctly in the orchestration.
   The datasets are cached under /futurewei-data/datasets/ and the experiments / programs under /futurewei-data/experiments.
Argument 4, default in our example is an OPTIONAL namespace name. The default value is namespace=default. Please specify correct namespace where Alluxio was deployed.
Argument 5, the "1", enables debug logs on screen, any other value will skip on screen logging.
```

If necessary, we can enhance the program argument handling etc later. The prioriy rightnow is to experiment with the orchestration v1.0.

The program will MOVE your data from ~/my-awesome-datasets/some_smaller_coco_dataset_dir to /mnt/fuse3/datasets/some_smaller_coco_dataset_dir and then copy it into Alluxio's in-memory cache, at /futurewei-data/datasets/some_smaller_coco_dataset_dir. _(Isn't that cool ?!)_

Your cached data is hosted inside /opt/domain on Alluxio worker pods [only], and it's a SHAME that Alluxio documentation keeps this key piece of info as poorly documented _secret_!

Alluxio will distribute / balance your data across all workers even though commands are sent to master or any one worker.

The above program will produce below screen output:
```
         Futurewei Data Orchestrator v0.1
         Data type = datasets
         Namespace = default
         Debug = 1

         Directory / file to be copied  : drwxr-xr-x 3 nikunj nikunj 4096 May  2 13:35 /home/nikunj/my-awesome-datasets/some_smaller_coco_dataset_dir
         Node this data is hosted on    : fw0013512
         Kubernetes cluster nodes       : fw0013511 fw0013512 fw0013513
         Data Orchestration Master Pods : alluxio-master-0
         Data Orchestration Worker Pods : alluxio-worker-57jvr alluxio-worker-zm2hd

         STEP 1 of 2: For speed, *MOVING*, not copying the data into filesystem volume mount /mnt/fuse3 on node fw0013512:

         New data is located at /mnt/fuse3/datasets/some_smaller_coco_dataset_dir on fw0013512:
total 64
-rw-r--r-- 1 nikunj nikunj  1323 May  2 13:35 asset-manifest.json
-rw-r--r-- 1 nikunj nikunj 24838 May  2 13:35 image1.jpg
-rw-r--r-- 1 nikunj nikunj  2221 May  2 13:35 index.html
-rw-r--r-- 1 nikunj nikunj   316 May  2 13:35 image2.jpg
-rw-r--r-- 1 nikunj nikunj 13966 May  2 13:35 image3.jpg
-rw-r--r-- 1 nikunj nikunj  1041 May  2 13:35 image4.jpg
drwxr-xr-x 5 nikunj nikunj  4096 May  2 13:35 image5.jpg
....

         STEP 2 of 2: Now hosting data at /mnt/fuse3/datasets/some_smaller_coco_dataset_dir on node fw0013512 into Futurewei Data Orchestrator v0.1 if it wasn't already hosted...

Copied file:///opt/domain/datasets/some_smaller_coco_dataset_dir/asset-manifest.json to /futurewei-data/datasets/some_smaller_coco_dataset_dir/asset-manifest.json
Copied file:///opt/domain/datasets/some_smaller_coco_dataset_dir/image1.jpg to /futurewei-data/datasets/some_smaller_coco_dataset_dir/image1.jpg
Copied file:///opt/domain/datasets/some_smaller_coco_dataset_dir/index.html to /futurewei-data/datasets/some_smaller_coco_dataset_dir/index.html
Copied file:///opt/domain/datasets/some_smaller_coco_dataset_dir/image2.jpg to /futurewei-data/datasets/some_smaller_coco_dataset_dir/image2.jpg
Copied file:///opt/domain/datasets/some_smaller_coco_dataset_dir/image3.jpg to /futurewei-data/datasets/some_smaller_coco_dataset_dir/image3.jpg
...
and so on
...
```

### There it is! You can now reference the data by adding a PersistentVolume and PersistentVolumeClaim like below in your existing Pod / Job description yaml:
```
apiVersion: v1
kind: Pod
metadata:
  name: my-awesome-training-pod
spec:
  containers:
  - name: my-awesome-training-service

    image: centaurusinfra/my-awesome-docker-image
    command: ["python3", "my-awesome-training-program.py"]
...
...
    volumeMounts:
      - mountPath: /the-path-to-dataset-that-your-python-training-wants # <------------ Line 1 of 2 changed
         #- mountPath: /opt/domain/cifar10-data/, or /opt/domain/datasets etc
        name: my-awesome-alluxio-data

  volumes:
  - name: my-awesome-alluxio-data
    hostPath:
    # For Alluxio
      path: /mnt/fuse3/datasets/some_smaller_coco_dataset_dir  # <---------- Line 2 of 2 changed! That's it! Remember I told you about this path above?
      type: DirectoryOrCreate

```

!! Note: Alluxio copies data from server into the case ONLY ONCE. If you repeatedly run the program, Alluxio will NOT copy same data again, and will generate error mesages. !!

The other scripts are easier and self-explanatory:
```
ls -l /home/nikunj/data-orchestration/alnair/storage-caching/file-system/futurewei-tools
-rwxrwxr-x 1 nikunj nikunj 2.0K Jul  9 19:17 dataorch-check-data
-rwxrwxr-x 1 nikunj nikunj 1.9K Jul  9 19:17 dataorch-delete-data
-rwxrwxr-x 1 nikunj nikunj 2.1K Jul  9 19:17 dataorch-download-data
-rwxrwxr-x 1 nikunj nikunj 4.1K Jul  9 19:32 dataorch-host-data
```



## Alluxio Cluster Setup

### I. Master and Worker node preparation

Just follow below commands to setup the Alluxio on master first. The workers will be setup as part of steps during preparation of the master.
This guide walks thruogh the option to setup Alluxio on Kubernetes using the Helm Charts. If you want to setup Alluxio on Bare-Metal cluster, that guide is here: https://docs.alluxio.io/os/user/stable/en/deploy/Running-Alluxio-On-a-Cluster.html.

Also, to deploy Alluxio on Kubernetes, the provided guide is here: https://docs.alluxio.io/os/user/stable/en/deploy/Running-Alluxio-On-Kubernetes.html. Although, this is being automated by my "auto-depoloy" tool.

#### 1) Install Helm on Ubuntu if you don't have it:
On Ubuntu / Debian:
Option 1: Using `apt`
```
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | \
sudo tee /usr/share/keyrings/helm.gpg > /dev/null
sudo apt-get install apt-transport-https --yes
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] \
https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
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
First, add the Helm repo for Alluxio charts, then upgrade / install / upgrade with install, as below:

```
helm repo add alluxio-charts https://alluxio-charts.storage.googleapis.com/openSource/2.8.0

helm upgrade --install alluxio --debug --values my-alluxio-values.yaml \
-f config.yaml -f alluxio-configmap.yaml --set journal.format.runFormat=true \
--set alluxio.master.hostname=_Hostname of your master_ \
--set alluxio.zookeeper.enabled=true --set alluxio.zookeeper.address=[ _IP of your master_ ] \
 alluxio-charts/alluxio  | tee  helm.out
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

## II. This Is All Great But How Do I Delete All This ...??

Ok, I hear you! Just do this:
```
helm delete alluxio
```
Thats it!

You should also delete your Persistent Volume like this:
```
kubectl delete -f alluxio-master-journal-pv.yaml
```

Volume deletion waits for a long time until all of the pods and the namespaces that use it are gone. You might want to run the below benign command to remove volumes after waiting for a few minutes:

```
kubectl delete persistentvolumeclaim/alluxio-journal-alluxio-master-0 persistentvolume/alluxio-journal-0 persistentvolume/alluxio-fuse3 persistentvolumeclaim/alluxio-worker-domain-socket
```


## III. PV/PVC Configurations

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
1. Within CPU cluster (cpu workloads)
<Later today>

2. Across CPU and GPU network (gpu workloads)
![image](https://user-images.githubusercontent.com/105383186/177635080-237cae64-22c2-4716-8587-1c42e3e910b9.png)


## Other Topics
1. User Data/Path Isolation - How To Isolate And Possibly Also Share Hosted Data Between Users
2. Network Bottlenecks - When Does Network Limit Our Training Performance

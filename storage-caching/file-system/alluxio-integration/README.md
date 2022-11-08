
# Introduction to Alluxio & The Alluxio Data Orchestration Integration with Alnair Platform

Alluxio is an open source Data Orchestration platform. It enables data orchestration in the cloud and fundamentally allows for separation of storage and compute. Alluxio also brings speed and agility to big data and AI workloads and reduces costs by eliminating data duplication and enables users to move to newer storage solutions like object stores.

![image](https://user-images.githubusercontent.com/105383186/198127485-44308fd4-3d58-4fc9-b9ab-73d7993423c5.png)


Alluxio’s data orchestration in the cloud solves for three critical challenges:
• Data locality: Data is local to compute, giving you memory-speed access for your big data and AI/ML workloads
• Data accessibility: Data is accessible through one unified namespace, regardless of where it resides
• Data on-demand: Data is as elastic as compute so you can abstract and independently scale compute and storage
Alluxio enables data orchestration for compute in any cloud. It unifies data silos on-premise and across any cloud, and reduces the complexities associated with orchestrating data for today’s big data and AI/ML workloads

This module introduces steps to setup one of the prime Cloud Data Orchestration and Caching products, called Alluxio. The page also describes method using which we integrated it with Alnair, steps to deploy and un-deploy it and ways to use it with your Machine Learning training datasets.

# Steps to Start Alluxio Cluster

To get Alluxio Data Orchestration Cluster started from scratch, just follow below commands / steps:

1) (One-time step) Create new Kubernetes cluster ....

2) (One-time step) Install Helm on Ubuntu if you don't have it:
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


3) (One-time step) Create alluxio-user on all workers for ssh commands. Since k8s CRD API doesn't fully work, this is the custom user to execute certain commands

	```
	WORKER_NODES=$(kubectl get nodes --selector='!node-role.kubernetes.io/control-plane' -o wide --no-headers | awk '{print $6}')
	for WORKER in ${WORKER_NODES}; do
		sudo useradd -s /bin/bash -U -m -b /home alluxio-user
			# Since `sudo` is used, you'd have to ssh to the node; loop won't work as-is
	done
	```

4) (One-time step) Enable password-less ssh for alluxio-user on all k8s workers

	```
	for WORKER in ${WORKER_NODES}; do
		sshpass -p ${ALLUXIO_PASS} ssh-copy-id -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -f alluxio-user@${WORKER}
	done
	```

	${ALLUXIO_PASS} is the environment variable that stores clear-text password of the `alluxio-user`. In real deployment the code uses Kubernetes secret `alnair-cache-operator-secret` from file `alnair-cache-crd-operator-secret.yml`. If you want to manually set this password in an experimental deployment, please ping nparekh@futurewei.com (Nikunj Parekh).

5) (One-time step) Create journal dirs on all workers and masters and specify correct ownerships and permissions for them

	```
	for WORKER in ${WORKER_NODES}; do
		ssh ${WORKER} sudo mkdir -p /mnt/fuse/{alluxio-journal/datasets,domain-socket}
			# Since `sudo` is used, you'd have to ssh to the node; loop won't work as-is
		ssh ${WORKER} sudo chown -R alluxio-user.alluxio-user /mnt/fuse
		ssh ${WORKER} chmod -R 0777 /mnt/fuse
	done
	```

6) (One-time step) Clone file-system caching code

	Create or go to the dir whenever you want the code to live, such as ${HONME}/code/. Let's call it <alnair-clone-dir>.
	Clone code and checkout my branch (or keep in the main branch once code is released).

	```
	mkdir -p <alnair-clone-dir>
	cd <alnair-clone-dir>
	git clone https://github.com/CentaurusInfra/alnair/
	git checkout alluxio-data-orchestration
	```

7) Delete any existing Alluxio deployment and volume

	```
	helm uninstall alluxio
	cd <alnair-clone-dir>/alnair/storage-caching/file-system/alluxio-integration/alluxio-2.8.1/singleMaster-localJournal
	kubectl delete -f alluxio-master-journal-pv.yaml
	```

	Wait several minutes for PV and PVC to be deleted. A brute-force way, not strongly recommended, is to use below command if this deletion takes an inordinate amount of time --

	```
	# _Weakly_ recommended:
	kubectl delete --grace-period=0 --wait=false  pv/alluxio-journal-0 pv/alluxio-fuse3 pvc/alluxio-journal-alluxio-master-0
	```

8) Add new Alluxio Helm Chart

	```
	helm repo remove alluxio-charts
	helm repo add alluxio-charts https://alluxio-charts.storage.googleapis.com/openSource/2.8.1  # Or a later version you want to try
	```

9) Create ClusterRole, ClusterRoleBinding and the Kubernetes Secret, `alnair-cache-operator-secret` to let Operator, Master etc all use operations to query, delete, create, list pods, deployments, jobs, nodes

	```
	cd <alnair-clone-dir>/alnair/storage-caching/file-system/alluxio-integration
	# Create RBAC Role and Binding
	kubectl create -f alnair-cache-crd-operator-rbac.yml

	# Create Kubernetes secret that provisions the password of alluxio-user
	kubectl create -f alnair-cache-crd-operator-secret.yml

	# Now start operator
	kubectl create -f alnair-cache-crd-operator.yml
	```

	(We can just do `kubectl create -f .` instead of all three commands above.)

10) Deploy Persistent Volume that'd work with Alluxio for data journal and Worker Domain Socket

	```
	cd <alnair-clone-dir>/alnair/storage-caching/file-system/alluxio-integration/alluxio-2.8.1/singleMaster-localJournal
	kubectl create -f alluxio-master-journal-pv.yaml
	```

11) Deploy Alluxio with Cache (ramdisk) = 50G, and quota = 50G, FuSE filesystem enabled, current node as Master

	```
	cd <alnair-clone-dir>/alnair/storage-caching/file-system/alluxio-integration/alluxio-2.8.1/singleMaster-localJournal
	helm upgrade --install alluxio --debug --values my-alluxio-values.yaml -f config.yaml -f alluxio-configmap.yaml  --set fuse.enabled=true --set fuse.clientEnabled=true --set alluxio.master.hostname=`hostname` --set alluxio.worker.ramdisk.size=50Gi --set alluxio.worker.tieredstore.level0.dirs.quota=50Gi   alluxio-charts/alluxio  2>&1>helm.out
	```

Check that Alluxio is deployed:

```
kubectl get pods -n alluxio

NAMESPACE      NAME                                   READY   STATUS    RESTARTS   AGE     IP              NODE         NOMINATED NODE   READINESS GATES
alluxio        alluxio-master-0                       2/2     Running   0          5d18h   10.244.12.253   edgeml2gpu   <none>           <none>
alluxio        alluxio-worker-7p42v                   2/2     Running   0          5d18h   10.244.16.103   titan34      <none>           <none>
alluxio        alluxio-worker-g9bc4                   2/2     Running   0          5d18h   10.244.12.254   edgeml2gpu   <none>           <none>
```

Note that we can finetune this later such that the Operator watches for the resource deployments only within certain namespace, instead of at cluster level, by creating Role and RoleBinding instead of ClusterRole and ClusterRoleBinding. However, querying the Kubernetes worker nodes, to iterate over them will continue to require Cluster level RBAC.

12) (Optional) View Alluxio dashboard (no authentication):
	Step 1) setup port forwarding using below command on your Unix workstation
	```
	kubectl port-forward --address 0.0.0.0 pods/alluxio-master-0 8080:19999 &
	```

	Step 2) browse to
	```
	http://<master node IP address>:8080/
	```

	For example, [My Current Alluxio Cluster on CPU32](http://10.145.41.32:8080/). Additional details are in the sections below.


!! There it is! You can now reference the data by adding a PersistentVolume and PersistentVolumeClaim like below in your existing Pod / Job description yaml:
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

### Scaling of the Alluxio Cluster:
#### Details of Scaling the Master:

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

### Details of Scaling the Workers:

The "alluxio-worker" is a DaemonSet. As such, you can NOT directly scale workers. It runs one pod per node. Just add nodes to scale Alluxio cluster.

### Health & Monitoring of the Alluxio Cluster:

In order to monitor the cluster, we can utilize Alluxio Dashboard. To do that we'd need to forward the Dashboard port (default port 19999, can be changed in config.yaml), like this, on any Ubuntu node in the network:
```
kubectl port-forward --address 0.0.0.0 pods/alluxio-master-0 8080:19999 &
```

To test this and view healt metrics, browse to `http://10.145.41.31:8080/metrics`.
To view healt metrics, browse to `http://10.145.41.31:8080/overview`.

### How To "get into" master:

Just execute this command:

```
kubectl exec -ti alluxio-master-0 -c alluxio-master -- /bin/bash
```

### How To Confirm Alluxio Cluster is Functional?:

One option is to persist the data for yourself, mount that as volume in your pod, and deploy your training pod. We'd get to describe that later, but there's an easier way.

Get into the master by following above step, and then just run below command. This will execute built in self tests that came with Alluxio. All tests need to pass, or you can show errors using the --debug flag:
```
alluxio runTests
```

### How To Verify and Repair Persistence of Data In the Cache:

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

### How Do I Delete All This ...??

Just do this:

```
helm delete alluxio
```

Thats it!

You must also delete your Persistent Volume like this:

```
kubectl delete -f alluxio-master-journal-pv.yaml
```

Volume deletion waits for a long time until all of the pods and the namespaces that use it are gone. You might want to run the below benign command to remove volumes after waiting for a few minutes:

```
kubectl delete persistentvolumeclaim/alluxio-journal-alluxio-master-0 persistentvolume/alluxio-journal-0 persistentvolume/alluxio-fuse3 persistentvolumeclaim/alluxio-worker-domain-socket
```


## PV/PVC Configurations

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
	
```
apiVersion: v1 # apiextensions.k8s.io/v1
kind: Pod
metadata:
  name: alluxio-cifar10

  annotations:
    cacheDataset: "yes"

  #namespace: data-orchestra
spec:
    # nodeName: edgeml2gpu
  nodeName: fw0013603
  containers:
  - name: pytorch

    image: centaurusinfra/alluxio-cifar10
    command: ["python3", "cifar10-demo.py"]

    securityContext:
      privileged: true

    volumeMounts:
    - mountPath: /alluxio-cifar10/data
      name: dataset-mount
  volumes:

  - name: dataset-mount # LOCAL LOCAL
    hostPath:
        path: /mnt/fuse3/alluxio-journal/datasets/cifar10
        type: Directory

  restartPolicy: OnFailure
```

## Project Completion Milestones and Status (August to October, 2022)

![image](https://user-images.githubusercontent.com/105383186/198131069-72702950-7496-4517-93cb-60c8fb8a6dd5.png)

## Sample Results on Data Loading Speed with and without Alluxio (in October, 2022)

![image](https://user-images.githubusercontent.com/105383186/198132738-17e6edfa-ef5e-4a39-acf6-b82b55171f53.png)

### Older Results (in July, 2022)

<img src=https://user-images.githubusercontent.com/105383186/177635080-237cae64-22c2-4716-8587-1c42e3e910b9.png width=60% height=60%>

## Concepts & Implementation Details: Kubernetes Operators & Custom Resource Descriptor (CRD)

<img src=https://user-images.githubusercontent.com/105383186/198134720-2aa02070-65c9-44f6-9294-9f40b424e960.png width=60% height=60%>

<img src=https://user-images.githubusercontent.com/105383186/198134803-5d1c6529-7e01-4665-a068-bf8d3a8ff482.png width=60% height=60%>

## S3 Integreation of Alluxio Data Orchestration

We also have direct S3 integration with CSI S3 plugin.
This uses the custom Filesystem in User Space (FUSE)
1. Mount S3 bucket on all GPU workers using Daemonset
1. FUSE is slightly unstable, but much convenient for Alnair user

<img src=https://user-images.githubusercontent.com/105383186/198135167-b494a955-7c0b-4132-9231-b3ce81a8c85d.png width=60% height=60%>

## Other Topics
1. User Data/Path Isolation - How To Isolate And Possibly Also Share Hosted Data Between Users
2. Network Bottlenecks - When Does Network Limit Our Training Performance

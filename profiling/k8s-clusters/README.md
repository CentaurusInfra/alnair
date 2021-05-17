# Goals
Your cluster will include the following physical resources:

- **One master node**

The master node (a node in Kubernetes refers to a server) is responsible for managing the state of the cluster. It runs Etcd, which stores cluster data among components that schedule workloads to worker nodes.

- **Two worker nodes**

Worker nodes are the servers where your workloads (i.e. containerized applications and services) will run. A worker will continue to run your workload once they’re assigned to it, even if the master goes down once scheduling is complete. A cluster’s capacity can be increased by adding workers.

After completing this guide, you will have a cluster ready to run containerized applications, provided that the servers in the cluster have sufficient CPU and RAM resources for your applications to consume. Almost any traditional Unix application including web applications, databases, daemons, and command-line tools can be containerized and made to run on the cluster. The cluster itself will consume around 300-500MB of memory and 10% of CPU on each node.

Once the cluster is set up, you will deploy the webserver Nginx to it to ensure that it is running workloads correctly.

# Prerequisites
- An SSH key pair on your local Linux/macOS/BSD machine. If you haven’t used SSH keys before, you can learn how to set them up by following this explanation of how to set up SSH keys on your local machine.

- Ansible installed on your local machine. If you’re running Ubuntu 18.04 as your OS, follow the “Step 1 - Installing Ansible” section in [How to Install and Configure Ansible on Ubuntu 18.04 to install Ansible](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-ansible-on-ubuntu-18-04#step-1-%E2%80%94-installing-ansible).

- Familiarity with Ansible playbooks. For review, check out [Configuration Management 101: Writing Ansible Playbooks](https://www.digitalocean.com/community/tutorials/configuration-management-101-writing-ansible-playbooks).

# Step 1 — Setting Up the Workspace Directory and Ansible Inventory File

This section will create a directory on your local machine that will serve as your workspace. You will configure Ansible locally so that it can communicate with and execute commands on your remote servers. Once that’s done, you will create a "hosts" file containing inventory information such as the IP addresses of your servers and the groups that each server belongs to.

Out of your three servers, one will be the master with an IP displayed as master_ip. The other two servers will be workers and will have the IPs worker_1_ip and worker_2_ip.

Create a directory named **~/kube-cluster** in the home directory of your local machine and cd into it:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ mkdir ~/kube-cluster
$ cd kube-cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This directory will be your workspace for the rest of the tutorial and will contain all of your Ansible playbooks. It will also be the directory inside which you will run all local commands.

Create a file named **hosts** using nano or your favorite text editor:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ nano hosts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the following text to the file, which will specify information about the logical structure of your cluster:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[masters]
master ansible_host=<master_ip>

[workers]
worker1 ansible_host=<worker_1_ip>
worker2 ansible_host=<worker_2_ip>

[all:vars]
ansible_connection=ssh
ansible_user=<username>
ansible_ssh_pass=<password>
ansible_python_interpreter=/usr/bin/python3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may recall that inventory files in Ansible are used to specify server information such as IP addresses, remote users, and groupings of servers to target as a single unit for executing commands. **hosts** will be your inventory file and you’ve added two Ansible groups (**masters** and **workers**) to it specifying the logical structure of your cluster.

In the **masters** group, there is a server entry named “master” that lists the master node’s IP (**master_ip**) and specifies that Ansible should run remote commands as the root user.

Similarly, in the **workers** group, there are two entries for the worker servers (**worker_1_ip** and **worker_2_ip**) that also specify the ansible_user as root.

The last line of the file tells Ansible to use the remote servers’ Python 3 interpreters for its management operations.

Save and close the file after you’ve added the text.

Having set up the server inventory with groups, let’s move on to installing operating system level dependencies and creating configuration settings.

# Step 2 — Creating a Non-Root User on All Remote Servers

Create a file named **initial.yml** in the workspace:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ nano initial.yml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Next, add the following play to the file to create a non-root user with sudo privileges on all of the servers. A play in Ansible is a collection of steps to be performed that target specific servers and groups. The following play will create a non-root sudo user:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- hosts: all
  become: yes
  tasks:
    - name: create the '<your username>' user
      user: name=<your username> append=yes state=present createhome=yes shell=/bin/bash

    - name: allow '<your username>' to have passwordless sudo
      lineinfile:
        dest: /etc/sudoers
        line: 'ubuntu ALL=(ALL) NOPASSWD: ALL'
        validate: 'visudo -cf %s'

    - name: set up authorized keys for the <your username> user
      authorized_key: user=ubuntu key="{{item}}"
      with_file:
        - ~/.ssh/id_rsa.pub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Save and close the file after you’ve added the text.

Next, execute the playbook by locally running:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ sudo ansible-playbook -i hosts initial.yml -K
BECOME password: <your sudo password>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Step 3 — Installing Kubernetetes’ Dependencies
In this section, you will install the operating-system-level packages required by Kubernetes with Ubuntu’s package manager. These packages are:

- Docker - a container runtime. It is the component that runs your containers. Support for other runtimes such as rkt is under active development in Kubernetes.

- kubeadm - a CLI tool that will install and configure the various components of a cluster in a standard way.

- kubelet - a system service/program that runs on all nodes and handles node-level operations.

- kubectl - a CLI tool used for issuing commands to the cluster through its API Server.

Create a file named **kube-dependencies.yml** in the workspace:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ nano kube-dependencies.yml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Add the following plays to the file to install these packages to your servers:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- hosts: all
  become: yes
  tasks:
   - name: install Docker
     apt:
       name: docker.io
       state: present
       update_cache: true

   - name: install APT Transport HTTPS
     apt:
       name: apt-transport-https
       state: present

   - name: add Kubernetes apt-key
     apt_key:
       url: https://packages.cloud.google.com/apt/doc/apt-key.gpg
       state: present

   - name: add Kubernetes' APT repository
     apt_repository:
      repo: deb http://apt.kubernetes.io/ kubernetes-xenial main
      state: present
      filename: 'kubernetes'

   - name: install kubelet
     apt:
       name: kubelet=1.18.13-00
       state: present
       update_cache: true

   - name: install kubeadm
     apt:
       name: kubeadm=1.18.13-00
       state: present

- hosts: master
  become: yes
  tasks:
   - name: install kubectl
     apt:
       name: kubectl=1.18.13-00
       state: present
       force: yes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save and close the file when you are finished.

Next, execute the playbook by locally running:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ sudo ansible-playbook -i hosts kube-dependencies.yml -K
BECOME password: <your sudo password>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Step 4 — Setting Up the Master Node

In this section, you will set up the master node. Before creating any playbooks, however, it’s worth covering a few concepts such as Pods and Pod Network Plugins, since your cluster will include both.

A pod is an atomic unit that runs one or more containers. These containers share resources such as file volumes and network interfaces in common. Pods are the basic unit of scheduling in Kubernetes: all containers in a pod are guaranteed to run on the same node that the pod is scheduled on.

Each pod has its own IP address, and a pod on one node should be able to access a pod on another node using the pod’s IP. Containers on a single node can communicate easily through a local interface. Communication between pods is more complicated, however, and requires a separate networking component that can transparently route traffic from a pod on one node to a pod on another.

This functionality is provided by pod network plugins. For this cluster, you will use Flannel, a stable and performant option.

1. SSH into your master node, and initialize the cluster:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ ssh <your username>@<master_ip>
$ sudo kubeadm init --pod-network-cidr=10.244.0.0/16
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The first task initializes the cluster by running **kubeadm init**. Passing the argument **--pod-network-cidr=10.244.0.0/16** specifies the private subnet that the pod IPs will be assigned from. Flannel uses the above subnet by default; we’re telling **kubeadm** to use the same subnet.

To start using your cluster, you need to run the following as a regular user:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ mkdir -p $HOME/.kube
$ sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
$ sudo chown $(id -u):$(id -g) $HOME/.kube/config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2. install Pod network:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ sudo kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output states that the master node has completed all initialization tasks and is in a Ready state from which it can start accepting worker nodes and executing tasks sent to the API Server.:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ sudo kubectl get nodes


NAME      STATUS    ROLES     AGE       VERSION
master    Ready     master    1d        v1.14.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Step 5 — Setting Up the Worker Nodes

## Creating a New Token
1. Using the kubeadm command, list your current tokens on the Master node. If your cluster was initialized over 24 hours ago, the list will likely be empty, since a token’s lifespan is only 24 hours.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ sudo kubeadm tokens list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2. Create a new token using kubeadm. By using the –print-join-command argument kubeadm will output the token and SHA hash required to securely communicate with the master.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ sudo kubeadm token create --print-join-command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Joining the New Worker to the Cluster
1. Using SSH, log onto the new worker node.
2. Use the kubeadm join command with our new token to join the node to our cluster.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ sudo kubeadm join --token <token> <master-ip>:<master-port> --discovery-token-ca-cert-hash sha256:<hash>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
3. List your cluster’s nodes to verify your new worker has successfully joined the cluster.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ sudo kubectl get nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
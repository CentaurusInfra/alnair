# master node
 1. install [docker](https://docs.docker.com/engine/install/ubuntu/)
 2. install [kubeadm, kubelet, kubectl](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/), if want specified version, example is kubeadm=1.21.4-00
 3. swap off memory $swapoff -a
 4. use kubeadm to init cluster ```sudo kubeadm init --pod-network-cidr=10.244.0.0/16```
 5. set kubeconfig (from the printout of kubeadm init), start cluster
 6. install network plugin, [flannel](https://github.com/flannel-io/flannel)

# worker node (with GPU)
 1. install nvidia-driver ```sudo apt-get install nvidia-driver-470```
 2. reboot, verify ```nvidia-smi``` works
 3. install docker like on the master node
 4. install [nvidia container runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
 5. set nvidia container runtime as default in the ```/etc/docker/daemon.json
 {
    "default-runtime":"nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}```
 7. install kubeadm, kubelet
 8. join the cluster (if forgot token, from master node ```kubeadm token create --print-join-command```)


# notes
 1. if master got reboot accidentially, recover with ```strace -eopenat kubectl version```
 2. if kubelet got upgrade accidentially, purge and reinstall

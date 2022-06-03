#!/bin/bash

cni=$1
sudo kubeadm reset -f
sudo rm -r /etc/cni/net.d
sudo rm $HOME/.kube/config

if [ "$cni" == "flannel" ] 
then
    sudo ip link delete flannel.1
    sudo ip link delete cni0
elif [ "$cni" == "calico" ] 
then
    sudo modprobe -r ipip
    sudo ip link delete cni0
fi

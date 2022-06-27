#!/bin/bash

# execute this script on all worker nodes
sudo apt install -y nfs-kernel-server
sudo mkdir /nfs_storage
sudo chmod -R 777 /nfs_storage/
sudo echo "/nfs_storage 192.168.41.0/24(rw,sync,no_all_squash,no_subtree_check)" >> /etc/exports
sudo exportfs -rv
sudo systemctl start nfs-kernel-server
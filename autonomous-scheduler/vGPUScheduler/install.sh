#!/bin/bash
backup_file_name="kube-scheduler-backup.yaml".$(date "+%Y-%m-%d.%H-%M-%S")
mv /etc/kubernetes/manifests/kube-scheduler.yaml /etc/kubernetes/manifests/$backup_file_name

kubectl apply -f alnair/autonomous-scheduler/vGPUScheduler/manifests/extra-rbac-kube-scheduler.yaml

cp alnair/autonomous-scheduler/vGPUScheduler/manifests/vGPUScheduler-config.yaml /etc/kubernetes/manifests/

cp alnair/autonomous-scheduler/vGPUScheduler/manifests/kube-scheduler.yaml /etc/kubernetes/manifests/

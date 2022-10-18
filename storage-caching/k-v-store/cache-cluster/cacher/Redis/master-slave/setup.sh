#!/bin/bash

if [ $1 == "init" ]
then
    kubectl apply -f ../sc.yaml
    kubectl apply -f pv.yaml
    kubectl apply -f redis-config.yaml
    kubectl apply -f redis-master-slave.yaml
elif [ $1 == "del" ]
then
    kubectl delete svc redis
    kubectl delete statefulset redis
    # kubectl delete pvc --all
    kubectl delete -f pv.yaml
    kubectl delete -f ../sc.yaml
    kubectl delete -f redis-config.yaml
    sudo rm -r /storage/data*
fi
kubectl config set-context --current --namespace=default

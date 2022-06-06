#!/bin/bash

checkns=$(kubectl get ns | grep redis)
if [ ${#checkns} == 0 ]
then
    kubectl create ns redis
fi
kubectl config set-context --current --namespace=redis

if [ $1 == "init" ] 
then
    kubectl apply -f ../sc.yaml
    kubectl apply -f pv.yaml
    kubectl apply -n redis -f redis-config.yaml
    kubectl apply -n redis -f redis-master-slave.yaml  
elif [ $1 == "del" ]  
then
    kubectl -n redis delete svc redis
    kubectl -n redis delete statefulset redis
    kubectl delete -n redis pvc --all
    kubectl delete -f pv.yaml
    kubectl delete -f sc.yaml
    kubectl delete -n redis -f redis-config.yaml
    sudo rm -r /storage/data*
fi
kubectl config set-context --current --namespace=default
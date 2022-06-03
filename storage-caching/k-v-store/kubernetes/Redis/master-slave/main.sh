#!/bin/bash

kubectl create ns redis
kubectl config set-context --current --namespace=redis

kubectl apply -f ../sc.yaml
kubectl apply -f pv.yaml
kubectl apply -n redis -f redis-config.yaml
kubectl apply -n redis -f redis-master-slave.yaml

kubectl -n redis delete svc redis
kubectl -n redis delete statefulset redis
kubectl delete -n redis pvc --all
kubectl delete -f pv.yaml
kubectl delete -f sc.yaml
kubectl delete -n redis -f redis-config.yaml
sudo rm -r /storage/data*

kubectl config set-context --current --namespace=default
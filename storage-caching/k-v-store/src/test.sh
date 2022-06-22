#!/bin/bash

kubectl apply -f manager/configmap.yaml
kubectl delete -f manager/deployment.yaml
docker rmi -f alnairpod:manager
docker rmi -f zhuangweikang/alnairpod:manager
docker build -t alnairpod:manager -f manager/Dockerfile .
docker tag alnairpod:manager zhuangweikang/alnairpod:manager
docker push zhuangweikang/alnairpod:manager
kubectl apply -f manager/deployment.yaml

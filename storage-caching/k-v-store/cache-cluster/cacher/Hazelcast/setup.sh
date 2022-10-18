#!/bin/bash

if [ $1 == "init" ] 
then
    kubectl apply -f rbac.yaml
    kubectl apply -f hazelcast-jet-config.yaml
    kubectl apply -f hazelcast-jet.yaml
    kubectl run hzmanager --image hazelcast/management-center:latest-snapshot --port=8080
    kubectl expose pod hzmanager --type=NodePort --port=8080
elif [ $1 == "del" ] 
then
    kubectl delete -f .
    kubectl delete pod hzmanager
    kubectl delete svc hzmanager
fi
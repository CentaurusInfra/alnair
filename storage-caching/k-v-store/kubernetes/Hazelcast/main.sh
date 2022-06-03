#!/bin/bash

if [ $1 == "init" ] 
then
    kubectl apply -f rbac.yaml
    kubectl apply -f hazelcast-jet-config.yaml
    kubectl apply -f hazelcast-jet.yaml
    kubectl run manager --image hazelcast/management-center:latest-snapshot --port=8080
    kubectl expose pod manager --type=NodePort --port=8080
elif [ $1 == "del" ] 
then
    kubectl delete -f .
    kubectl delete pod manager
    kubectl delete svc manager
fi
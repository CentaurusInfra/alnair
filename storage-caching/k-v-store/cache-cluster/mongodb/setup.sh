#!/bin/bash

if [ $1 == "init" ]
then
    sudo mkdir -p /mnt/data
    NODES=$(kubectl get nodes | awk '(NR>1) { print $1 }')
    kubectl label nodes --overwrite $NODES db=mongo
    kubectl apply -k .
    sleep 20
    while [[ $(kubectl get pod mongo-0 | awk '{print $3}' | tail -n 1) != "Running" ]]
    do
        sleep 3
    done
    
    kubectl cp init.js mongo-0:/tmp
    kubectl exec -it mongo-0 -n default -- mongo admin < /tmp/init.js
elif [ $1 == "del" ]
then
    kubectl delete -k .
    sudo rm -r /mnt/data
fi

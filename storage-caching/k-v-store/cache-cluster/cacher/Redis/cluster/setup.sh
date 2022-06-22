#!/bin/bash

if [ $1 == "init" ]
then
    kubectl apply -f ../sc.yaml
    kubectl apply -f pv.yaml
    kubectl apply -f redis-config.yaml
    kubectl apply -f redis-cluster.yaml
    kubectl apply -f redis-cluster-proxy-config.yaml
    sleep 10
    serverIPs=()
    for((i=2;i<5;i++));
    do
        ip=$(kubectl get pods -o wide | grep redis-cluster | awk '{ print $6 }')
        serverIPs+=($ip)
    done
    kubectl exec -it redis-cluster-0 -- redis-cli --pass redispwd --cluster create "${serverIPs[0]}":6379 "${serverIPs[1]}":6379 "${serverIPs[2]}":6379 --cluster-replicas 0
    kubectl apply -f redis-cluster-proxy-deploy.yaml
elif [ $1 == "del" ]
then
    kubectl delete -f .
    kubectl delete -f ../sc.yaml
    sudo rm -r /storage/data*
fi

#!/bin/bash

if [ $1 == "init" ]
then
    kubectl apply -f secret.yaml
    kubectl apply -f ../sc.yaml
    kubectl apply -f pv.yaml
    kubectl apply -f config.yaml
    kubectl apply -f cluster.yaml
    sleep 20
    serverIPs=()
    for((i=2;i<5;i++));
    do
        ip=$(kubectl get pods -o wide | grep redis-cluster | awk '{ print $6 }')
        serverIPs+=($ip)
    done
    kubectl exec -it redis-cluster-0 -- redis-cli --pass redispwd --cluster create "${serverIPs[0]}":6379 "${serverIPs[1]}":6379 "${serverIPs[2]}":6379 --cluster-replicas 0
    kubectl apply -f envoy-proxy-config.yaml
    kubectl apply -f envoy-proxy-deploy.yaml
elif [ $1 == "del" ]
then
    kubectl delete -f .
    kubectl delete -f ../sc.yaml
    sudo rm -r /storage/data*
fi

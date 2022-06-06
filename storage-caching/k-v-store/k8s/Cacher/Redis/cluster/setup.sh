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
    kubectl apply -f redis-config.yaml
    kubectl apply -f redis-cluster.yaml
    kubectl apply -f redis-cluster-proxy-config.yaml

    serverIPs=()
    for((i=2;i<5;i++));
    do
        ip=$(kubectl get pods -o wide | awk '{ print $6 }' | sed -n "$i p")
        serverIPs+=($ip)
    done
    kubectl exec -it redis-cluster-0 -- redis-cli --pass redispwd --cluster create "${serverIPs[0]}":6379 "${serverIPs[1]}":6379 "${serverIPs[2]}":6379 --cluster-replicas 0
    kubectl apply -f redis-cluster-proxy-deploy.yaml  
elif [ $1 == "del" ] 
then
    kubectl delete svc --all
    kubectl delete statefulset --all
    kubectl delete pvc --all
    kubectl delete pv --all
    kubectl delete -f ../sc.yaml
    kubectl delete -f redis-config.yaml
    kubectl delete -f redis-cluster-proxy-config.yaml
    kubectl delete -f redis-cluster-proxy-deploy.yaml
    sudo rm -r /storage/data*
fi
kubectl config set-context --current --namespace=default 
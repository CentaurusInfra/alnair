#!/bin/bash

kubectl apply -f https://raw.githubusercontent.com/hazelcast/hazelcast/master/kubernetes-rbac.yaml
kubectl run hz1 --image="hazelcast/hazelcast:latest" --port=5701 --overrides='{"apiVersion": "v1", "spec": {"nodeSelector": { "hz_cluster": "1" }}}' --labels="role=server"
kubectl run hz2 --image="hazelcast/hazelcast:latest" --port=5701 --overrides='{"apiVersion": "v1", "spec": {"nodeSelector": { "hz_cluster": "2" }}}' --labels="role=server"
kubectl run manager --image hazelcast/management-center:latest-snapshot --port=8080
kubectl expose pod manager --type=NodePort --port=8080
kubectl expose pod hz1 hz2 --type=NodePort --selector="role=server" --port=5701

kubectl delete svc --all
kubectl delete pods --all

HAZELCAST_VERSION=latest
kubectl apply -f https://raw.githubusercontent.com/hazelcast/hazelcast-kubernetes/master/rbac.yaml
kubectl create service loadbalancer hz-hazelcast-0 --tcp=5701
kubectl run hz-hazelcast-0 --image=hazelcast/hazelcast:$HAZELCAST_VERSION --port=5701 -l "app=hz-hazelcast-0,role=hazelcast"
kubectl create service loadbalancer hz-hazelcast-1 --tcp=5701
kubectl run hz-hazelcast-1 --image=hazelcast/hazelcast:$HAZELCAST_VERSION --port=5701 -l "app=hz-hazelcast-1,role=hazelcast"
kubectl create service loadbalancer hz-hazelcast --tcp=5701 -o yaml --dry-run=client | kubectl set selector --local -f - "role=hazelcast" -o yaml | kubectl create -f -


kubectl run manager --image hazelcast/management-center:latest-snapshot --port=8080
kubectl expose pod manager --type=NodePort --port=8080


HAZELCAST_VERSION=latest
kubectl apply -f https://raw.githubusercontent.com/hazelcast/hazelcast-kubernetes/master/rbac.yaml
kubectl create service nodeport hz-hazelcast-0 --tcp=5701:5701 --node-port=31001
kubectl run hz-hazelcast-0 --image=hazelcast/hazelcast:$HAZELCAST_VERSION --port=5701 -l "app=hz-hazelcast-0,role=hazelcast"
kubectl create service nodeport hz-hazelcast-1 --tcp=5701:5701 --node-port=31002
kubectl run hz-hazelcast-1 --image=hazelcast/hazelcast:$HAZELCAST_VERSION --port=5701 -l "app=hz-hazelcast-1,role=hazelcast"
kubectl create service nodeport hz-hazelcast --tcp=5701:5701 --node-port=31000 -o yaml --dry-run=client | kubectl set selector --local -f - "role=hazelcast" -o yaml | kubectl create -f -

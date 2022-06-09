#!/bin/bash


NODES=$(kubectl get nodes | awk '(NR>1) { print $1 }')
kubectl label nodes --overwrite $NODES db=mongo
kubectl apply -k .
sleep 5
kubectl exec -it mongo-0 -n default -- mongo

# rs.initiate()
# var cfg = rs.conf()
# cfg.members[0].host="mongo-0.mongo:27017"
# rs.reconfig(cfg)
# rs.add("mongo-1.mongo:27017")
# rs.status()

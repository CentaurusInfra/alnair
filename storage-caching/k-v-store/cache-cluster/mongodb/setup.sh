#!/bin/bash

if [ $1 == "init" ]
then
    sudo mkdir -p /mnt/data
    NODES=$(kubectl get nodes | awk '(NR>1) { print $1 }')
    kubectl label nodes --overwrite $NODES db=mongo
    kubectl apply -k .
    sleep 5
    kubectl exec -it mongo-0 -n default -- mongo admin
    # TODO: Execute the following commands in mongodb
    # rs.initiate()
    # var cfg = rs.conf()
    # cfg.members[0].host="mongo-0.mongo:27017"
    # rs.reconfig(cfg)
    # rs.add("mongo-1.mongo:27017")
    # rs.status()
    # db.createUser(
    # {
    #     user: "alnair",
    #     pwd: "alnair",
    #     roles: [
    #             { role: "userAdminAnyDatabase", db: "admin" },
    #             { role: "readWriteAnyDatabase", db: "admin" },
    #             { role: "dbAdminAnyDatabase", db: "admin" },
    #             { role: "clusterAdmin", db: "admin" }
    #         ]
    # })
elif [ $1 == "del" ]
then
    kubectl delete -k .
    sudo rm -r /mnt/data
fi

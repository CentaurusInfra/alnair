#!/bin/bash

# install whereouts
crds="https://github.com/k8snetworkplumbingwg/whereabouts/blob/master/doc/crds"
kubectl apply -f $crds/daemonset-install.yaml
kubectl apply -f $crds/whereabouts.cni.cncf.io_ippools.yaml
kubectl apply -f $crds/whereabouts.cni.cncf.io_overlappingrangeipreservations.yaml

# install Multus
kubectl apply -f https://github.com/k8snetworkplumbingwg/multus-cni/blob/master/deployments/multus-daemonset-thick-plugin.yml
cat <<EOF | kubectl create -f -
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
  name: macvlan-conf
spec:
  config: '{
      "cniVersion": "0.3.1",
      "name": "macvlan-conf",
      "type": "macvlan",
      "master": "enp4s0f0",
      "mode": "bridge",
      "ipam": {
        "type": "whereabouts",
        "range": "10.10.0.0/16"
      }
    }'
EOF
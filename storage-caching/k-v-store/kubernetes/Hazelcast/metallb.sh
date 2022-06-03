kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.12.1/manifests/namespace.yaml
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.12.1/manifests/metallb.yaml

sudo cat <<EOF | kubectl create -f -
apiVersion: v1
kind: ConfigMap
metadata:
    namespace: metallb-system
    name: config
data:
    config: |
        address-pools:
        - name: default
          protocol: layer2
          addresses:
          - 10.145.31.30-10.145.31.40
EOF

kubectl delete -n metallb-system -f https://raw.githubusercontent.com/metallb/metallb/v0.12.1/manifests/metallb.yaml
kubectl delete -n metallb-system -f https://raw.githubusercontent.com/metallb/metallb/v0.12.1/manifests/namespace.yaml
kubectl delete -n metallb-system configmap config
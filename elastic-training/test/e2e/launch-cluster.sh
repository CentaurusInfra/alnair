#!/bin/bash
set -euo pipefail

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $CWD
BIN="./../../bin"
mkdir -p $BIN
K8S_VERSION=1.21.1
KUBECTL="$BIN/kubectl"
KIND_VERSION=0.11.0
KIND="$BIN/kind"
JQ="$BIN/jq"

[ -e "$KUBECTL" ] || curl -sL -o "$KUBECTL" "https://storage.googleapis.com/kubernetes-release/release/v${K8S_VERSION}/bin/linux/amd64/kubectl"
chmod +x $KUBECTL

[ -e "$KIND" ] || curl -sL -o "$KIND" "https://kind.sigs.k8s.io/dl/v${KIND_VERSION}/kind-linux-amd64"
chmod +x $KIND

[ -e "$JQ" ] || curl -sL -o "$JQ" "https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64"
chmod +x $JQ

if $KIND get clusters | grep -q kind; then
  echo "The test cluster already exists, skip creating"
else
  echo "Creating a test kind cluster"
  $KIND create cluster --config ./kind-config.yaml --wait 10m
fi

SECRET=$($KUBECTL get serviceaccount node-controller -o json -n kube-system | $JQ -Mr '.secrets[].name')
TOKEN=$($KUBECTL get secrets $SECRET -n kube-system -o json | $JQ -Mr '.data.token' | base64 -d)
PORT=$(docker ps -a | grep kind-control-plane | awk '{print $(NF-1)}' | cut -d: -f2 | cut -d- -f1)

# kubectl has a bug (or by design) that it cannot patch the status of a node.
# A workaround is to directly cURL the node status endpoint.
# See https://github.com/kubernetes/kubernetes/issues/67455.
patchGPU () {
  local nodename=$1
  local numGPUs=$2
  local response_code=$( \
    curl -sk -w '%{http_code}' -o /dev/null -X PATCH \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json-patch+json" \
    -d '[{"op": "add", "path": "/status/capacity/nvidia.com~1gpu", "value": "'${numGPUs}'"}]' \
    "https://localhost:${PORT}/api/v1/nodes/${nodename}/status" \
  )
  if [ "$response_code" = "200" ]; then
    echo "Successfully patched $numGPUs GPU devices to node $nodename"
  else
    echo "Error code in patching $numGPUS GPU devices to node $nodename: $response_code"
    exit 1
  fi
}

patchGPU kind-worker 4
patchGPU kind-worker2 4
echo "FINISHED launching a test cluster with mock GPU devices"


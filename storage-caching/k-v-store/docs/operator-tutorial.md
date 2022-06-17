# install GO
```shell
wget https://go.dev/dl/go1.18.3.linux-amd64.tar.gz
tar -xzf go1.18.3.linux-amd64.tar.gz
echo "export PATH=$PATH:$HOME/go/bin" > $HOME/.profile
source $HOME/.profile
go version
```

# install controller runtime
```shell
git clone https://github.com/kubernetes/kubernetes.git
cp -R  kubernetes/staging/src/k8s.io $GOPATH/src/k8s.io
GOPATH=$(go env GOPATH)
mkdir $GOPATH/src/sigs.k8s.io
git clone https://github.com/kubernetes-sigs/controller-runtime $GOPATH/src/sigs.k8s.io
```

# install operator-sdk
```shell
export ARCH=$(case $(uname -m) in x86_64) echo -n amd64 ;; aarch64) echo -n arm64 ;; *) echo -n $(uname -m) ;; esac)
export OS=$(uname | awk '{print tolower($0)}')
export OPERATOR_SDK_DL_URL=https://github.com/operator-framework/operator-sdk/releases/download/v1.22.0
curl -LO ${OPERATOR_SDK_DL_URL}/operator-sdk_${OS}_${ARCH}
gpg --keyserver keyserver.ubuntu.com --recv-keys 052996E2A20B5C7E
curl -LO ${OPERATOR_SDK_DL_URL}/checksums.txt
curl -LO ${OPERATOR_SDK_DL_URL}/checksums.txt.asc
gpg -u "Operator SDK (release) <cncf-operator-sdk@cncf.io>" --verify checksums.txt.asc
grep operator-sdk_${OS}_${ARCH} checksums.txt | sha256sum -c -
chmod +x operator-sdk_${OS}_${ARCH} && sudo mv operator-sdk_${OS}_${ARCH} /usr/local/bin/operator-sdk
```

# create an operatoe
```shell
unset GOPATH
operator-sdk init dltpod-operator --repo=github.com/CentaurusInfra/alnair/tree/main/storage-caching/k-v-store/dltp-operator
operator-sdk create api --group alnair.com --version=v1alpha1 --kind=DLTPod --resource=true --controller=true

make generate
make manifests
```
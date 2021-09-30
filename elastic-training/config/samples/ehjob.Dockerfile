FROM horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1
RUN apt update && apt install -y dnsutils
RUN pip install tqdm
COPY scripts/discover_hosts.sh scripts/
COPY elastic-pyscripts/* examples/

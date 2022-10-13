FROM python:3
RUN pip3 install requests

COPY __init__.py /__init__.py
COPY crd-operator /crd-operator
COPY cache_manager.py /cache_manager.py
COPY create-alnair-datacache-resource /create-alnair-datacache-resource
COPY deploy-proxy /deploy-proxy

RUN apt update && \
      apt install -y vim curl && \
            curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl && \
            chmod +x ./kubectl && \
            mv ./kubectl /usr/local/bin/kubectl

ENTRYPOINT [ "python", "/crd-operator" ]

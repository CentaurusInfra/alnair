FROM nvcr.io/nvidia/pytorch:21.09-py3
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="America/San_Francisco" apt-get install -y tzdata


RUN mkdir /root/{data,logs,scripts}

RUN pip3 install jupyterlab==3.5.0
# RUN echo "http://dl-cdn.alpinelinux.org/alpine/v3.12/main/" >> /etc/apk/repositories \
#     && apk add --update nodejs=12.22.12-r0 npm=12.22.12-r0 other-package
# RUN pip3 install jupyterlab_nvdashboard
# RUN jupyter labextension install jupyterlab-nvdashboard

EXPOSE 8888
WORKDIR /root/scripts

COPY  test.txt ./
COPY strrev_gds.co ./


COPY test/*.py ./ 
COPY test/git-lfs-linux-amd64-v3.2.0.tar.gz ./ 
RUN tar -zxvf ./git-lfs-linux-amd64-v3.2.0.tar.gz 
RUN cd git-lfs-3.2.0
RUN bash git-lfs-3.2.0/install.sh
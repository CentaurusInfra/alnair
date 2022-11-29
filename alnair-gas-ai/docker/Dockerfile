FROM nvcr.io/nvidia/magnum-io/magnum-io:21.07
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="America/San_Francisco" apt-get install -y tzdata

RUN apt-get update && apt-get install -y git vim 

RUN mkdir /root/{data,logs,scripts}
RUN apt-get install -y python3-pip

RUN pip install jupyterlab==2.2
# RUN echo "http://dl-cdn.alpinelinux.org/alpine/v3.12/main/" >> /etc/apk/repositories \
#     && apk add --update nodejs=12.22.12-r0 npm=12.22.12-r0 other-package
# RUN pip3 install jupyterlab_nvdashboard
# RUN jupyter labextension install jupyterlab-nvdashboard


EXPOSE 8888
WORKDIR /root/scripts

COPY  test.txt ./
COPY strrev_gds.co ./

COPY mnist ./
RUN mkdir mnist_data
COPY mnist_data ./mnist_data

COPY test/*.py ./ 
COPY test/git-lfs-linux-amd64-v3.2.0.tar.gz ./ 
RUN tar -zxvf ./git-lfs-linux-amd64-v3.2.0.tar.gz 
RUN cd git-lfs-3.2.0
RUN bash git-lfs-3.2.0/install.sh
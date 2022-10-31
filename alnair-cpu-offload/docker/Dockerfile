FROM nvcr.io/nvidia/pytorch:22.10-py3
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="America/San_Francisco" apt-get install -y tzdata

RUN apt-get update && apt-get install -y git vim 
# RUN apt-get install -y cuda-toolkit-11-4


RUN pip3 install scipy dataset pytorch-ignite matplotlib seaborn
RUN mkdir /root/{data,logs,scripts}

RUN apt-get install -y nodejs
RUN pip3 install jupyter matplotlib
RUN pip3 install jupyter_http_over_ws ipykernel nbformat
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN pip3 install jupyterlab==3.5.0
# RUN jupyter --version
# RUN jupyter labextension install jupyterlab_tensorboard
RUN pip3 install jupyter_tensorboard torchvision scikit-image
RUN pip3 install jupyterlab_nvdashboard

# # tensorboard
# EXPOSE 6006
# jupyter notebook
# RUN pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
# RUN pip3 install -U scikit-learn scipy

# RUN apt update
# RUN apt install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt install -y python3.9
EXPOSE 8888
WORKDIR /root/scripts

# COPY start.sh /

# CMD ["/start.sh"]


# ENV CU_HOOK_DEBUG=1

COPY test/*.py ./ 
COPY test/git-lfs-linux-amd64-v3.2.0.tar.gz ./ 
RUN tar -zxvf ./git-lfs-linux-amd64-v3.2.0.tar.gz 
RUN cd git-lfs-3.2.0
RUN bash git-lfs-3.2.0/install.sh
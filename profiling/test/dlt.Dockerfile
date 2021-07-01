FROM tensorflow/tensorflow:nightly-gpu

RUN apt-get update && apt-get install -y git vim

#download tensorflow models and sample training file                      
RUN git clone https://github.com/tensorflow/models.git 
RUN python3 -m pip install --upgrade pip
ENV PYTHONPATH="$PYTHONPATH:/models"
RUN pip3 install -r /models/official/requirements.txt
RUN pip3 install torch==1.8.1 torchvision==0.9.1
RUN mkdir /tmp/{model,data,logs,scripts}
WORKDIR /tmp/scripts
COPY ./resnet-cifar10-tf2.py .
COPY ./resnet-cifar10-pytorch.py .
COPY ./resnet_imagenet.sh .
COPY ./data_dump.py .	
CMD ["tail", "-f", "/dev/null"]

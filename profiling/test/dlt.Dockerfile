FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y git vim

#download tensorflow models and sample training file                      
RUN git clone https://github.com/tensorflow/models.git 
RUN python3 -m pip install --upgrade pip
ENV PYTHONPATH="$PYTHONPATH:/models"
RUN pip3 install -r /models/official/requirements.txt

RUN mkdir /tmp/{model,data,logs,scripts}
WORKDIR /tmp/scripts
COPY ./resnet-cifar10.py .
CMD ["tail", "-f", "/dev/null"]

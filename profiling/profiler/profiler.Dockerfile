FROM nvidia/cuda:11.4.0-base-ubuntu20.04

RUN apt-get update && apt-get install -y \
        vim \
        curl \
        python3 \
        python3-pip 
RUN python3 -m pip install --upgrade pip
RUN pip3 install prometheus-api-client kubernetes statsmodels sklearn
RUN pip3 install nvidia-ml-py3 pynvml pymongo

RUN mkdir /app
WORKDIR /app
COPY ./app.py .
COPY ./metadata_store.py .
COPY ./nvml_gpu_watch.py .
COPY ./util.py .
CMD ["bash", "-c", "python3 app.py"]

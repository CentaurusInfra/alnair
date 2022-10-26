FROM ubuntu:20.04

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
COPY ./mongo_upsert.py .
COPY ./pod_event_watch.py .
COPY ./prometheus_query.py .
COPY ./util.py .
CMD ["bash", "-c", "python3 app.py"]

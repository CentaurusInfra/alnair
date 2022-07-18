# FROM python:3.8-alpine
FROM ubuntu:20.04
RUN apt update && apt install -y python3-pip
RUN mkdir app
WORKDIR /app
COPY manager ./
COPY grpctool ./grpctool
RUN pip install -r requirements.txt
RUN chmod +x *
CMD [ "python3", "manager.py" ]
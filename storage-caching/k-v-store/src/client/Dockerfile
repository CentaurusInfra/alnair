FROM ubuntu:20.04
RUN apt update && apt install -y python3-pip
RUN mkdir app
WORKDIR /app
COPY client ./
COPY grpctool ./grpctool
RUN pip3 install -r requirements.txt
RUN chmod +x *
RUN apt update && apt install -y nano nvidia-utils-460
CMD [ "python3", "client.py" ]
#use ubuntu as build image, golang alpine cause symbol not found, binary cannot executed
FROM ubuntu:20.04 as build

RUN apt-get update
RUN apt-get install -y wget git gcc vim
RUN wget -P /tmp https://dl.google.com/go/go1.17.6.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf /tmp/go1.17.6.linux-amd64.tar.gz
RUN rm /tmp/go1.17.6.linux-amd64.tar.gz
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"

RUN mkdir /alnair-device-plugin
WORKDIR /alnair-device-plugin
COPY . .
RUN go build -o /bin/alnair-device-plugin cmd/alnair-device-plugin/main.go
RUN go build -o /bin/alnair-vgpu-server cmd/vgpu-server/main.go

FROM debian:stretch-slim

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=utility
COPY --from=build /bin/alnair-device-plugin /bin/alnair-device-plugin
COPY --from=build /bin/alnair-vgpu-server /bin/alnair-vgpu-server
WORKDIR /bin
CMD ["alnair-device-plugin"]

FROM ubuntu:20.04 as build

RUN apt-get update && apt-get install -y vim git curl wget gcc

RUN wget -P /tmp https://dl.google.com/go/go1.18.3.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf /tmp/go1.18.3.linux-amd64.tar.gz
RUN rm /tmp/go1.18.3.linux-amd64.tar.gz
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"

# build locally
WORKDIR /alnair/
COPY . . 
RUN go build -o /bin/alnair-exporter main.go

# build from alnair repo 
#RUN git clone https://github.com/CentaurusInfra/alnair.git
#WORKDIR /alnair/
#RUN cd /alnair/alnair-profiler/alnair-exporter && go build -o /bin/alnair-exporter cmd/exporter/main.go

# need to based on a nvidia container toolkit image, so the gpu driver is exposed to container then nvml works
FROM nvidia/cuda:11.4.0-base-ubuntu20.04
COPY --from=build /bin/alnair-exporter /bin/alnair-exporter
WORKDIR /bin

CMD ["alnair-exporter"]

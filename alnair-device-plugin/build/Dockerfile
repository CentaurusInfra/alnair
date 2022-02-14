FROM debian:stretch-slim

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=utility

COPY bin/alnair-device-plugin /bin/alnair-device-plugin

WORKDIR /bin
CMD ["alnair-device-plugin"]

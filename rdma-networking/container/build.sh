#!/usr/bin/env bash
docker build -t rdma:ofed-cuda102-cudnn8-torch1.8-ompi5.0rc-vision0.9 - < ./Dockerfile

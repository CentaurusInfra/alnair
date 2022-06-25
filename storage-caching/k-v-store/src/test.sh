#!/bin/bash

if [ $1 == "manager" ]
then
    docker rmi -f alnairpod:manager
    docker rmi -f zhuangweikang/alnairpod:manager
    docker build -t alnairpod:manager -f manager/Dockerfile .
    docker tag alnairpod:manager zhuangweikang/alnairpod:manager
    docker push zhuangweikang/alnairpod:manager
else
    docker rmi -f alnairpod:client
    docker rmi -f zhuangweikang/alnairpod:client
    docker build -t alnairpod:client -f client/Dockerfile .
    docker tag alnairpod:client zhuangweikang/alnairpod:client
    docker push zhuangweikang/alnairpod:client  
fi
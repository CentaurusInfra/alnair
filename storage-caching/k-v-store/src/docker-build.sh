#!/bin/bash

if [ $1 == "manager" ]
then
    docker rmi -f alnairpod:manager
    docker rmi -f centaurusinfra/alnairpod:manager
    docker build -t alnairpod:manager -f manager/Dockerfile .
    docker tag alnairpod:manager centaurusinfra/alnairpod:manager
    docker push centaurusinfra/alnairpod:manager
else
    docker rmi -f alnairpod:client
    docker rmi -f centaurusinfra/alnairpod:client
    docker build -t alnairpod:client -f client/Dockerfile .
    docker tag alnairpod:client centaurusinfra/alnairpod:client
    docker push centaurusinfra/alnairpod:client  
fi
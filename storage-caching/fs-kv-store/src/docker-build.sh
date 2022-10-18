#!/bin/bash

if [ $1 == "manager" ]
then
    docker rmi -f alnairpod-dev:manager
    docker rmi -f centaurusinfra/alnairpod-dev:manager
    docker build -t alnairpod-dev:manager -f manager/Dockerfile .
    docker tag alnairpod-dev:manager centaurusinfra/alnairpod-dev:manager
    docker push centaurusinfra/alnairpod-dev:manager
else
    docker rmi -f alnairpod-dev:client
    docker rmi -f centaurusinfra/alnairpod-dev:client
    docker build -t alnairpod-dev:client -f client/Dockerfile .
    docker tag alnairpod-dev:client centaurusinfra/alnairpod-dev:client
    docker push centaurusinfra/alnairpod-dev:client  
fi
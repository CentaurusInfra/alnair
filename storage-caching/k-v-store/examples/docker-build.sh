docker rmi -f $1:latest
docker rmi -f centaurusinfra/$1:latest
docker build -t $1:latest -f imagenet/Dockerfile .
docker tag $1:latest centaurusinfra/$1:latest
docker push centaurusinfra/$1:latest
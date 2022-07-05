docker rmi -f $1:latest
docker rmi -f zhuangweikang/$1:latest
docker build -t $1:latest -f $1/Dockerfile .
docker tag $1:latest zhuangweikang/$1:latest
docker push zhuangweikang/$1:latest
# build and push image to personal docker Hub address
sudo docker build -f profiler.Dockerfile -t fizzbb/profiler:v1.1 .
sudo docker push fizzbb/profiler:v1.1

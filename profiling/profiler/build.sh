# build and push image to personal docker Hub address
sudo docker build -f profiler.Dockerfile -t fizzbb/profiler:v0.1 .
# need $docker login -u XXX first
sudo docker push fizzbb/profiler:v0.1

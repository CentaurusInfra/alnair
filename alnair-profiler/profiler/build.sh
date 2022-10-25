# build and push image to personal docker Hub address
sudo docker build -f profiler.Dockerfile -t centaurusinfra/profiler:0.5.0 .
# need $docker login -u XXX first
sudo docker push centaurusinfra/profiler:0.5.0

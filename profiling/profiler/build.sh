# build and push image to personal docker Hub address
sudo docker build -f profiler.Dockerfile -t centaurusinfra/profiler:latest .
# need $docker login -u XXX first
sudo docker push centaurusinfra/profiler:latest

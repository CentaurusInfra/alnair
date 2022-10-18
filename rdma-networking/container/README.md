# Dockerfile

## The way to trigger the container:
```
docker run -v /sys/class/:/sys/class/ -e "LD_LIBRARY_PATH=/usr/lib;/root/anaconda3/envs/p38/lib:/root/anaconda3/lib" -v /dev/:/dev/ --privileged --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm -it -d <img:tag>  /bin/bash
```


## Run training task
```
docker exec -it <container> /bin/bash

# conda activate p38

# git clone https://github.com/CentaurusInfra/alnair.git
# cd alnai/test/distrubuted-training
... /* change IP  */ ...
# python mnist-distribured.py -n 1 -g 4 -i 0

```

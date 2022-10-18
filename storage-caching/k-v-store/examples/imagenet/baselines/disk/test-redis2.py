import pickle
import numpy as np
import redis
import time
import sys

if __name__=="__main__":
    client = redis.RedisCluster(host="10.244.1.4", port=6379, password='redispwd')
    with open('keys.txt', 'r') as f:
        keys = f.readlines()
    

    for k in keys:
        k = k.strip('\n')
        client.set(k, k)
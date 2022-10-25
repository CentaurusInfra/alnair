import pickle
import numpy as np
import redis
import time
import sys

if __name__=="__main__":
    client = redis.RedisCluster(host="10.244.1.4", port=6379, password='redispwd')
    with open('keys.txt', 'r') as f:
        keys = f.readlines()
    

    # for k in keys:
    #     k = k.strip('\n')
    #     client.set(k, k)
        
    latency = []
    throughput = []
    for k in keys:
        k = k.strip('\n')
        t = time.time()
        raw_val = client.get(k)
        val = pickle.loads(raw_val)
        e = time.time()
        latency.append(e-t)
        throughput.append(sys.getsizeof(raw_val)/(e-t))
    
    print(np.mean(latency)*1000)
    print(np.mean(throughput))
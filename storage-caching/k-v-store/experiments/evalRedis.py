#!/usr/bin/python3
import logging
import time
import numpy as np
import redis
import boto3

import warnings
warnings.filterwarnings('ignore')

# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)


train_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
test_list = ["test_batch"]


def upload(client):
    bucket = 'zhuangwei-bucket'
    s3 = boto3.resource('s3')
    bucket_obj = s3.Bucket(bucket)
    for obj in bucket_obj.objects.all():
        key = obj.key
        body = obj.get()['Body'].read()
        client.set(key, body)

def measure(client):
    measures = []
    for _ in range(5):
        start = time.time()
        for chunk_index in range(len(train_list)):
            client.get(train_list[chunk_index])
        end = time.time()
        print(end-start)
        measures.append(end-start)
    print('\n>>>>>>>> duration: %.2fs +/- %.2fs\n' % (np.mean(measures), np.std(measures)))
    
    
if __name__=="__main__":
    standalone = False
    if standalone:
        client = redis.Redis(
            host="10.145.41.32",
            port=30007,
            password='redispwd'
        )
    else:
        client = redis.RedisCluster(host="10.145.41.33", port=30007)
    upload(client)
    measure(client)
    client.close()
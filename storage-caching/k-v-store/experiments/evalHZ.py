#!/usr/bin/python3
import logging
import sys
import time
import numpy as np
import hazelcast as hz
import boto3

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


train_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
test_list = ["test_batch"]
    

def upload(hz_map, dtype='b'):
    bucket = 'zhuangwei-bucket'
    s3 = boto3.resource('s3')
    bucket_obj = s3.Bucket(bucket)
    for obj in bucket_obj.objects.all():
        key = obj.key
        body = obj.get()['Body'].read()
        if dtype == 's':
            hz_map.put_if_absent(key, body.decode('latin1'))
        else:
            hz_map.put_if_absent(key, bytearray(body))

def measure(hz_map):
    measures = []
    for _ in range(5):
        start = time.time()
        for chunk_index in range(len(train_list)):
            if asyc:
                hz_map.get(train_list[chunk_index]).result()
            else:
                hz_map.get(train_list[chunk_index])
        end = time.time()
        print(end-start)
        measures.append(end-start)
    print('\n>>>>>>>> duration: %.2fs +/- %.2fs\n' % (np.mean(measures), np.std(measures)))
    
    
if __name__=="__main__":
    # Start the Hazelcast Client and connect to an already running Hazelcast Cluster on 127.0.0.1
    hz_client = hz.HazelcastClient(
        cluster_name="dev",
        cluster_members=["10.145.41.32:31001", "10.145.41.32:31002"],
        use_public_ip=True,
        smart_routing=True,
        client_name='hz.client_0',
        lifecycle_listeners=[
            lambda state: print("Lifecycle event >>>", state),
        ],
        connection_timeout=120
    )

    # Get the Distributed Map from Cluster.
    asyc = True
    map = 'cifar10-map' 
    if asyc:
        my_map = hz_client.get_map(map)
    else:
        my_map = hz_client.get_map(map).blocking()
    if sys.argv[1] == 'init':
        upload(my_map, dtype='s')
        measure(my_map)
    elif sys.argv[1] == 'test':
        measure(my_map)
    if sys.argv[1] == 'del':
        my_map.clear()
        my_map.destroy()
    hz_client.shutdown()
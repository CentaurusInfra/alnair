#!/usr/bin/python3
import boto3
import os
import time
import numpy as np


def uploadDirectory(path, bucketname):
        for root,dirs,files in os.walk(path):
            for file in files:
                client.upload_file(os.path.join(root,file),bucketname,file)
                
                
if __name__=="__main__":
    bucket = 'zhuangwei-bucket'
    client = boto3.client('s3')
    s3 = boto3.resource('s3')
    bucket_obj = s3.Bucket(bucket)

    # create the bucket if not exist
    bnames = [item['Name'] for item in client.list_buckets()['Buckets']]
    if bucket not in bnames:
        response = client.create_bucket(Bucket=bucket)
        print(response)
    # uploadDirectory('./data', bucket)
    
    measures = []
    for _ in range(5):
        start = time.time()
        for obj in bucket_obj.objects.all():
            key = obj.key
            if 'data_batch' in key:
                # operation: directly load data to memory 
                print('reading %s, size: %fMB' % (key, obj.size/1024/1024))
                body = obj.get()['Body'].read()
                print('-------------')
        end = time.time()
        measures.append(end-start)
    print('\n>>>>>>>> duration: %.2fs +/- %.2fs\n' % (np.mean(measures), np.std(measures)))
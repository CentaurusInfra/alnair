#! /usr/bin/python3

import boto3

bkname = 'zhuangwei-bucket'
client = boto3.client('s3')
s3 = boto3.resource('s3')

paginator = client.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket='zhuangwei-bucket')

keys = []
for bucket in page_iterator:
    for file in bucket['Contents']:
        keys.append(file['Key'])
        try:
            metadata = client.head_object(Bucket='zhuangwei-bucket', Key=file['Key'])
            print(metadata)
        except:
            print("Failed {}".format(file['Key']))
            
# for k in keys:
#     obj = s3.Object(bkname, k)
#     print(k, obj.get())
#     break
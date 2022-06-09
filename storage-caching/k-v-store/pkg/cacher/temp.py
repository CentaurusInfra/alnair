#! /usr/bin/python3

import boto3
s3client = boto3.client('s3')

paginator = s3client.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket='zhuangwei-bucket')

for bucket in page_iterator:
    for file in bucket['Contents']:
        try:
            metadata = s3client.head_object(Bucket='zhuangwei-bucket', Key=file['Key'])
            print(metadata)
        except:
            print("Failed {}".format(file['Key']))
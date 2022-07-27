import logging
from google.protobuf.timestamp_pb2 import Timestamp
import pickle
import hashlib
import boto3


def get_logger(name=__name__, level:str ='INFO', file=None):
    levels = {"info": logging.INFO, "error": logging.ERROR, "debug": logging.DEBUG}
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
    logger = logging.getLogger(name)
    logger.setLevel(levels[level.lower()])

    cl = logging.StreamHandler()
    cl.setLevel(levels[level.lower()])
    cl.setFormatter(formatter)
    logger.addHandler(cl)
    
    if file is not None:
        fl = logging.FileHandler(file)
        fl.setLevel(levels[level.lower()])
        fl.setFormatter(formatter)
        logger.addHandler(fl)
    return logger

grpc_ts = lambda ts: Timestamp(seconds=int(ts), nanos=int(ts % 1 * 1e9))

def hashing(data):
    if type(data) is not bytes:
        data = pickle.dumps(data)
    return hashlib.sha256(data).hexdigest()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def parse_config(section: str) -> dotdict:
        with open("/config/{}".format(section), 'r') as f:
            config_str = f.readlines()
        result = {}
        for item in map(lambda x: x.split("="), config_str):
            result[item[0]] = item[1]
        return dotdict(result)

def MessageToDict(message):
    message_dict = {}
    
    for descriptor in message.DESCRIPTOR.fields:
        key = descriptor.name
        value = getattr(message, descriptor.name)
        
        if descriptor.label == descriptor.LABEL_REPEATED:
            message_list = []
            
            for sub_message in value:
                if descriptor.type == descriptor.TYPE_MESSAGE:
                    message_list.append(MessageToDict(sub_message))
                else:
                    message_list.append(sub_message)
            
            message_dict[key] = message_list
        else:
            if descriptor.type == descriptor.TYPE_MESSAGE:
                message_dict[key] = MessageToDict(value)
            else:
                message_dict[key] = value
    
    return message_dict


class S3Helper:
    def __init__(self, s3auth: dict):
        session = boto3.Session(
            aws_access_key_id=s3auth['aws_access_key_id'],
            aws_secret_access_key=s3auth['aws_secret_access_key'],
            region_name=s3auth['region_name']
        )
        self.client = session.client('s3')
        s3 = session.resource('s3')
        
    def list_objects(self, bucket_name, prefix):
        paginator = self.client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        results = []
        for page in pages:
            for info in page['Contents']:
                results.append(info)
        return results
    
    def get_object(self, bucket_name, key):
        return self.client.get_object(Bucket=bucket_name, Key=key)['Body'].read()
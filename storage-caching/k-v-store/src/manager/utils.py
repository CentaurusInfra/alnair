import ast
import logging
from google.protobuf.timestamp_pb2 import Timestamp
import pickle
import hashlib


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
    

def parse_redis_conf(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    configs = []
    results = {}
    for l in lines:
        if len(l) == 0 or len(l.strip().split()) != 2 or l.strip()[0] == '#':
            continue
        configs.append(l)
        k, v = l.strip().split()
        k = k.strip().replace('\n', '')
        v = v.strip().replace('\n', '')            
        results[k] = v
    return results

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
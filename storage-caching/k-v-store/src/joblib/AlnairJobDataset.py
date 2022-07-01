import os
import json
from typing import Any, List, Tuple
import redis
from torch.utils.data import Dataset


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

class AlnairJobDataset(Dataset):
    def __init__(self, keys: List[str] = None):
        """An abstract class subclassing the torch.utils.data.Dataset class
        
        All datasets that represent a map from keys to data samples should subclass
        it. All subclasses should overwrite :meth:`__preprocess__`, supporting pre-processing loaded data. 
        Subclasses should also overwrite meth:`__getitem__`, supporting fetching a
        data sample for a given key. Subclasses could also optionally overwrite
        :meth:`__len__`, which is expected to return the size of the dataset by many
        :class:`~torch.utils.data.Sampler` implementations and the default options
        of :class:`~AlnairJobDataLoader`.
        
        .. note::
        Subclassing ~AlnairJobDataset will load data under provided keys from Redis to var:`self.__data` as Map<Key, Value>.
        Overwriting meth:`__preprocess__` allows you to replace var:`self.__data` and var:`self.__targets` with
        iteratable variables that can be iterated in meth:`__get_item__`.
        
        Args:
            keys (List, optional): a list of bucket keys. Defaults to None, meaning loading all keys in the bucket.
        """
        
        self.jobname = os.environ.get('JOBNAME')
        self.keys = keys
        self.qos = self.load_metainfo()['qos']
        self.jobinfo = self.load_jobinfo()
        if self.qos['UseCache']: # use datasource from Redis
            r = dotdict(self.jobinfo.jinfo).redisauth
            r = dotdict(r)
            self.client = redis.Redis(host=r.host, port=r.port, username=r.username, password=r.password)
            self.chunks = self.get_all_redis_keys()
        else:
            import configparser, boto3
            
            parser = configparser.ConfigParser()
            parser.read('/secret/client.conf')
            s3auth = parser['aws_s3']
            s3_session = boto3.Session(
                aws_access_key_id=s3auth['aws_access_key_id'],
                aws_secret_access_key=s3auth['aws_secret_access_key'],
                region_name=s3auth['region_name']
            )
            self.client = s3_session.client('s3')
            self.bucket_name = self.jobinfo['datasource']['bucket']
            self.chunks = self.get_all_s3_keys()
        
        self.start = 0
        self.loaded_chunks = []
        self.load_data()

    def load_metainfo(self):
        with open('/jobs/{}.json'.format(self.jobname), 'r') as f:
            jobinfo = json.load(f)
        return jobinfo
    
    def load_jobinfo(self):
        if self.qos['UseCache']:
            while True:
                try:
                    with open("/share/{}.json".format(self.jobname), 'rb') as f:
                        jobinfo = json.load(f)
                        break
                except (FileNotFoundError, json.decoder.JSONDecodeError):
                    pass
        else:
            jobinfo = self.load_metainfo()
        return dotdict(jobinfo)

    def get_all_redis_keys(self):
        chunks = dotdict(self.jobinfo.policy).chunkKeys
        if self.keys is not None:
            temp = []
            for chunk in chunks:
                for k in self.keys:
                    if chunk['name'].startswith(k):
                        temp.append(chunk)   
            chunks = temp
        return chunks
    
    def get_all_s3_keys(self):
        chunks = []
        for bk in self.jobinfo['datasource']['keys']:
            paginator = self.client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=bk)
            for page in pages:
                for item in page['Contents']:
                    chunks.append({'name': item['Key'], 'key': item['Key'], 'size': item['Size']})
        return chunks
    
    def load_chunksubset(self, start):
        if self.qos['MaxMemory'] == 0: # load all data into memory
            chunks = self.get_all_redis_keys() if self.qos['UseCache'] else self.get_all_s3_keys()
            return chunks, len(self.chunks)
        else:
            chunks = []
            total_size = 0
            i = start
            while i < len(self.chunks):
                total_size += int(self.chunks[i]['size'])
                if total_size > self.qos['MaxMemory']*1024*1024:
                    break
                else:
                    chunks.append(self.chunks[i])
                    i += 1
            return chunks, i
        
    @property
    def data(self):
        return self.__data
    @property
    def targets(self):
        return self.__targets
    
    def load_data(self):
        if len(self.loaded_chunks) == len(self.chunks):  # case 1: load all keys in memory
            return
        elif self.start < len(self.chunks):  # case 2: load a subset of keys in memory
            self.loaded_chunks, self.start = self.load_chunksubset(self.start)
            self.__data = {}
            for chunk in self.loaded_chunks:
                k = chunk['key']
                n = chunk['name']
                if self.qos['UseCache']:
                    while True:
                        value = self.client.get(k)
                        if value is None: # cache missing
                            with open('/share/cachemiss', 'w') as f:
                                f.writelines([k])
                        else:
                            break
                else:
                    value = self.client.get_object(Bucket=self.bucket_name, Key=k)['Body'].read()
                self.__data[n] = value
            
            if self.start == len(self.chunks):
                self.start = 0
        self.__data, self.__targets = self.__preprocess__()
    
    def __preprocess__(self) -> Tuple[List, List]:
        """preprocess self.__data

        Return iteratable X, y that can be indexed by __get_item__
        
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        return NotImplementedError
import pickle
import multiprocessing
import os
import json
from typing import Any, List, Tuple
import redis
from torch.utils.data import Dataset
import concurrent.futures


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
        self.__keys = keys
        self.qos = self.load_metainfo()['qos']
        self.jobinfo = self.load_jobinfo()
        if self.qos['UseCache']: # use datasource from Redis
            r = dotdict(self.jobinfo.jinfo).redisauth
            r = dotdict(r)
            self.client = redis.RedisCluster(host=r.host, port=r.port, username=r.username, password=r.password)
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
        
        self.__index = 0
        self.__targets = None
        self.load_data()

    @property
    def data(self):
        return self.__data
    @property
    def keys(self):
        return self.__keys
    @property
    def targets(self):
        return self.__targets

    @property
    def index(self):
        return self.__index
    @index.setter
    def index(self, value):
        self.__index = value
    
    def get_data(self, index):
        if self.qos['Singular']:
            return self.client.get(self.data[index])
        else:
            return self.data[index]
    
    def get_target(self, index):
        return self.targets[index]
        
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
        snapshot = dotdict(self.jobinfo.policy).snapshot
        def helper(keys):
            chunks = []
            for k in keys:
                if self.__keys is not None:
                    k = "{}/{}".format(dotdict(self.jobinfo.jinfo).jobId, k)
                chunk = pickle.loads(self.client.get(k))
                for v in chunk.values():
                    chunks.extend(v)
            return chunks
        
        return helper(snapshot if self.__keys is None else self.__keys)
    
    def get_all_s3_keys(self):
        chunks = []
        for bk in self.jobinfo['datasource']['keys']:
            paginator = self.client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=bk)
            for page in pages:
                for item in page['Contents']:
                    if self.__keys is not None:
                        for k in self.__keys:
                            if item['Key'].startswith(k):
                                chunks.append({'name': item['Key'], 'key': item['Key'], 'size': item['Size']})
                                break
                    else:
                        chunks.append({'name': item['Key'], 'key': item['Key'], 'size': item['Size']})
        return chunks
    
    def load_chunksubset(self):
        if self.qos['MaxMemory'] == 0: # load all data into memory
            self.index = len(self.chunks)
            return 0, self.index
        else:
            total_size = 0
            start = self.index
            maxmem = self.qos['MaxMemory']*1024*1024
            while self.index < len(self.chunks):
                s = int(self.chunks[self.index]['size'])
                total_size += s
                if total_size > maxmem:
                    break
                elif s > maxmem:
                    raise Exception('File {} size is greater than assigned MaxMemory.'.format(self.chunks[self.index]['name']))
                else:
                    self.index += 1
            return start, self.index
    
    def load_data(self):
        self.__data = {}
        self.__keys = []
        if self.qos['Singular']:
            for chunk in self.chunks:
                self.__keys.append(chunk['name'])
                self.__data[chunk['name']] = chunk['key']
        elif self.index < len(self.chunks):
            def helper(chunk):
                key = chunk['key']
                if self.qos['UseCache']:
                    val = self.client.get(key)
                    if val is None: # cache missing
                        with open('/share/cachemiss', 'w') as f:
                            f.writelines(key)
                        while True:
                            val = self.client.get(key)
                            if val is not None:
                                break
                else:
                    val = self.client.get_object(Bucket=self.bucket_name, Key=key)['Body'].read()
                self.__keys.append(chunk['name'])
                self.__data[chunk['name']] = val
                
            start, end = self.load_chunksubset()
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for chunk in self.chunks[start: end]:
                    futures.append(executor.submit(helper, chunk))
                concurrent.futures.wait(futures)
        self.__data, self.__targets = self.__convert__()
    
    def __convert__(self) -> Tuple[List, List]:
        """convert self.__data

        Return iteratable X, y that can be indexed by __get_item__
        
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError


    def __len__(self) -> int:
        raise NotImplementedError
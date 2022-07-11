from math import inf
import pickle
import os
import json
from typing import Any, List, Tuple
import redis
import concurrent.futures
import multiprocessing
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
        self.__keys = keys
        self.qos = self.load_metainfo()['qos']
        self.jobinfo = self.load_jobinfo()
        self.chunks = []
        self.__targets = None
        if self.qos['UseCache']: # use datasource from Redis
            r = dotdict(self.jobinfo.jinfo).redisauth
            r = dotdict(r)
            self.client = redis.RedisCluster(host=r.host, port=r.port, username=r.username, password=r.password)
            self.load_all_redis_keys()
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
            self.load_all_s3_keys()
        self.load_data(0)

    @property
    def data(self):
        return self.__data
    @property
    def keys(self):
        return self.__keys
    @property
    def targets(self):
        return self.__targets
    
    def load(self, key):
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
        return val

    def mload(self, chunks):
        if self.qos['UseCache']:
            keys = [x['key'] for x in chunks]
            vals = self.client.mget(keys)
            for i in range(len(vals)):
                if vals[i] is None:
                    with open('/share/cachemiss', 'w') as f:
                        f.writelines(keys[i])
                    while True:
                        val = self.client.get(keys[i])
                        if val is not None:
                            vals[i] = val
                            break
                self.__keys.append(chunks[i]['name'])
                self.__data[chunks[i]['name']] = vals[i]
        else:
            vals = []
            def helper(chunk):
                val = self.client.get_object(Bucket=self.bucket_name, Key=chunk['key'])['Body'].read()
                self.__keys.append(chunk['key'])
                self.__data[chunk['name']] = val
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for chunk in chunks:
                    futures.append(executor.submit(helper, chunk))
            concurrent.futures.wait(futures)
        return vals
                    
    def get_data(self, index):
        if self.qos['LazyLoading']:
            return self.load(self.data[index])
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

    def load_all_redis_keys(self):
        snapshot = dotdict(self.jobinfo.policy).snapshot
        maxmem = self.qos['MaxMemory']*1024*1024
        keys = snapshot if self.__keys is None else self.__keys
        keys = []
        if self.__keys is not None:
            for k in self.__keys:
                keys.append("{%s}%s" % (dotdict(self.jobinfo.jinfo).jobId, k))
        else:
            keys = snapshot
        tmp = []
        for s in self.client.mget(keys):
            for obj in pickle.loads(s).values():
                tmp.extend(obj)
        snapshot = tmp

        if maxmem == 0:
            self.chunks.append(snapshot)
        else:
            total_size = 0
            self.chunks.append([])
            for chunk in snapshot:
                s = int(chunk['size'])
                total_size += s
                if total_size > maxmem:
                    self.chunks.append([])
                    total_size = 0
                elif s > maxmem:
                    raise Exception('File {} size is greater than assigned MaxMemory.'.format(chunk['name']))
                else:
                    self.chunks[-1].append(chunk)
        
    def load_all_s3_keys(self):
        paginator = self.client.get_paginator('list_objects_v2')
        if self.__keys is not None:
            pages = []
            for k in self.__keys:
                pages.extend(paginator.paginate(Bucket=self.bucket_name, Prefix=k))
        else:
            pages = paginator.paginate(Bucket=self.bucket_name)
        
        maxmem = self.qos['MaxMemory']*1024*1024
        maxmem = inf if maxmem==0 else maxmem
        total_size = 0
        self.chunks.append([])
        for page in pages:
            for item in page['Contents']:
                chk = {'name': item['Key'], 'key': item['Key'], 'size': item['Size']}
                s = int(chk['size'])
                total_size += s
                if total_size > maxmem:
                    self.chunks.append([])
                    total_size = 0
                elif s > maxmem:
                    raise Exception('File {} size is greater than assigned MaxMemory.'.format(chk['name']))
                else:
                    self.chunks[-1].append(chk)
    
    def load_data(self, index):
        self.__data = {}
        self.__keys = []        
        if self.qos['LazyLoading']:
            for chunk in self.chunks[index]:
                self.__keys.append(chunk['name'])
                self.__data[chunk['name']] = chunk['key']
        else:
            # def helper(chk):
            #     val = self.load(chk['key'])
            #     self.__keys.append(chk['name'])
            #     self.__data[chk['name']] = val
            # with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            #     futures = []
            #     for chunk in self.chunks[index]:
            #         futures.append(executor.submit(helper, chunk))
            # concurrent.futures.wait(futures)
            self.mload(self.chunks[index])
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
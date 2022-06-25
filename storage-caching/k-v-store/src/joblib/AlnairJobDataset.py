import os
import pickle
import json
from typing import Any
import redis
from torch.utils.data import Dataset


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

class AlnairJobDataset(Dataset):
    def __init__(self, keys = None, characters = "utf-8"):
        self.characters = characters
        self.used_keys = []
        
        self.jobinfo = self.load_jobinfo()
        if keys is None:
            self.keys = self.jobinfo.policy.chunkKeys
        else:
            self.keys = keys
        
        r = self.jobinfo.jinfo.redisauth
        self.client = redis.Redis(host=r.host, port=r.port, username=r.username, password=r.password)
        self.load_data()

    def load_jobinfo(self):
        try:
            while not os.path.exists('/share/job.json'):
                pass
            with open("/share/job.json", 'r') as f:
                jobinfo = dotdict(json.load(f))
        except Exception as ex:
            print("Not found job info file")
            raise ex
        return jobinfo
        
    @property
    def data(self):
        return self.__data__
    @data.setter
    def data(self, value):
        self.__data__ = value
    
    def load_data(self):
        self.__data__ = {}
        self.jobinfo = self.load_jobinfo()
        for key in self.keys:
            # 数据以环形方式下发，如果出现重复，说明所有数据已被访问一次
            # 重置 used_keys
            if key in self.used_keys:
                self.used_keys = []
                return None
            else:
                self.used_keys.append(key)
            while True:
                value = self.client.get(key)
                if value is None: # cache missing
                    with open('/share/cachemiss', 'w') as f:
                        f.writelines([key])
                else:
                    break
            value = pickle.loads(value, encoding=self.characters)
            self.__data__[key] = value
        self.__preprocess__()
    
    def __preprocess__(self) -> Any:
        raise NotImplementedError
    
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        return NotImplementedError
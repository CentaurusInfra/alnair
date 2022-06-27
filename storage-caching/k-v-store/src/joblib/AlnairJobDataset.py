import os
import pickle
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
    def __init__(self, keys: List[str] = None, characters = "utf-8"):
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
            characters (str, optional): character of data saved in S3 bucket. Defaults to "utf-8".
        """
        
        self.jobname = os.environ.get('JOBNAME')
        self.keys = keys
        self.characters = characters
        self.used_keys = []
        
        self.jobinfo = self.load_jobinfo()
        self.chunks = dotdict(self.jobinfo.policy).chunkKeys
        if keys is not None:
            self.chunks = [chunk for chunk in self.chunks if chunk['name'] in keys]
        
        r = dotdict(self.jobinfo.jinfo).redisauth
        r = dotdict(r)
        self.client = redis.Redis(host=r.host, port=r.port, username=r.username, password=r.password)
        self.load_data()

    def load_jobinfo(self):
        try:
            while not os.path.exists('/share/{}.json'.format(self.jobname)):
                pass
            with open("/share/{}.json".format(self.jobname), 'rb') as f:
                jobinfo = dotdict(json.load(f))
        except Exception as ex:
            print("Not found job info file")
            raise ex
        return jobinfo
        
    @property
    def data(self):
        return self.__data
    @property
    def targets(self):
        return self.__targets
    
    def load_data(self):
        self.__data = {}
        self.jobinfo = self.load_jobinfo()
        for chunk in self.chunks:
            key = chunk['key']
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
            self.__data[chunk['name']] = value
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
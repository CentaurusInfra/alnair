import os
import pickle
import json
from typing import Any, Callable, Optional, Callable
import redis
import numpy as np
from utils.utils import *
from torch.utils.data import Dataset


class DLTDataset(Dataset):
    """DLTDeploy Dataset

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
        transform (callable, optional): A function/transform that takes in the dataset
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, 
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        self.used_keys = []
        self.jobinfo = self.load_jobinfo()
        r = self.jobinfo.jinfo.redisauth
        self.client = redis.Redis(host=r.host, port=r.port, username=r.username, password=r.password)
        self.data, self.target = self.load_data(train)
        if transform is not None:
            transform(self.data)
        if target_transform is not None:
            target_transform(self.target)

    def load_jobinfo(self):
        try:
            jobfile = [".json" in f for f in os.listdir("/data")][0]
            with open(os.path.join("/data", jobfile), 'r') as f:
                jobinfo = dotdict(json.load(f))
        except Exception as ex:
            print("Not found job info file")
            raise ex
        return jobinfo
    
    def load_data(self, train=True):
        self.jobinfo = self.load_jobinfo()
        chunkKeys = self.jobinfo.policy.chunkKeys
        filter_keys = lambda X, v: [x for x in X if v in x]
        
        def readchunks(keys):
            data = []
            for key in keys:
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
                        with open('/data/cachemiss', 'w') as f:
                            f.writelines([key])
                    else:
                        break
                value = pickle.loads(value)
                data = np.stack(data, value)
            return data

        X = readchunks(filter_keys(chunkKeys, 'X_train' if train else 'X_test'))
        y = readchunks(filter_keys(chunkKeys, 'y_train' if train else 'y_test'))
        return X, y
            

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        return NotImplementedError
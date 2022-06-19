import os
import pickle
import json
from typing import Any, Callable, Optional, Tuple
import redis
import numpy as np
from utils.utils import *
from torch.utils.data import Dataset


class DLTDataset(Dataset):
    """DLTDeploy Dataset

    Args:
        client (Redis Client)
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
    }

    def __init__(self) -> None:
        try:
            jobfile = [".json" in f for f in os.listdir("/data")][0]
            with open(os.path.join("/data", jobfile), 'r') as f:
                jobinfo = dotdict(json.load(f))
        except Exception as ex:
            print("Not found job info file")
            raise ex

        r = jobinfo.redisauth
        self.client = redis.Redis(host=r.host, port=r.port, username=r.username, password=r.password)
        self.data: Any = []

    """
    1. 是否把所有key的数据都load进来
    2. 如果是，对于该dataset而言，redis存储的上界为client memory的大小
    3. 如果不是，client需要逐个key读取数据，如果所有的key都已经被读了一遍，client需要做什么，manager需要做什么
    """
    def load(self, key, encoding='utf-8'):
        """load data from Redis under specified keys

        Args:
            key (_type_): _description_
            encoding (str, optional): characters of loaded data. Defaults to 'utf-8'.
        """
        value = self.client.get(key)
        entry = pickle.loads(value, encoding=encoding)
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)
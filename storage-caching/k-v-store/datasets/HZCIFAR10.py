import os
from typing import Any, Callable, List, Optional, Tuple
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets.utils import download_url, check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset


class HZCIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        hz_map (Hazelcast Map): Hazelcast Map object where data are stored
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
    }

    def __init__(
        self,
        hz_obj,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        asyc=True
    ) -> None:
        self.hz_obj = hz_obj
        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform
        self.asyc = asyc
        
        self.data: Any = []
        self.targets = []     
        self._load_meta_map()

    def load_data(self, key, reset=True):
        if reset:
            self.data = []
            self.targets = []
        if self.asyc:
            value = self.hz_obj.get(key).result().encode('latin1')
        else:
            value = self.hz_obj.get(key).encode('latin1')
        entry = pickle.loads(value, encoding='latin1')
        self.data.append(entry["data"])
        if "labels" in entry:
            self.targets.extend(entry["labels"])
        else:
            self.targets.extend(entry["fine_labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
    def _load_meta_map(self) -> None:
        if self.asyc:
            value = self.hz_obj.get(self.meta['filename']).result().encode('latin1')
        else:
            value = self.hz_obj.get(self.meta['filename']).encode('latin1')
        data = pickle.loads(value, encoding='latin1')
        self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """    
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)
from typing import Callable, Optional
import numpy as np
import pickle
from PIL import Image
from lib.AlnairJobDataset import AlnairJobDataset as AJDataset


class CIFAR10Meta(AJDataset):
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
    }

    def __init__(self):
        super().__init__(keys=["batches.meta"]) 
        self.load_meta_map()
          
    def load_meta_map(self) -> None:
        data = self.data[self.meta['filename']]
        self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def preprocess(self):
        return [], []
        
    def __getitem__(self, index: int):
        return

    def __len__(self):
        return
    

class CIFAR10Datset(AJDataset):
    def __init__(self, keys, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(keys)
        self.transform = transform
        self.target_transform = target_transform

    def __preprocess__(self):
        processed_data = []
        targets = []
        for key in self.keys:
            entry = pickle.loads(self.data[key], encoding='latin1')
            processed_data.append(entry["data"])
            if "labels" in entry:
                targets.extend(entry["labels"])
            else:
                targets.extend(entry["fine_labels"])
        processed_data = np.vstack(processed_data).reshape(-1, 3, 32, 32)
        processed_data = processed_data.transpose((0, 2, 3, 1))  # convert to HWC
        return processed_data, targets
        
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)
import os
import sys
import time
import torch
import pickle
import numpy as np
import nvidia.dali.ops as ops
from base import DALIDataloader
from torchvision import datasets
from sklearn.utils import shuffle
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms

import nvtx
from timeit import default_timer as timer

test_batch_size = 64

def speedtest(tgt, data_loader, batch, n_threads, device, iter_num=10):
    t_start = timer()
    cnt = 0
    for i, data in enumerate(data_loader):
        cnt = cnt + 1
        if (cnt > iter_num):
            break
        if device=='cpu':
            images = data[0].cpu()
            labels = data[1].cpu()
        else:
            images = data[0].cuda(non_blocking=True)
            labels = data[1].cuda(non_blocking=True)
    t = timer() - t_start
    print("{} Speed: {} imgs/s".format(tgt, (cnt * batch)/t))
    

#
# imagenet
#
IMAGENET_MEAN = [0.49139968, 0.48215827, 0.44653124]
IMAGENET_STD = [0.24703233, 0.24348505, 0.26158768]
IMAGENET_IMAGES_NUM_TRAIN = 1281167
IMAGENET_IMAGES_NUM_TEST = 50000
IMG_DIR = '/root/data/imagenet'
TRAIN_BS = 256
TEST_BS = 20
NUM_WORKERS = 4
VAL_SIZE = 256
CROP_SIZE = 224



@nvtx.annotate("file reader data", color="red")
def read_data(file_root, shard_id=0, num_shards=1, random_shuffle=True):
    ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)    
    return np.random.random((size, size))

# class HybridCocoPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
#         super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
#         # self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
#         self.input = read_data(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
#         self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
#         self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
#         self.cmnp = ops.CropMirrorNormalize(device="gpu",
#                                             output_dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             image_type=types.RGB,
#                                             mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#                                             std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
#         self.coin = ops.CoinFlip(probability=0.5)

#     def define_graph(self):
#         rng = self.coin()
#         self.jpegs, self.labels = self.input(name="Reader")
#         images = self.decode(self.jpegs)
#         images = self.res(images)
#         output = self.cmnp(images, mirror=rng)
#         return [output, self.labels]

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


    
pip_train = HybridTrainPipe(batch_size=TRAIN_BS, num_threads=NUM_WORKERS, device_id=0, data_dir=IMG_DIR+'/train', crop=CROP_SIZE, world_size=1, local_rank=0)
dali_loader = DALIDataloader(pipeline=pip_train, size=IMAGENET_IMAGES_NUM_TRAIN, batch_size=TRAIN_BS, onehot_label=True)

print("[DALI] train dataloader length: %d"%len(dali_loader))
print('[DALI] start iterate train dataloader')
iter_n = 2000
speedtest('dali dataloader', dali_loader, test_batch_size, 4, 'gpu', iter_num=iter_n)

# start = time.time()

# for i, data in enumerate(train_loader):
#     images = data[0].cuda(non_blocking=True)
#     labels = data[1].cuda(non_blocking=True)
# end = time.time()
# train_time = end-start

# print('[DALI] end train dataloader iteration')
# print('[DALI] iteration time: %fs [train]' % (train_time))

# A quick intro to implement the Distributed Data Parallel (DDP) training in Pytorch.
# To simply this example, we directly load the ResNet50 using ```torch.hub.load()```,
# and train it from the scratch using the CIFAR10 dataset.

# Run this python script in terminal like "python3 DDP_training.py -n 1 -g 8 -nr 0"

import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from base import DALIDataloader
from cifar10 import HybridTrainPipe_CIFAR
import time


IMG_DIR = '/data/cifar10/cifar100'
TRAIN_BS = 256
TEST_BS = 200
NUM_WORKERS = 4
CROP_SIZE = 32
CIFAR_IMAGES_NUM_TRAIN = 50000


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-cf', '--conf', default=0, type=int,
                        help='cf: 0-std, 1-dali, 2-daliNmve')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8889'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu, args):
    alnair_pf = os.getenv('PFLOG')
    if (alnair_pf  is None):
        alnair_pf= "/root/test"     
    batch_time = AverageMeter()

    # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
    model = torchvision.models.densenet121(pretrained=True, progress=True)    

    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    if(args.conf==0): 
        #pytorch dataloader
        print("torch dataloader ...")
        trainset = torchvision.datasets.CIFAR10(root=IMG_DIR,
                                                train=True,
                                                download=True,
                                                transform=transform)
        
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=NUM_WORKERS,
                                                pin_memory=True,
                                                sampler=trainsampler)
    else:
        #DALI dataloader
        print("DALI dataloader ...")
        pip_train = HybridTrainPipe_CIFAR(batch_size=batch_size,
                                        num_threads=NUM_WORKERS,
                                        device_id=0, 
                                        data_dir=IMG_DIR, 
                                        crop=CROP_SIZE, 
                                        world_size=1, 
                                        local_rank=0, 
                                        cutout=0)
        trainloader = DALIDataloader(pipeline=pip_train,
                                    size=CIFAR_IMAGES_NUM_TRAIN, 
                                    batch_size=batch_size, 
                                    onehot_label=True)     

    total_step = len(trainloader)
    train_start = datetime.now()
    alnair_log = os.path.join(str(alnair_pf), "py_step.log")
    print_freq = 10
    end = time.time()

    with open(alnair_log, "w") as f:
        for epoch in range(args.epochs):
            for i, (images, labels) in enumerate(trainloader):
                start = datetime.now()
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_time.update((time.time() - end)/print_freq)
                end = time.time()

                if (i + 1) % print_freq == 0 and gpu == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                            loss.item()))
                    print("Training complete in: " + str(datetime.now() - start))
                f.write("step:" + str(i+1) + ", start: " + str(start) + ", duation:" + str((datetime.now()-start).microseconds) + "\n")

    print("Training done, total epoch {}, total time {}".format(args.epochs, datetime.now()-train_start))

    if gpu == 0 or args.gpus == 1:
        alnair_bm = os.path.join(str(alnair_pf), "benchmark.log")
        print("benchmark log: {}".format(alnair_bm))
        f_bm = open(alnair_bm, "w")
        if args.conf==0: 
            loadername = "torch" 
        else: 
            loadername = "DALI"
        f_bm.write("loader:" + loadername  + ", GPU: " + str(args.gpus) + ", batch time: " + str(batch_time.val) + "(" + str(batch_time.avg) + ")\n")
        f_bm.write("Training done, total epoch {}, total time {}\n".format(args.epochs, datetime.now()-train_start))

        f_bm.close()

if __name__ == '__main__':
    main()

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
import horovod.torch as hvd
from filelock import FileLock
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch', default=16, type=int,
                        help='batch size')
    args = parser.parse_args()
    args.data_dir = '~/data'
    args.cuda = torch.cuda.is_available()

    train(args)


def train(args):
    hvd.init()
    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
          mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    with FileLock(os.path.expanduser("~/.horovod_lock")):
        trainset = \
            torchvision.datasets.CIFAR10(root=args.data_dir,
                                         train=True,
                                         download=True,
                                         transform=transform)

    trainsampler = hvd.elastic.ElasticSampler(trainset)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch,
                                              sampler=trainsampler,
                                              **kwargs)

    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
    model.cuda()

    lr_scaler = hvd.local_size()

    optimizer = torch.optim.SGD(model.parameters(), lr=.0001 * lr_scaler)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    total_step = len(trainloader)
    train_start = datetime.now()
    epoch_time = datetime.now()

    criterion = nn.CrossEntropyLoss().cuda(hvd.local_rank())

    @hvd.elastic.run
    def run_epochs(state):
        batch_offset = state.batch
        for state.epoch in range(state.epoch, args.epochs):
            start = datetime.now()
            iter_trainloader = iter(trainloader)
            for state.batch in range(state.batch, total_step):
                images, labels = next(iter_trainloader)
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (state.batch + 1) % 10 == 0 and hvd.local_rank() == 0:
                    start = datetime.now()

                    print(f'Epoch [{state.epoch + 1}/{args.epochs}], Step [{state.batch + 1}/{total_step}], Loss {round(loss.item(), 4)}')
                    print(f'Training complete in: {datetime.now() - start}')
                    print(f'Number of GPUs: {torch.cuda.device_count()}')
                    state.commit()

    state = hvd.elastic.TorchState(model, optimizer, batch=0, epoch=0)
    run_epochs(state)

    print("Training done, total epoch {}, total time {}".format(args.epochs, datetime.now() - train_start))


if __name__ == '__main__':
    main()

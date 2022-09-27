#need to change the masterIP
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default="", type=str,
                        help='gpu number list of workers (starting with the master worker). For example, if you have two machines one with 2 gpus and the other one with 8 gpus, the list should be 2, 8')
    parser.add_argument('-i', '--id', default=0, type=int,
                        help='id for each worker and this id should match with the gpu number list. For example, if the GPU list is 2, 8, the id of the machine with 2 gpus should be 0')                        
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    gpu_list = [int(gpu) for gpu in args.gpus.split(',')]
    args.world_size = sum(gpu_list)
    # the current code only works if we assign the node with rank 0 as the master
    os.environ['MASTER_ADDR'] = '10.145.83.35'
    os.environ['MASTER_PORT'] = '9004'
    mp.spawn(train, nprocs=gpu_list[args.id], args=(args,))


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu, args):
    gpu_list = [int(gpu) for gpu in args.gpus.split(',')]
    rank = sum(gpu_list[:args.id]) + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    print("my process rank:{} on gpu {}\n".format(rank, gpu))
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print('Process {},Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(rank,epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()

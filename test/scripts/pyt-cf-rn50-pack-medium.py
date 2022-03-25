# A quick intro to implement the Distributed Data Parallel (DDP) training in Pytorch.
# To simply this example, we directly load the ResNet50 using ```torch.hub.load()```,
# and train it from the scratch using the CIFAR10 dataset.

# Run this python script in terminal like "python3 DDP_training.py -n 1 -g 8 -nr 0"

import os
from datetime import datetime
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pickle as pkl

TIMES = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-e', '--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch', default=16, type=int,
                        help='batch size')
    parser.add_argument('-d', '--directory', default=16, type=int,
                        help='parent directory of pickle dump')
    parser.add_argument('-f', '--frac', default=.35, type=float,
                        help='per process memory fraction')
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.frac)
    train(args, gpu=args.nr)


def train(args, gpu):
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = args.batch
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Data loading code
    trainset = torchvision.datasets.CIFAR10(root='~/data',
                                            train=True,
                                            download=True,
                                            transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)

    total_step = min(len(trainloader), 240)
    train_start = datetime.now()
    for epoch in range(args.epochs):
        start = datetime.now()
        for i, (images, labels) in enumerate(trainloader):
            if i > total_step: 
                break
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
                TIMES.append((datetime.now() - start).microseconds)
                print("Training complete in: " + str(TIMES[-1]))
                start = datetime.now()

    print("Training done, total epoch {}, total time {}".format(args.epochs, datetime.now()-train_start))
    print('===========================')
    print(sum(TIMES) / len(TIMES))
    print('===========================')

if __name__ == '__main__':
    main()

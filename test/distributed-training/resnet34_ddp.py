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
from torchvision import models
from torch.nn.functional import nll_loss, cross_entropy


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
    os.environ['MASTER_ADDR'] = '10.175.20.129'
    os.environ['MASTER_PORT'] = '9876'
    mp.spawn(train, nprocs=gpu_list[args.id], args=(args,))


torch.manual_seed(42) 

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

validation_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

valset= torch.utils.data.DataLoader(validation_data, batch_size=128)

def validate(model, device, val_loader: torch.utils.data.DataLoader) -> (float, float):
    """Loop used to validate the network"""

    criterion = nn.CrossEntropyLoss().cuda(device)
    model.eval()
    model.cuda(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_loss += cross_entropy(output, target).item()  # sum up batch loss
            correct += predicted.eq(target).sum().item()

    test_loss /= len(val_loader)

    accuracy = 100. * correct / len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return accuracy, test_loss

def cleanup():
    dist.destroy_process_group()

def train(gpu, args):
    gpu_list = [int(gpu) for gpu in args.gpus.split(',')]
    rank = sum(gpu_list[:args.id]) + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    print("my process rank:{} on gpu {}\n".format(rank, gpu))
    torch.manual_seed(0)
    model = models.resnet.resnet34()

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 256
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    #optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                               train=True,
                                               transform=transform,
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
            if (i + 1) % 30 == 0:
                print('Process {},Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(rank,epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
    print("Validiation Accuracy and loss: ", validate(model, gpu, valset))
    cleanup()

        


if __name__ == '__main__':
    main()

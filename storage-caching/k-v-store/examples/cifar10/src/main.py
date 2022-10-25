import torch
import torchvision.transforms as transforms
import argparse
from AlnairJob import AlnairJobDataLoader
from CIFAR10Dataset import *
from model import *


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
test_list = ["test_batch"]
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
jobname = "job1"

PATH = './cifar_net.pth'


def train(epochs, batch_size, lr):
    trainset = CIFAR10Datset(train_list, transform)
    trainloader = AlnairJobDataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), PATH)


def test(batch_size):
    testset = CIFAR10Datset(test_list, transform)
    testloader = AlnairJobDataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net()
    net.load_state_dict(torch.load(PATH))
    
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr)
    test(args.batch_size)
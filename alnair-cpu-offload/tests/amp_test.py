import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from apex import amp

import copy


use_amp = False
clean_opt = False

device='cuda'

model = models.resnet18()
model.to(device)
state_dict = copy.deepcopy(model.state_dict())
optimizers = [optim.Adam(model.parameters(), lr=1e-3) for _ in range(3)]
if use_amp:
    model, optimizers = amp.initialize(model, optimizers, opt_level='O1')

dataset = datasets.FakeData(transform=transforms.ToTensor())
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    shuffle=True
)
criterion = nn.CrossEntropyLoss()

print('Memory allocated {:.3f}MB'.format(
    torch.cuda.memory_allocated() / 1014**2))

for opt_idx, optimizer in enumerate(optimizers):
    # reset model
    model.load_state_dict(state_dict)
    
    # Train
    for epoch in range(5):
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            if use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            
        print('OptIdx {}, epoch {}, loss {}, mem allocated {:.3f}MB'.format(
            opt_idx, epoch, loss.item(), torch.cuda.memory_allocated()/1024**2))

    if clean_opt:
        optimizers[opt_idx] = None
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from apex import amp

import copy
import time


use_amp = True
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
    num_workers=0,
    pin_memory=False,
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
        start = time.perf_counter()

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
        elapsed = time.perf_counter() - start
            
        print('OptIdx {}, epoch {}, loss {}, mem allocated {:.3f}MB, time: {:.3f} seconds.'.format(
            opt_idx, epoch, loss.item(), torch.cuda.memory_allocated()/1024**2, elapsed))

    if clean_opt:
        optimizers[opt_idx] = None
'''
Author: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
Date: 2023-11-12 20:49:33
LastEditors: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
LastEditTime: 2023-11-12 21:10:42
FilePath: \AutonomousDriving\PyTorch\src\conv2d.py
Description: use conv2d
'''
import torch
import torchvision
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10("./PyTorch/data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, 64)


writer = SummaryWriter("./PyTorch/logs/Conv2d_log")

step = 0
for data in dataloader:
    img, target = data
    writer.add_images("input", img, step)

    layer = torch.nn.Conv2d(3, 6, 3, 1, 1)
    conv2d = layer(img).reshape((-1, 3, 32, 32))
    
    writer.add_images("conv2d", conv2d, step)

    step += 1
writer.close()
    
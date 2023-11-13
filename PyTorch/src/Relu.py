'''
Author: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
Date: 2023-11-12 21:48:55
LastEditors: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
LastEditTime: 2023-11-12 21:58:10
FilePath: \AutonomousDriving\PyTorch\src\Relu.py
Description: use ReLU
'''
import torch
import torchvision
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import tqdm


transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10("./PyTorch/data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, 64)


writer = SummaryWriter("./PyTorch/logs/Relu_log")

step = 0
for data in tqdm.tqdm_gui(dataloader):
    img, target = data
    writer.add_images("input", img, step)

    layer = torch.nn.ReLU()
    relu = layer(img)
    writer.add_images("rule", relu, step)

    step += 1
writer.close()
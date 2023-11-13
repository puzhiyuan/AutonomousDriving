'''
Author: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
Date: 2023-11-12 21:30:37
LastEditors: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
LastEditTime: 2023-11-12 21:48:42
FilePath: \AutonomousDriving\PyTorch\src\Maxpool.py
Description: use maxpoll
'''
import torch
import torchvision
from torch.utils.data import DataLoader
import tqdm
from torch.utils.tensorboard import SummaryWriter

transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10("./PyTorch/data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64)

writer = SummaryWriter("./PyTorch/logs/Maxpool_log")
step = 0
for data in tqdm.tqdm(dataloader, "porcessing"):
    img, target = data
    writer.add_images("input", img, step)

    layer = torch.nn.MaxPool2d(3, 1)
    maxpool = layer(img)
    writer.add_images("maxpool", maxpool, step)

    step += 1
writer.close()
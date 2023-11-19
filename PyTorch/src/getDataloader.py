'''
Author: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
Date: 2023-11-19 19:14:13
LastEditors: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
LastEditTime: 2023-11-19 19:19:04
FilePath: \AutonomousDriving\PyTorch\src\getFashionMNIST.py
Description: get dataloader (train_dataloader, test_dataloader)
'''
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def getFashionMNIST(root, transform, batch_size):
    train_dataset = datasets.FashionMNIST(root, train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root, train=False, transform=transform, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (train_dataloader, test_dataloader)
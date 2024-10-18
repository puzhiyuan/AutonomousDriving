'''
Author: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
Date: 2023-11-19 16:59:44
LastEditors: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
LastEditTime: 2023-11-19 21:59:04
FilePath: \AutonomousDriving\PyTorch\src\VGG.py
Description: VGG16 and VGG18
'''

from turtle import Turtle
from getDataloader import getFashionMNIST
from getDevice import get_device
from model_test import model_test
from model_train import model_train
from sympy import ode_order
import torch
import torch.nn as nn
from torchvision import datasets, transforms


class VGG_16(nn.Module):
    def __init__(self, in_channels=1, num_classes=1000, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(7*7*512, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, X):
        return self.layers(X)



class VGG_19(nn.Module):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                                          512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'])
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    def forward(self, X):
        return self.classifier(nn.Flatten(self.features(X)))
    

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for i in cfg:
            if i == "M":
                layers.append(nn.MaxPool2d(2,2))
            else:
                layers.append(nn.Conv2d(in_channels, i, 3, padding=1))
                layers.append(nn.ReLU())
                in_channels = i
        return nn.Sequential(*layers)


if __name__ == "__main__":
    batch_size, lr, epochs = 64, 0.01, 30
    device = get_device()
    print(f"device: {device}")
    root = "PyTorch/data"
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    train_dataloader, test_dataloader = getFashionMNIST(root, transform, batch_size)

    model = VGG_16().to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for i in range(epochs):
        train_loss = model_train(model, train_dataloader, loss, optimizer, device)
        test_accuracy = model_test(model, test_dataloader, device)
        print(f"eopch: {i+1}/{epochs}  train_loss: {train_loss}  test_accuracy: {test_accuracy}")

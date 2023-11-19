'''
Author: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
Date: 2023-11-15 09:45:20
LastEditors: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
LastEditTime: 2023-11-19 16:48:42
FilePath: \AutonomousDriving\PyTorch\src\AlexNet.py
Description: 实现 AlexNet
'''
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_train import *
from model_test import *
import tqdm



class AlexNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(1,96, 11, 4, 1), 
            nn.MaxPool2d(3, 2), 
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.Conv2d(384,384, 3, 1, 1),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.MaxPool2d(3,2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(0.5), 
            nn.Linear(4096, 4096),nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )


    def forward(self, X):
        return self.model(X)

# model = AlexNet()
# ip = torch.ones((64,1,224,224))
# print(model(ip).shape)


if __name__ == "__main__":
        

    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    train_dataset = datasets.FashionMNIST("/PyTorch/data", train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST("/PyTorch/data", train=False, transform=transform, download=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    print(len(train_dataloader), len(test_dataloader))

    for data in train_dataloader:
        inp, _ = data
        print(inp.shape)
        break

    epochs = 20
    batch_size = 256
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")

    
    model = AlexNet().to(device=device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)



    for i in tqdm.tqdm(range(epochs)):
        train_loss = model_train(model, train_dataloader, loss, optimizer, device)
        test_accuracy = model_test(model, test_dataloader, device)
        print(f"epoch {i+1} / {epochs}  train_loss:{train_loss} test_accuracy:{test_accuracy}")

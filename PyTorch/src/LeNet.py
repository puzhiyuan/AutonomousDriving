'''
Author: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
Date: 2023-11-13 16:14:47
LastEditors: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
LastEditTime: 2023-11-15 21:34:26
FilePath: \AutonomousDriving\PyTorch\src\LeNet.py
Description: LeNet
'''
import time
import torch 
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm



class LeNet_5(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10), nn.Softmax(dim=1)
        )

    def forward(self, X):
        return self.layers(X)
    
# 测试输出结果维度
# input = torch.ones((64, 3, 32, 32))
# model = LeNet_5()
# output = model(input)

def train(model, train_dataloader, loss, optimizer, device):
    model.train()
    running_loss = 0.0
    for data in train_dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 注意这里要清零梯度*
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_score = loss(outputs, labels)
        loss_score.backward()
        optimizer.step()
        running_loss += loss_score.item()
    return running_loss/len(train_dataloader)
  
def test(model, test_dataloader, device):
    model.eval()
    test_loss = 0.0
    
    # 分类问题测试准确率
    correct = 0
    total = 0

    # 禁用梯度计算
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 计算预测准确度
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    return correct / total


if __name__ == "__main__":
    batch_size = 256
    lr = 0.1
    epochs = 50

    train_dataset = torchvision.datasets.MNIST("./PyTorch/data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST("./PyTorch/data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    for data in train_dataloader:
        img, _ = data
        print(img.size())
        break

    model = LeNet_5()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device:{device}")

    start_time = time.time()
    for i in range(epochs):
        train_loss = train(model=model, train_dataloader=train_dataloader, loss=loss, optimizer=optimizer, device=device)
        test_accuracy = test(model=model, test_dataloader=test_dataloader, device=device)
        print(f'epoch {i+1} / {epochs}  train_loss:{train_loss}, test_accuracy:{test_accuracy}')
    all_time = time.time() - start_time
    torch.save(model.state_dict(), "./PyTorch/weight/LeNet_5.pt")
    print(f"time:{all_time}")
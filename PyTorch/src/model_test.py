'''
Author: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
Date: 2023-11-14 20:10:33
LastEditors: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
LastEditTime: 2023-11-15 21:45:39
FilePath: \AutonomousDriving\PyTorch\src\model_test.py
Description: test code
'''


import torch

def model_test(model, dataloader, device):
    model.eval()
    correct = 0
    total = len(dataloader)
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predict = torch.max(outputs.data, 1)
        correct += (predict == labels).sum().item()
    return correct / total




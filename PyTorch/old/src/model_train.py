'''
Author: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
Date: 2023-11-15 14:15:12
LastEditors: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
LastEditTime: 2023-11-15 21:45:31
FilePath: \AutonomousDriving\PyTorch\src\model_train.py
Description: tarin code
'''


def model_train(model, dataloader, loss, optimizer, device):
    model.train()
    all_loss = 0.0
    for data in dataloader:
        optimizer.zero_grad()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss_score = loss(outputs, labels)
        loss_score.backward()
        optimizer.step()
        all_loss += loss_score
    return all_loss / len(dataloader)

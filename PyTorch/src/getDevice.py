'''
Author: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
Date: 2023-11-19 19:01:41
LastEditors: ‘puzhiyuan’ ‘puzhiyuan185489643@gmail.com’
LastEditTime: 2023-11-19 19:05:22
FilePath: \AutonomousDriving\PyTorch\src\get_device.py
Description: try to get gpu
'''
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cup" )
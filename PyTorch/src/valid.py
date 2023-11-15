import torch
from torch import nn
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from LeNet import *

model = LeNet_5()
model.load_state_dict(torch.load("PyTorch/weight/LeNet_5.pt"))

image = Image.open("PyTorch/data/7.png")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

image = transform(image).reshape((1,1,28,28))

print(image.size())

output = model(image)

print(output)
print(torch.max(output, 1))
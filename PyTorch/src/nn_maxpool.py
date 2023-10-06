import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.pool = MaxPool2d(3, ceil_mode=True)

    def forward(self, x):
        x = self.pool(x)
        return x


writer = SummaryWriter("../logs_MaxPool")
maxpool = MaxPool()
step = 0
print("start...")
for data in dataloader:
    image, targets = data
    output = maxpool(image)
    writer.add_images("input", image, step)
    writer.add_images("output", output, step)
    step += 1
print("over...")
writer.close()

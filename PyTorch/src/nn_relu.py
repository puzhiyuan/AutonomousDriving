import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Relu(nn.Module):
    def __init__(self):
        super(Relu, self).__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.Sigmoid()

    def forward(self, input):
        output = self.relu2(input)
        return output


relu = Relu()

writer = SummaryWriter("../logs_Relu")
step = 0
for data in dataloader:
    image, targets = data
    output = relu(image)
    writer.add_images("input", image, step)
    writer.add_images("output", output, step)
    step += 1
writer.close()

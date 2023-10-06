import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class MyConv2d(nn.Module):
    def __init__(self):
        super(MyConv2d, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


myconv2d = MyConv2d()

write = SummaryWriter("../logs_Conv2d")

step = 0
print("start...")
for data in dataloader:
    image, targets = data
    output = myconv2d(image)
    output = torch.reshape(output, (-1, 3, 30, 30))
    write.add_images("input", image, step)
    write.add_images("output", output, step)
    step += 1
print("over...")
write.close()


import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 相对路径为相对于当前工程的
dataset = torchvision.datasets.CIFAR10("./PyTorch/data", train=True, transform=torchvision.transforms.ToTensor(), download=True, )
dataloader = DataLoader(dataset, 64)
# 打印图片数量
print(f'size of dataset:{len(dataset)}')
# 实例化writer对象
writer = SummaryWriter("./PyTorch/logs/tensorboard")

step = 0
for data in dataloader:
    img, target = data
    # 这里有坑,注意add_images和add_image
    writer.add_images("input", img, step)
    step += 1
writer.close()


# 在命令行使用 tensorboard --logdir=/PyTorch/logs 
# 在浏览器访问对应的链接查看



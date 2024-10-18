import torch
from torch import conv2d, nn


def function_name_decorator(func):
    def wrapper(*arg, **args):
        print(f"\nExecuting function:{func.__name__}")
        return func(*arg, **args)
    return wrapper

@function_name_decorator
def test_conv1d():
    data = torch.rand(size=(2,10,512))
    conv1d = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, stride=2, padding=2)
    out = conv1d(data)
    print(data.shape)
    print(out.shape)

@function_name_decorator
def test_conv2d():
    data = torch.rand(size=(2,3,512,512))
    conv2d = nn.Conv2d(in_channels=3,out_channels=20,kernel_size=3,stride=2,padding=1)
    out = conv2d(data)
    print(data.shape)
    print(out.shape)


if __name__ == '__main__':
    test_conv1d()
    test_conv2d()
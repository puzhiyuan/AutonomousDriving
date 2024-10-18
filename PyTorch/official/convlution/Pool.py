import torch
from torch import nn 

# Define a decorator to print the function name
def function_name_decorator(func):
    def warpper(*arg, **agrs):
        print("\nExecuting function:{}".format(func.__name__))
        return func(*arg, **agrs)
    return warpper

@function_name_decorator
def test_MaxPool1d():
    data = torch.rand(size=(8,512,512))
    MaxPool1d = nn.MaxPool1d(4)
    out = MaxPool1d(data)
    print("origin:", data.shape)
    print("MaxPool1d:", out.shape)

@function_name_decorator
def test_MaxPool2d():
    data = torch.rand(size=(8,16,512,512))
    MaxPool2d = nn.MaxPool2d(4)
    out = MaxPool2d(data)
    print("origin:", data.shape)
    print("MaxPool2d:", out.shape)


if __name__ == '__main__':
    test_MaxPool1d()
    test_MaxPool2d()
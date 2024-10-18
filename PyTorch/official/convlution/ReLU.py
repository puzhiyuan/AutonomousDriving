import torch
from torch import nn


def function_name_decorator(func):
    def wrapper(*arg, **args):
        print(f"\nExecuting function:{func.__name__}")
        return func(*arg, **args)
    return wrapper


@function_name_decorator
def test_ReLU():
    data = torch.randn(2,2)
    ReLU = nn.ReLU()
    out = ReLU(data)
    print(data)
    print(out)


@function_name_decorator
def test_Sigmod():
    data = torch.randn(2,2)
    Sigmod = nn.Sigmoid()
    out = Sigmod(data)
    print(data)
    print(out)


@function_name_decorator
def test_SoftMax():
    data = torch.randn(2,4)
    SoftMax = nn.Softmax(dim=1)
    out = SoftMax(data)
    print(data)
    print(data.sum(dim=1))
    print(out)
    print(out.sum(dim=1))


if __name__ == '__main__':
    test_ReLU()
    test_Sigmod()
    test_SoftMax()
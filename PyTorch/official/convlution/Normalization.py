import torch
from torch import nn


def function_name_decorator(func):
    def wrapper(*arg, **args):
        print(f"\nExecuting function:{func.__name__}")
        return func(*arg, **args)
    return wrapper


@function_name_decorator
def test_BatchNorm1d():
    data = torch.randn(2, 3, 4)
    BatchNorm1d = nn.BatchNorm1d(3)  # 3 is the number of features (BCL中的C)
    out = BatchNorm1d(data)
    print(data.shape)
    print(out.shape)


@function_name_decorator
def test_BatchNorm2d():
    data = torch.randn(2, 16, 4, 4)
    BatchNorm2d = nn.BatchNorm2d(16)  # 3 is the number of features (BCWH中的C)
    out = BatchNorm2d(data)
    print(data.shape)
    print(out.shape)


@function_name_decorator
def test_LayerNorm():
    data = torch.randn(8, 16, 64, 64)
    LayerNorm = nn.LayerNorm(normalized_shape=(16, 64, 64))
    out = LayerNorm(data)
    print(data.shape)
    print(out.shape)


if __name__ == "__main__":
    test_BatchNorm1d()
    test_BatchNorm2d()
    test_LayerNorm()

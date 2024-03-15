__all__ = [
    'square',
    'cube',
    'pow4',
    'pow5',
    'pow6',
    'pow7',
    'square_',
    'cube_',
    'pow4_',
    'pow5_',
    'pow6_',
    'pow7_',
    'floor_div',
    'floor_div_int',
    'trunc_div',
    'trunc_div_int',
]
import torch


@torch.jit.script
def square(x):
    return x * x


@torch.jit.script
def square_(x):
    return x.mul_(x)


@torch.jit.script
def cube(x):
    return x * x * x


@torch.jit.script
def cube_(x):
    return square_(x).mul_(x)


@torch.jit.script
def pow4(x):
    return square(square(x))


@torch.jit.script
def pow4_(x):
    return square_(square_(x))


@torch.jit.script
def pow5(x):
    return x * pow4(x)


@torch.jit.script
def pow5_(x):
    return pow4_(x).mul_(x)


@torch.jit.script
def pow6(x):
    return square(cube(x))


@torch.jit.script
def pow6_(x):
    return square_(cube_(x))


@torch.jit.script
def pow7(x):
    return pow6(x) * x


@torch.jit.script
def pow7_(x):
    return pow6_(x).mul_(x)


@torch.jit.script
def floor_div(x, y) -> torch.Tensor:
    return torch.div(x, y, rounding_mode='floor')


@torch.jit.script
def floor_div_int(x, y: int) -> torch.Tensor:
    return torch.div(x, y, rounding_mode='floor')


@torch.jit.script
def trunc_div(x, y) -> torch.Tensor:
    return torch.div(x, y, rounding_mode='trunc')


@torch.jit.script
def trunc_div_int(x, y: int) -> torch.Tensor:
    return torch.div(x, y, rounding_mode='trunc')

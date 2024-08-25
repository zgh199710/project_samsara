import numpy as np
gpu_enable = True
try:
    import torch
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False
from samsara import Variable


def get_array_module(x):
    """返回 `x` 的数组模块。

    参数:
        x (dezero.Variable 或 numpy.ndarray 或 cupy.ndarray): 用于确定应该使用 NumPy 还是 CuPy 的值。

    返回:
        module: 根据参数返回 `cupy` 或 `numpy`。
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    """转换为 `numpy.ndarray`。

    参数:
        x (`numpy.ndarray` 或 `cupy.ndarray`): 可以转换为 `numpy.ndarray` 的任意对象。
    返回:
        `numpy.ndarray`: 转换后的数组。
    """
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    """转换为 `cupy.ndarray`。

    参数:
        x (`numpy.ndarray` 或 `cupy.ndarray`): 可以转换为 `cupy.ndarray` 的任意对象。
    返回:
        `cupy.ndarray`: 转换后的数组。
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('无法加载 CuPy。请安装 CuPy！')
    return cp.asarray(x)
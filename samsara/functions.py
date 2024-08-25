import numpy as np
import samsara
from samsara import cuda, utils
from samsara.core import Function, Variable, as_variable, as_array


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)

import torch
import cupy as cp

# 编写 CUDA 内核
scatter_add_kernel = cp.RawKernel(r'''
extern "C" __global__
void scatter_add(float* gx, const int* slices, const float* gy, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&gx[slices[idx]], gy[idx]);
    }
}
''', 'scatter_add')


# 定义 scatter_add 函数
def scatter_add(gx, slices, gy):
    size = len(gy)
    block_size = 256
    grid_size = (size + block_size - 1) // block_size

    scatter_add_kernel((grid_size,), (block_size,), gx, slices, gy, size)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = samsara.cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


def expand_dims(x, axis):
    x = as_variable(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))


def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))


# =============================================================================
# concat / sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================


class Concat(Function):
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, *xs):
        # 如果输入是单个元组，则解包它
        if len(xs) == 1 and isinstance(xs[0], tuple):
            xs = xs[0]

        xp = cuda.get_array_module(xs[0])
        self.x_shapes = [x.shape for x in xs]
        y = xp.concatenate([x.data if isinstance(x, Variable) else x for x in xs], axis=self.axis)
        return y

    def backward(self, gy):
        xp = cuda.get_array_module(gy)
        shapes = self.x_shapes
        indices = xp.cumsum([0] + [shape[self.axis] for shape in shapes[:-1]])
        gxs = xp.split(gy, indices, axis=self.axis)
        return tuple([gx.reshape(shape) for gx, shape in zip(gxs, shapes)])


def concat(*xs, axis=0):
    # 如果输入是单个元组，则解包它
    if len(xs) == 1 and isinstance(xs[0], tuple):
        xs = xs[0]
    return Concat(axis)(*xs)

class Sqrt(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sqrt(x)
        return y

    def backward(self, gy):
        x,  = self.inputs
        xp = cuda.get_array_module(x)
        gx = gy / (2 * xp.sqrt(x.data))
        return gx


def sqrt(x):
    return Sqrt()(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis,
                                        self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx



def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = samsara.cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)


mean = average


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


class Linear(Function):
    """实现仿射变换的运算。
       数学公式: y = x * W + b
       其中 x 是输入，W 是权重矩阵，b 是偏置向量。
       """
    def forward(self, x, W, b):
        """前向传播计算线性变换。

            参数:
                x (ndarray): 输入数据，形状为 (batch_size, input_dim)
                W (ndarray): 权重矩阵，形状为 (input_dim, output_dim)
                b (ndarray or None): 偏置向量，形状为 (output_dim,)

            返回:
                ndarray: 线性变换的结果
            """
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None  # Release t.data (ndarray) for memory efficiency
    return y


# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu / mish / silu
# =============================================================================
def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class Sigmoid(Function):
    """实现 Sigmoid 激活函数。

    数学公式: σ(x) = 1 / (1 + exp(-x))
    """
    def forward(self, x):
        """前向传播计算 Sigmoid。

           参数:
               x (ndarray): 输入数据

           返回:
               ndarray: Sigmoid 激活后的结果

           注意:
               使用 tanh 实现以提高数值稳定性。
           """
        xp = cuda.get_array_module(x)
        # y = 1 / (1 + xp.exp(-x))
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    """实现 ReLU（Rectified Linear Unit）激活函数。

       数学公式: ReLU(x) = max(0, x)
       """

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


class Mish(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x * xp.tanh(xp.log(1 + np.exp(x)))
        return y

    def backward(self, gy):
        x, = self.inputs
        omega = 4 * (x + 1) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x) * (4 * x + 6)
        delta = 2 * np.exp(x) + np.exp(2 * x) + 2
        gx = gy * (omega / (delta * delta))
        return gx


def mish(x):
    return Mish()(x)


class SiLU(Function):
    def forward(self, x):
        y = x * sigmoid(x).data
        return y

    def backward(self, gy):
        x,  = self.inputs
        sigmoid_x = sigmoid(x).data
        gx = gy * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
        return gx


def silu(x):
    return SiLU()(x)


class Softmax(Function):
    """实现 Softmax 函数。

    数学公式: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    """

    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        """前向传播计算 Softmax。

           参数:
               x (ndarray): 输入数据

           返回:
               ndarray: Softmax 后的结果

           注意:
               使用减去最大值的技巧来提高数值稳定性。
           """
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        """反向传播计算梯度。

          参数:
              gy (ndarray): 输出梯度

          返回:
              ndarray: 输入梯度

          数学推导:
          ∂softmax(x_i)/∂x_j = softmax(x_i) * (δ_ij - softmax(x_j))
          """
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


class LogSoftmax(Function):
    """实现 Log-Softmax 函数。

    数学公式: log_softmax(x_i) = x_i - log(Σ_j exp(x_j))
    """
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)


class LeakyReLU(Function):
    """实现 Leaky ReLU 激活函数。

    数学公式:
    LeakyReLU(x) = x       if x > 0
                 = slope*x if x <= 0
    """
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope
        return y

    def backward(self, gy):
        """反向传播计算梯度。

           参数:
               gy (ndarray): 输出梯度

           返回:
               ndarray: 输入梯度

           数学推导:
           ∂LeakyReLU(x)/∂x = 1     if x > 0
                            = slope if x <= 0
           """
        x, = self.inputs
        mask = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        gx = gy * mask
        return gx


def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)


# =============================================================================
# loss function: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# =============================================================================
def mean_squared_error_simple(x0, x1):
    x0, x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1
    y = sum(diff ** 2) / len(diff)
    return y


class MeanSquaredError(Function):
    """实现均方误差（MSE）损失函数。

       数学公式: MSE = (1/n) * Σ (y_pred - y_true)^2
       """

    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        """反向传播计算梯度。

           参数:
               gy (float): 输出梯度

           返回:
               tuple: 包含预测值和真实值的梯度

           数学推导:
           ∂MSE/∂x0 = (2/n) * (x0 - x1)
           ∂MSE/∂x1 = (-2/n) * (x0 - x1)
           """
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)  # To avoid log(0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
    """实现 Softmax 函数和交叉熵损失的组合。

      这个类combines了 softmax 激活函数和交叉熵损失，以提高数值稳定性和计算效率。

      数学公式:
      Softmax: σ(x_i) = exp(x_i) / Σ_j exp(x_j)
      交叉熵: H(y, p) = -Σ_i y_i * log(p_i)
      其中 y 是真实标签（one-hot编码），p 是预测概率。

      Forward计算:
      L = -(1/N) * Σ_n Σ_i y_ni * log(p_ni)
      其中 N 是batch size，i 是类别索引。
      """
    def forward(self, x, t):
        """前向传播计算 softmax 交叉熵损失。

           参数:
               x (ndarray): 形状为 (N, C) 的输入 logits，其中 N 是批量大小，C 是类别数。
               t (ndarray): 形状为 (N,) 的真实标签索引。

           返回:
               float: 批量的平均交叉熵损失。

           注意:
               这个方法使用 log-sum-exp 技巧来提高数值稳定性。
           """
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        """反向传播计算梯度。

           参数:
               gy (float): 输出梯度。

           返回:
               ndarray: 相对于输入 x 的梯度。

           数学推导:
           ∂L/∂x_i = p_i - y_i
           其中 p_i 是 softmax 输出，y_i 是真实标签（one-hot编码）。
           """
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def sigmoid_cross_entropy(x, t):
    """Sigmoid 交叉熵损失函数实现。

       数学公式:
       L = -(1/N) * Σ_n [t * log(σ(x)) + (1-t) * log(1-σ(x))]
       其中 σ(x) 是 sigmoid 函数

       参数:
           x (Variable): 输入 logits
           t (Variable): 真实标签（0 或 1）

       返回:
           Variable: Sigmoid 交叉熵损失
       """
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = as_variable(x), as_variable(t)
    N = len(x)
    p = sigmoid(x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


def binary_cross_entropy(p, t):
    """二元交叉熵损失函数实现。

       数学公式:
       L = -(1/N) * Σ_n [t * log(p) + (1-t) * log(1-p)]

       参数:
           p (Variable): 预测概率
           t (Variable): 真实标签（0 或 1）

       返回:
           Variable: 二元交叉熵损失
       """
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


# =============================================================================
# accuracy / dropout / batch_norm /
# =============================================================================
def accuracy(y, t):
    """
    注意:
        此函数不可微。
    """
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))


def dropout(x, dropout_ratio=0.5):
    """实现 Dropout 正则化。

      数学公式:
      y = x * mask / (1 - dropout_ratio)  (训练时)
      y = x                               (测试时)
      其中 mask 是一个随机二元掩码

      参数:
          x (Variable): 输入数据
          dropout_ratio (float): Dropout 比率

      返回:
          Variable: Dropout 后的结果
      """
    x = as_variable(x)

    if samsara.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x


class BatchNorm(Function):
    """实现批量归一化（Batch Normalization）。

       数学公式:
       y = γ * (x - μ) / √(σ^2 + ε) + β
       其中 μ 是均值，σ^2 是方差，γ 是缩放因子，β 是偏移因子，ε 是小常数。
       """

    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        """前向传播计算批量归一化。

           参数:
               x (ndarray): 输入数据
               gamma (ndarray): 缩放因子
               beta (ndarray): 偏移因子

           """
        assert x.ndim == 2 or x.ndim == 4

        x_ndim = x.ndim
        if x_ndim == 4:
            N, C, H, W = x.shape
            # (N, C, H, W) -> (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        xp = cuda.get_array_module(x)

        if samsara.Config.train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            m = x.size // gamma.size
            s = m - 1. if m - 1. > 1. else 1.
            adjust = m / s  # unbiased estimation
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std
        y = gamma * xc + beta

        if x_ndim == 4:
            # (N*H*W, C) -> (N, C, H, W)
            y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        gy_ndim = gy.ndim
        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gy = gy.transpose(0, 2, 3, 1).reshape(-1, C)

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        if gy_ndim == 4:
            gx = gx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return gx, ggamma, gbeta


def batch_nrom(x, gamma, beta, mean, var, decay=0.9, eps=2e-5):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


def embed_id(x, W):
    return W[x]


# =============================================================================
# max / min / minimum / maximum / clip
# =============================================================================
class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Minimum(Function):
    def forward(self, x0, x1):
        xp = cuda.get_array_module(x0)
        y = xp.minimum(x0, x1)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        mask = x0.data <= x1.data
        gx0 = gy * mask
        gx1 = gy * (~mask)
        return gx0, gx1


def minimum(x0, x1):
    return Minimum()(x0, x1)


class Maximum(Function):
    def forward(self, x0, x1):
        xp = cuda.get_array_module(x0)
        y = xp.maximum(x0, x1)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        mask = x0.data >= x1.data
        gx0 = gy * mask
        gx1 = gy * (~mask)
        return gx0, gx1


def maximum(x0, x1):
    return Maximum()(x0, x1)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        if self.x_max is None:
            self.x_max = 0
        if self.x_min is None:
            self.x_min = 0
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

# =============================================================================
# conv2d / col2im / im2col / basic_math
# =============================================================================
from samsara.functions_conv import conv2d
from samsara.functions_conv import deconv2d
from samsara.functions_conv import conv2d_simple
from samsara.functions_conv import im2col
from samsara.functions_conv import col2im
from samsara.functions_conv import pooling_simple
from samsara.functions_conv import pooling
from samsara.functions_conv import average_pooling
from samsara.core import add
from samsara.core import sub
from samsara.core import rsub
from samsara.core import mul
from samsara.core import div
from samsara.core import neg
from samsara.core import pow
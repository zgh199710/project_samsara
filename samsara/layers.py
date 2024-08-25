import os
import weakref
import numpy as np
import samsara.functions as F
from samsara import cuda
from samsara.core import Parameter
from samsara.utils import pair
from samsara.functions_conv import pooling


# =============================================================================
# 神经网络层的基类
# =============================================================================
class Layer:
    """神经网络层的基类。

    这个类提供了管理参数、前向传播、梯度清零等基本功能。
    """

    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        """设置属性时，如果是 Parameter 或 Layer 类型，则加入到 _params 集合中。"""
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        """调用层实例时执行前向传播，并保存输入和输出的弱引用。"""
        outputs = self.forward(*inputs)
        if isinstance(outputs, dict):
            self.inputs = [weakref.ref(x) for x in inputs]
            self.outputs = {}
            for k, v in outputs.items():
                if v is None:
                    self.outputs[k] = None
                elif isinstance(v, (int, float, bool, str)):
                    self.outputs[k] = v
                else:
                    try:
                        self.outputs[k] = weakref.ref(v)
                    except TypeError:
                        self.outputs[k] = v
            return outputs
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = []
        for y in outputs:
            if y is None:
                self.outputs.append(None)
            elif isinstance(y, (int, float, bool, str)):
                self.outputs.append(y)
            else:
                try:
                    self.outputs.append(weakref.ref(y))
                except TypeError:
                    self.outputs.append(y)
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# =============================================================================
# Linear / Conv2d / Deconv2d
# =============================================================================
class Linear(Layer):
    """全连接层。

    数学公式：y = x * W^T + b

    参数:
        out_size (int): 输出特征的数量
        nobias (bool): 如果为 True，则不使用偏置
        dtype: 数据类型
        in_size (int): 输入特征的数量（可选）
    """
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        """初始化权重。使用 kaiming 初始化方法。"""
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    """2D 卷积层。

     数学公式：
     y[n,f,i,j] = Σ_c Σ_h Σ_w x[n,c,i*s+h,j*s+w] * W[f,c,h,w] + b[f]

     参数:
         out_channels (int): 输出通道数
         kernel_size (int or tuple): 卷积核大小
         stride (int or tuple): 步长
         pad (int or tuple): 填充大小
         nobias (bool): 如果为 True，则不使用偏置
         dtype: 数据类型
         in_channels (int): 输入通道数（可选）
     """
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        # print("Conv2d input shape:", x.shape)
        # print("Conv2d weight shape:", self.W.shape)
        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        # print("Conv2d output shape:", y.shape)
        return y


class Deconv2d(Layer):
    """2D 反卷积（转置卷积）层。"""
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y


# =============================================================================
# RNN / LSTM
# =============================================================================
class RNN(Layer):
    """简单循环神经网络 (Elman RNN) 层，使用 tanh 激活函数。

    数学公式：
    h_t = tanh(W_xh * x_t + W_hh * h_(t-1))

    Args:
        hidden_size (int): 隐藏状态的特征数
        in_size (int): 输入的特征数（可选）
    """
    def __init__(self, hidden_size, in_size=None):

        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new


class LSTM(Layer):
    """长短期记忆 (LSTM) 层。

    数学公式：
    f_t = σ(W_xf * x_t + W_hf * h_(t-1) + b_f)
    i_t = σ(W_xi * x_t + W_hi * h_(t-1) + b_i)
    o_t = σ(W_xo * x_t + W_ho * h_(t-1) + b_o)
    u_t = tanh(W_xu * x_t + W_hu * h_(t-1) + b_u)
    c_t = f_t ⊙ c_(t-1) + i_t ⊙ u_t
    h_t = o_t ⊙ tanh(c_t)

    其中 σ 是 sigmoid 函数，⊙ 表示元素wise乘法。

    Args:
        hidden_size (int): 隐藏状态的特征数
        in_size (int): 输入的特征数（可选）
    """

    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        H, I = hidden_size, in_size
        self.x2f = Linear(H, in_size=I)
        self.x2i = Linear(H, in_size=I)
        self.x2o = Linear(H, in_size=I)
        self.x2u = Linear(H, in_size=I)
        self.h2f = Linear(H, in_size=H, nobias=True)
        self.h2i = Linear(H, in_size=H, nobias=True)
        self.h2o = Linear(H, in_size=H, nobias=True)
        self.h2u = Linear(H, in_size=H, nobias=True)
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)

        h_new = o * F.tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new


# =============================================================================
# EmbedID / BatchNorm / MaxPool2d / LayerNormalization / GroupNorm
# =============================================================================
class EmbedID(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = Parameter(np.random.randn(in_size, out_size), name='W')

    def __call__(self, x):
        y = self.W[x]
        return y


class BatchNorm(Layer):
    def __init__(self):
        super().__init__()
        # `.avg_mean` and `.avg_var` are `Parameter` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Parameter(None, name='avg_mean')
        self.avg_var = Parameter(None, name='avg_var')
        self.gamma = Parameter(None, name='gamma')
        self.beta = Parameter(None, name='beta')

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def __call__(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_nrom(x, self.gamma, self.beta, self.avg_mean.data,
                            self.avg_var.data)


class MaxPool2d(Layer):
    """2D 最大池化层。

    参数:
        kernel_size (int or tuple): 池化核的大小
        stride (int or tuple): 步长(默认等于kernel_size)
        pad (int or tuple): 填充大小
    """

    def __init__(self, kernel_size, stride=None, pad=0):
        super().__init__()
        self.kernel_size = pair(kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x):
        y = pooling(x, self.kernel_size, self.stride, self.pad)
        return y


class LayerNormalization(Layer):
    """层归一化。

    数学公式：
    y = (x - E[x]) / sqrt(Var[x] + epsilon) * gamma + beta

    Args:
        d_model (int): 模型的维度
        eps (float): 用于数值稳定的小值
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = Parameter(np.ones(d_model), name='gamma')
        self.beta = Parameter(np.zeros(d_model), name='beta')
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class GroupNorm(Layer):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = Parameter(cuda.as_numpy(np.ones(num_channels)), name='gamma')
        self.beta = Parameter(cuda.as_numpy(np.zeros(num_channels)), name='beta')

    def forward(self, x):
        xp = cuda.get_array_module(x)
        # x shape: (N, C, H, W)
        N, C, H, W = x.shape
        G = self.num_groups

        # Reshape x to (N, G, C//G, H, W)
        x = x.reshape((N, G, C // G, H, W))

        # Calculate mean and var
        mean = F.mean(x, axis=(2, 3, 4), keepdims=True)
        var = F.mean((x - mean) ** 2, axis=(2, 3, 4), keepdims=True)

        # Normalize
        x = (x - mean) / F.sqrt(var + self.eps)

        # Reshape back to (N, C, H, W)
        x = x.reshape((N, C, H, W))

        # Scale and shift
        gamma = cuda.as_cupy(self.gamma.data) if xp == cuda.cupy else self.gamma.data
        beta = cuda.as_cupy(self.beta.data) if xp == cuda.cupy else self.beta.data
        y = x * F.reshape(gamma, (1, -1, 1, 1)) + F.reshape(beta, (1, -1, 1, 1))

        return y
# =============================================================================
# attention
# =============================================================================


class MultiHeadAttention(Layer):
    """多头注意力层。

    数学公式：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

    Args:
        d_model (int): 模型的维度
        num_heads (int): 注意力头的数量
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.W_q(query).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = self.W_k(key).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = self.W_v(value).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = F.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = F.where(mask, scores, -1e9)
        attention = F.softmax(scores, axis=-1)

        out = F.matmul(attention, V)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        return self.W_o(out)


class PositionwiseFeedForward(Layer):
    """位置前馈网络层。

    数学公式：
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Args:
        d_model (int): 模型的维度
        d_ff (int): 前馈网络的隐藏层维度
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = Linear(d_model, d_ff)
        self.w_2 = Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class EncoderLayer(Layer):
    """Transformer 编码器层。

    Args:
        d_model (int): 模型的维度
        num_heads (int): 注意力头的数量
        d_ff (int): 前馈网络的隐藏层维度
        dropout_rate (float): Dropout 比率
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout = dropout_rate

    def forward(self, x, mask):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + F.dropout(attn_output, self.dropout))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + F.dropout(ff_output, self.dropout))
        return x


class DecoderLayer(Layer):
    """Transformer 解码器层。

    Args:
        d_model (int): 模型的维度
        num_heads (int): 注意力头的数量
        d_ff (int): 前馈网络的隐藏层维度
        dropout_rate (float): Dropout 比率
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout = dropout_rate

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + F.dropout(attn_output, self.dropout))
        attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + F.dropout(attn_output, self.dropout))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + F.dropout(ff_output, self.dropout))
        return x


class PositionalEncoding(Layer):
    """位置编码层。

    数学公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model (int): 模型的维度
        max_len (int): 最大序列长度
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = Parameter(pe[np.newaxis, :, :], name='pe')

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]



# =============================================================================
# Conv3x3 / Conv1x1 / BasicBlock
# =============================================================================
class Conv3x3(Layer):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv = Conv2d(out_planes, kernel_size=3, stride=stride, pad=1, nobias=True)
        self.in_planes = in_planes

    def forward(self, x):
        return self.conv(x)


class Conv1x1(Layer):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv = Conv2d(out_planes, kernel_size=1, stride=stride, pad=0, nobias=True)
        self.in_planes = in_planes

    def forward(self, x):
        return self.conv(x)




















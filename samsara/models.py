import numpy as np
from samsara import Layer
import samsara.functions as F
import samsara.layers as L
from samsara import utils


# =============================================================================
# Model / Sequential / MLP
# =============================================================================
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

    def __repr__(self):
        return self._repr_helper()

    def _repr_helper(self, indent=''):
        lines = [self.__class__.__name__ + '(']
        for name, module in self.__dict__.items():
            if isinstance(module, (L.Layer, Model, Sequential)):
                mod_str = self._format_module(name, module, indent + '  ')
                lines.append(mod_str)
        lines.append(indent + ')')
        return '\n'.join(lines)

    def _format_module(self, name, module, indent):
        if isinstance(module, L.Conv2d):
            return f"{indent}({name}): Conv2d({module.in_channels}, {module.out_channels}, kernel_size={module.kernel_size}, stride={module.stride}, padding={module.pad}, bias={module.b is not None})"
        elif isinstance(module, L.BatchNorm):
            return f"{indent}({name}): BatchNorm()"
        elif isinstance(module, L.MaxPool2d):
            return f"{indent}({name}): MaxPool2d(kernel_size={module.kernel_size}, stride={module.stride}, padding={module.pad})"
        elif isinstance(module, L.Linear):
            return f"{indent}({name}): Linear(in_size={module.in_size}, out_size={module.out_size})"
        elif isinstance(module, Sequential):
            lines = [f"{indent}({name}): Sequential("]
            for idx, layer in enumerate(module.layers):
                lines.append(self._format_module(str(idx), layer, indent + '  '))
            lines.append(indent + ')')
            return '\n'.join(lines)
        elif isinstance(module, Model):
            if module.__class__.__name__ == 'Conv':
                conv_info = f"{indent}({name}): Conv("
                if hasattr(module, 'convs') and module.convs:
                    first_conv = module.convs[0]
                    if isinstance(first_conv, L.Conv2d):
                        conv_info += f"in={first_conv.in_channels}, out={first_conv.out_channels}, "
                        conv_info += f"k={first_conv.kernel_size}, s={first_conv.stride}, p={first_conv.pad}"
                conv_info += ")"
                return conv_info
            elif module.__class__.__name__ == 'DecoupleHead':
                lines = [f"{indent}({name}): DecoupleHead("]
                lines.append(f"{indent}  cls_feats:")
                for i, conv in enumerate(module.cls_feats):
                    lines.append(self._format_module(f"    {i}", conv, indent + '  '))
                lines.append(f"{indent}  reg_feats:")
                for i, conv in enumerate(module.reg_feats):
                    lines.append(self._format_module(f"    {i}", conv, indent + '  '))
                lines.append(indent + ')')
                return '\n'.join(lines)
            elif module.__class__.__name__ == 'SPPF':
                lines = [f"{indent}({name}): SPPF("]
                lines.append(self._format_module("conv1", module.conv1, indent + '  '))
                lines.append(self._format_module("conv2", module.conv2, indent + '  '))
                lines.append(f"{indent}  pooling_size: {module.pooling_size}")
                lines.append(indent + ')')
                return '\n'.join(lines)
            else:
                lines = [f"{indent}({name}): {module.__class__.__name__}("]
                for subname, submodule in module.__dict__.items():
                    if isinstance(submodule, (L.Layer, Model, Sequential)):
                        lines.append(self._format_module(subname, submodule, indent + '  '))
                lines.append(indent + ')')
                return '\n'.join(lines)
        else:
            return f"{indent}({name}): {module.__class__.__name__}()"


class Sequential(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


# =============================================================================
# VGG
# =============================================================================
class VGG16(Model):
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz'
    # WEIGHTS_LOCAL_PATH = "vgg16.npz"

    def __init__(self, pretrained=False, weight=None):
        super().__init__()
        self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(1000)

        if pretrained:
            try:
                weights_path = utils.get_local_file(weight)

            except:
                weights_path = utils.get_file(VGG16.WEIGHTS_PATH)

            self.load_weights(weights_path)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        image = image.convert('RGB')
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=dtype)
        image = image[:, :, ::-1]
        image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
        image = image.transpose((2, 0, 1))
        return image


# =============================================================================
# ResNet
# =============================================================================


class ResNet(Model):
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/resnet{}.npz'

    def __init__(self, n_layers=152, pretrained=False):
        super().__init__()

        if n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))

        self.conv1 = L.Conv2d(64, 7, 2, 3)
        self.bn1 = L.BatchNorm()
        self.res2 = BuildingBlock(block[0], 64, 64, 256, 1)
        self.res3 = BuildingBlock(block[1], 256, 128, 512, 2)
        self.res4 = BuildingBlock(block[2], 512, 256, 1024, 2)
        self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 2)
        self.fc6 = L.Linear(1000)

        if pretrained:
            weights_path = utils.get_file(ResNet.WEIGHTS_PATH.format(n_layers))
            self.load_weights(weights_path)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.pooling(x, kernel_size=3, stride=2)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = _global_average_pooling_2d(x)
        x = self.fc6(x)
        return x


class ResNet152(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(152, pretrained)


class ResNet101(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(101, pretrained)


class ResNet50(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(50, pretrained)


def _global_average_pooling_2d(x):
    N, C, H, W = x.shape
    h = F.average_pooling(x, (H, W), stride=1)
    h = F.reshape(h, (N, C))
    return h


class BuildingBlock(Layer):
    def __init__(self, n_layers=None, in_channels=None, mid_channels=None,
                 out_channels=None, stride=None, downsample_fb=None):
        super().__init__()

        self.a = BottleneckA(in_channels, mid_channels, out_channels, stride,
                             downsample_fb)
        self._forward = ['a']
        for i in range(n_layers - 1):
            name = 'b{}'.format(i + 1)
            bottleneck = BottleneckB(out_channels, mid_channels)
            setattr(self, name, bottleneck)
            self._forward.append(name)

    def forward(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x


class BottleneckA(Layer):

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, downsample_fb=False):
        super().__init__()
        # In the original MSRA ResNet, stride=2 is on 1x1 convolution.
        # In Facebook ResNet, stride=2 is on 3x3 convolution.
        stride_1x1, stride_3x3 = (1, stride) if downsample_fb else (stride, 1)

        self.conv1 = L.Conv2d(mid_channels, 1, stride_1x1, 0,
                              nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(mid_channels, 3, stride_3x3, 1,
                              nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L.Conv2d(out_channels, 1, 1, 0, nobias=True)
        self.bn3 = L.BatchNorm()
        self.conv4 = L.Conv2d(out_channels, 1, stride, 0,
                              nobias=True)
        self.bn4 = L.BatchNorm()

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleneckB(Layer):

    def __init__(self, in_channels, mid_channels):
        super().__init__()

        self.conv1 = L.Conv2d(mid_channels, 1, 1, 0, nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(mid_channels, 3, 1, 1, nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L.Conv2d(in_channels, 1, 1, 0, nobias=True)
        self.bn3 = L.BatchNorm()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)


# =============================================================================
# SqueezeNet
# =============================================================================
class SqueezeNet(Model):
    def __init__(self, pretrained=False):
        pass

    def forward(self, x):
        pass


# =============================================================================
# Transfomer
# =============================================================================

class Transformer(Layer):
    """Transformer 模型。

    Args:
        src_vocab_size (int): 源语言词汇表大小
        tgt_vocab_size (int): 目标语言词汇表大小
        d_model (int): 模型的维度
        num_heads (int): 注意力头的数量
        num_layers (int): 编码器和解码器的层数
        d_ff (int): 前馈网络的隐藏层维度
        max_seq_length (int): 最大序列长度
        dropout_rate (float): Dropout 比率
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length,
                 dropout_rate):
        super().__init__()
        self.encoder_embed = L.Linear(src_vocab_size, d_model)
        self.decoder_embed = L.Linear(tgt_vocab_size, d_model)
        self.positional_encoding = L.PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = [L.EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        self.decoder_layers = [L.DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

        self.final_layer = L.Linear(d_model, tgt_vocab_size)
        self.dropout = dropout_rate

    def encode(self, src, src_mask):
        x = self.encoder_embed(src)
        x = self.positional_encoding(x)
        x = F.dropout(x, self.dropout)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.decoder_embed(tgt)
        x = self.positional_encoding(x)
        x = F.dropout(x, self.dropout)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        output = self.final_layer(dec_output)
        return output

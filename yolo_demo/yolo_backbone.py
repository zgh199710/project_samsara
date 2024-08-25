import numpy as np
import samsara
import samsara.functions as F
import samsara.layers as L
from samsara.models import Model, Layer, Sequential
from samsara import utils


# =============================================================================
# backbone
# =============================================================================

WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/resnet{}.npz'


def conv3x3(in_channels, out_channels, stride=1):
    '''

    :param in_channels:
    :param out_channels:
    :param stride:
    :return:
    '''

    return L.Conv2d(out_channels=out_channels, kernel_size=3, stride=stride, pad=1, nobias=True, in_channels=in_channels)


# 1x1
def conv1x1(in_channels, out_channels, stride=1):
    '''

    :param in_channels:
    :param out_channels:

    :return:
    '''

    return L.Conv2d(out_channels=out_channels, kernel_size=1, stride=stride, pad=0, nobias=True, in_channels=in_channels)


class BasicBlock(Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = L.BatchNorm()

        # 第二个已经是用于提取特征了，步长主要看要不要下采样才用
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = L.BatchNorm()
        self.downsample = downsample
        self.stride = stride

    def __repr__(self):
        return f'BasicBlock(inplanes={self.conv1.in_channels}, planes={self.conv2.out_channels})'

    def forward(self, x):
        identity = x

        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_2 = self.bn2(self.conv2(out_1))

        if self.downsample is not None:
            identity = self.downsample(x)

        out_2 += identity
        out = F.relu(out_2)
        return out


class Bottleneck(Model):
    """
    expansion 因子设为 4 是 ResNet 设计中基于大量实验和经验得出的最佳选择。这一选择在实践中证明能够在减少计算量的同时，
    保持和增强网络的特征表示能力。
    当然，根据具体应用和需求，expansion 因子是可以调整的，但需要通过实验验证其效果和效率。
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        '''
        1x1 卷积层用于降维和升维，3x3 卷积层用于特征提取。
        每个卷积层之后跟随一个批量归一化和 ReLU 激活函数。
        适合较大的网络，通常在 ResNet-50、ResNet-101 和 ResNet-152 中使用。
        参数量：
        相对较多，因为有三个卷积层，但通过 1x1 卷积层减少了计算量。
        1x1 卷积层在降维和升维过程中减少了参数量。
        '''
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = L.BatchNorm()
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = L.BatchNorm()

        # 维度恢复后的特征融合：最后一个 1x1 卷积层再将通道数扩展为 4 倍，
        # 可以使得每个输入特征图块都能充分利用所有通道的信息。
        self.conv3 = conv1x1(planes, planes * self.expansion)

        self.bn3 = L.BatchNorm()

        self.downsample = downsample
        self.stride = stride

    def __repr__(self):
        return f'Bottleneck(inplanes={self.conv1.in_channels}, planes={self.conv3.out_channels // self.expansion})'

    def forward(self, x):
        identity = x

        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_2 = F.relu(self.bn2(self.conv2(out_1)))
        out_3 = self.bn3(self.conv3(out_2))

        if self.downsample is not None:
            identity = self.downsample(x)

        out_3 += identity
        out = F.relu(out_3)

        return out


class ResNet(Model):
    # 如果 zero_init_residual 为 True，
    # 初始化每个残差块的最后一层批量归一化权重为 0，
    # 以使残差块初始为恒等映射，从而改善训练效果。=
    def __init__(self, block, layers, zero_init_residual=False, pretrained=False):
        super(ResNet, self).__init__()

        self.inplanes = 64
        # 第一个卷积层，接收 3 通道的输入（如 RGB 图像），输出 64 通道特征图。卷积核大小为 7x7，步幅为 2，填充为 3。
        self.conv1 = L.Conv2d(64, 7, 2, 3, in_channels=3)
        self.bn1 = L.BatchNorm()

        self.maxpool = L.MaxPool2d(kernel_size=3, stride=2, pad=1)

        # 残差块层：layer1到layer4
        # 是四个不同的残差块，每个块包含若干个残差单元。
        # 每一层的通道数逐渐增加（64 -> 128 -> 256 -> 512），每一层的输出特征图尺寸逐渐减小（由于步幅和池化操作）。
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 确定是否需要下采样:
        # 如果步幅 stride 不等于 1 或输入通道数 self.inplanes 不等于输出通道数 planes * block.expansion，
        # 则需要进行下采样。
        # 下采样通过一个包含 1x1 卷积和批量归一化的序列实现，用于调整特征图的大小和通道数，以匹配残差块的输入和输出。
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = samsara.models.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                L.BatchNorm()
            )
        # 创建残差块列表：
        # 使用 block 创建第一个残差块，并将其添加到 layers 列表中。这个残差块可能包含下采样。
        # 更新 self.inplanes 为当前层的输出通道数，以便下一层的输入通道数正确匹配。
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        # 添加剩余的残差块：
        # 使用循环创建剩余的残差块，这些残差块不包含下采样，因为只需要第一个残差块进行下采样以调整特征图的尺寸和通道数。
        # 返回残差块层：
        # 将 layers列表中的所有残差块使用nn.Sequential包装成一个连续的层，并返回。
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        """
        在 Python 中，前面的星号 * 是一个解包操作符，用于将一个列表或元组中的元素作为独立的参数传递给函数或方法。
        在 nn.Sequential(*layers) 中，*layers 的作用是将 layers 列表中的所有元素解包并作为独立的参数传递给 
        nn.Sequential 构造函数。

        在 _make_layer 方法中，layers 列表包含若干个残差块。通过 *layers，这些残差块会被传递给 nn.Sequential，
        从而创建一个顺序连接的残差层。
        这是 ResNet 结构中的核心部分之一，通过这种方式，我们可以方便地将多个残差块组合成更复杂的网络结构。
        """
        return samsara.models.Sequential(*layers)

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C, H, W]
        Output:
            c5: (Tensor) -> [B, C, H/32, W/32]
        """
        # print("ResNet input shape:", x.shape)
        # print("Conv1 weight shape:", self.conv1.W.shape)
        c1 = self.conv1(x)  # [B, C, H/2, W/2]
        print("+++++++++++++++++++")
        # print(c1)
        c1 = self.bn1(c1)  # [B, C, H/2, W/2]

        c1 = F.relu(c1)  # [B, C, H/2, W/2]

        c2 = self.maxpool(c1)  # [B, C, H/4, W/4]
        # 按顺序通过四个残差块层。
        c2 = self.layer1(c2)  # [B, C, H/4, W/4]
        c3 = self.layer2(c2)  # [B, C, H/8, W/8]
        c4 = self.layer3(c3)  # [B, C, H/16, W/16]
        c5 = self.layer4(c4)  # [B, C, H/32, W/32]

        return c5


# --------------------- 构建ResNet网络的函数 -----------------------
"""
**kwargs 是 Python 函数定义中的一个特殊语法，用于接受任意数量的关键字参数。这些参数会被收集到一个字典中。
在 resnet18 函数中，**kwargs 允许传递额外的关键字参数给 ResNet 类的构造函数。
"""


def resnet18(pretrained=False, n_layers=18, **kwargs):
    # 对应 ResNet-18 的四个残差层，每个层中包含 2 个 BasicBlock
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        weights_path = utils.get_file(WEIGHTS_PATH.format(n_layers))
        model.load_weights(weights_path)

    return model


def resnet34(pretrained=False, n_layers=34, **kwargs):
    """搭建 ResNet-34 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weights_path = utils.get_file(WEIGHTS_PATH.format(n_layers))
        model.load_weights(weights_path)

    return model


## 搭建ResNet-50网络
def resnet50(pretrained=False, n_layers=50, **kwargs):
    """搭建 ResNet-50 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weights_path = utils.get_file(WEIGHTS_PATH.format(n_layers))
        model.load_weights(weights_path)
    return model


## 搭建ResNet-101网络
def resnet101(pretrained=False, n_layers=101, **kwargs):
    """搭建 ResNet-101 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        weights_path = utils.get_file(WEIGHTS_PATH.format(n_layers))
        model.load_weights(weights_path)
    return model


## 搭建ResNet-152网络
def resnet152(pretrained=False, n_layers=152, **kwargs):
    """搭建 ResNet-152 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        weights_path = utils.get_file(WEIGHTS_PATH.format(n_layers))
        model.load_weights(weights_path)
    return model


## 搭建ResNet网络
def build_backbone(model_name='resnet18', pretrained=False):
    if model_name == 'resnet18':
        model = resnet18(pretrained)
        feat_dim = 512  # 网络的最终输出的feature的通道维度为512
    elif model_name == 'resnet34':
        model = resnet34(pretrained)
        feat_dim = 512  # 网络的最终输出的feature的通道维度为512
    elif model_name == 'resnet50':
        model = resnet50(pretrained)
        feat_dim = 2048  # 网络的最终输出的feature的通道维度为2048
    elif model_name == 'resnet101':
        model = resnet101(pretrained)
        feat_dim = 2048  # 网络的最终输出的feature的通道维度为2048

    return model, feat_dim


if __name__ == '__main__':
    model, feat_dim = build_backbone(model_name='resnet101', pretrained=False)

    print(model)

    # 输入图像的参数
    batch_size = 2
    image_channel = 3
    image_height = 512
    image_width = 1024

    # 随机生成一张图像
    image = np.random.randn(batch_size, image_channel, image_height, image_width).astype(np.float32)

    output = model(image)

    print(output.shape)
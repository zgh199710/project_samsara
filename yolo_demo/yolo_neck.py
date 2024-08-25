import numpy as np
import samsara
import samsara.functions as F
import samsara.layers as L
from samsara.models import Model


def get_conv2d(c1, c2, k, p, s, d, g, bias=False):
    return L.Conv2d(out_channels=c2, kernel_size=k, stride=s, pad=p, nobias=not bias, in_channels=c1)


def get_activation(act_type=None):
    if act_type == 'relu':
        return F.relu
    elif act_type == 'LeakyReLU':
        return lambda x: F.leaky_relu(x, slope=0.1)
    elif act_type == 'mish':
        return lambda x: F.mish(x)
    elif act_type == 'silu':
        return lambda x: F.silu(x)
    elif act_type is not None:
        return lambda x: x
    else:
        raise NotImplementedError('Activation {} not implemented.'.format(act_type))


def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return L.BatchNorm()
    elif norm_type == 'GN':
        return L.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is not None:
        return lambda x: x
    else:
        raise NotImplementedError('Normalization {} not implemented.'.format(norm_type))


class Conv(Model):
    def __init__(self,
                 inplanes,
                 outplanes,
                 kernel_size=1,
                 padding=0,
                 stride=1,
                 dilation=1,
                 act_type='LeakyReLU',
                 norm_type='BN',
                 depthwise=False
                 ):
        super(Conv, self).__init__()
        self.convs = []
        add_bias = False if norm_type else True

        if depthwise:
            self.convs.append(get_conv2d(inplanes, inplanes, k=kernel_size, p=padding, s=stride, d=dilation,
                                         g=inplanes, bias=add_bias))
            if norm_type:
                self.convs.append(get_norm(norm_type, inplanes))
            if act_type:
                self.convs.append(get_activation(act_type))

            self.convs.append(get_conv2d(inplanes, outplanes, k=1, p=0, s=1,
                                         d=dilation, g=1, bias=add_bias))
            if norm_type:
                self.convs.append(get_norm(norm_type, outplanes))
            if act_type:
                self.convs.append(get_activation(act_type))
        else:
            self.convs.append(get_conv2d(inplanes, outplanes, k=kernel_size, p=padding, s=stride,
                                         d=dilation, g=1, bias=add_bias))
            if norm_type:
                self.convs.append(get_norm(norm_type, outplanes))
            if act_type:
                self.convs.append(get_activation(act_type))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class SPPF(Model):
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, pooling_size=5,
                 act_type='LeakyReLU', norm_type='BN'):
        super(SPPF, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.conv1 = Conv(in_dim, inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.conv2 = Conv(inter_dim * 4, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.pooling_size = pooling_size
        self.pad = pooling_size // 2

    def forward(self, x):
        x = self.conv1(x)
        out1 = F.pooling(x, kernel_size=self.pooling_size, stride=1, pad=self.pad)
        out2 = F.pooling(out1, kernel_size=self.pooling_size, stride=1, pad=self.pad)
        out3 = F.pooling(out2, kernel_size=self.pooling_size, stride=1, pad=self.pad)

        concat = F.concat((x, out1, out2, out3), axis=1)
        return self.conv2(concat)


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))

    if model == 'sppf':
        neck = SPPF(
            in_dim=in_dim,
            out_dim=out_dim,
            expand_ratio=cfg['expand_ratio'],
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm']
        )
    else:
        raise NotImplementedError('Neck {} not implemented.'.format(cfg['neck']))

    return neck


if __name__ == '__main__':
    # 测试代码
    cfg = {
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'LeakyReLU',
        'neck_norm': 'BN'
    }

    in_dim = 512
    out_dim = 256

    neck = build_neck(cfg, in_dim, out_dim)
    print(neck)
    # 创建随机输入
    batch_size = 2
    height = 20
    width = 20
    x = np.random.randn(batch_size, in_dim, height, width).astype(np.float32)

    # 前向传播
    output = neck(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
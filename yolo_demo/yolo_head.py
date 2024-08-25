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


class DecoupleHead(Model):
    def __init__(self, cfg, in_dim, out_dim, num_classes=80):
        super(DecoupleHead, self).__init__()
        print("===================================")
        print('Head: Decoupled Head')
        self.in_dim = in_dim
        self.num_cls_head = cfg['num_cls_head']
        self.num_reg_head = cfg['num_reg_head']
        self.act_type = cfg['head_act']
        self.norm_type = cfg['head_norm']

        cls_feats = []
        self.cls_out_dim = max(out_dim, num_classes)
        for i in range(cfg['num_cls_head']):
            if i == 0:
                cls_feats.append(
                    Conv(in_dim, self.cls_out_dim, kernel_size=3,
                         padding=1, stride=1, act_type=self.act_type,
                         norm_type=self.norm_type,
                         depthwise=cfg['head_depthwise'])
                )
            else:
                cls_feats.append(
                    Conv(self.cls_out_dim, self.cls_out_dim, kernel_size=3,
                         padding=1, stride=1, act_type=self.act_type,
                         norm_type=self.norm_type,
                         depthwise=cfg['head_depthwise'])
                )

        reg_feats = []
        self.reg_out_dim = max(out_dim, 64)
        for i in range(cfg['num_reg_head']):
            if i == 0:
                reg_feats.append(
                    Conv(in_dim, self.reg_out_dim, kernel_size=3,
                         padding=1, stride=1, act_type=self.act_type,
                         norm_type=self.norm_type,
                         depthwise=cfg['head_depthwise'])
                )
            else:
                cls_feats.append(
                    Conv(self.reg_out_dim, self.reg_out_dim, kernel_size=3,
                         padding=1, stride=1, act_type=self.act_type,
                         norm_type=self.norm_type,
                         depthwise=cfg['head_depthwise'])
                )

        self.cls_feats = cls_feats
        self.reg_feats = reg_feats

    def forward(self, x):
        cls_feats = x
        for feat in self.cls_feats:
            cls_feats = feat(cls_feats)

        reg_feats = x
        for feat in self.reg_feats:
            reg_feats = feat(reg_feats)

        return cls_feats, reg_feats


def build_head(cfg, in_dim, out_dim, num_classes=80):
    head = DecoupleHead(cfg, in_dim, out_dim, num_classes)
    return head


if __name__ == '__main__':
    cfg = {
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_depthwise': False,
        'reg_max': 16,
    }

    batch_size = 2
    feat_channel = 512
    feat_height = 20
    feat_width = 20

    feature = np.random.randn(batch_size, feat_channel, feat_height, feat_width).astype(np.float32)

    model = build_head(cfg=cfg, in_dim=feat_channel, out_dim=256, num_classes=80)
    print(model)
    cls_feat, reg_feat = model(feature)

    print(cls_feat.shape)
    print(reg_feat.shape)
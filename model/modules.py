import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_weights import init_weights
from timm.models.layers import DropPath, trunc_normal_


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation
    # default_act=nn.ReLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    

class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class ChannelAttention(nn.Module):
    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))
    

class MAM(nn.Module):
    ### Multi-level Aggregation Module
    def __init__(self, in_ch):
        super(MAM, self).__init__()

        # previous conv
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()

        # latter conv
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()

        self.cbam=CBAM(in_ch)
        self.fuse=ConvModule(in_ch, in_ch)

    def forward(self, x_pre, x_cur, x_lat):
        # previois conv
        x_pre = self.downsample2(x_pre)
        pre_sa = x_cur.mul(self.pre_sa(x_pre))

        # latter conv
        x_lat = self.upsample2(x_lat)
        lat_sa = x_cur.mul(self.lat_sa(x_lat))

        x=self.cbam(x_cur)
        x=x+pre_sa+lat_sa+x_cur
        x_LocAndGlo = self.fuse(x)
        return x_LocAndGlo
  

class MAM1(nn.Module):
    ### Multi-level Aggregation Module
    def __init__(self, in_ch):
        super(MAM1, self).__init__()

        # latter conv
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()

        self.cbam=CBAM(in_ch)
        self.fuse=ConvModule(in_ch, in_ch)

    def forward(self, x_cur, x_lat):

        # latter conv
        x_lat = self.upsample2(x_lat)
        lat_sa = x_cur.mul(self.lat_sa(x_lat))

        x=self.cbam(x_cur)
        x=x+lat_sa+x_cur
        x_LocAndGlo = self.fuse(x)
        return x_LocAndGlo
  
class MAM5(nn.Module):
    ### Multi-level Aggregation Module
    def __init__(self, in_ch):
        super(MAM5, self).__init__()

        # previous conv
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()

        self.cbam=CBAM(in_ch)
        self.fuse=ConvModule(in_ch, in_ch)

    def forward(self, x_pre, x_cur):
        # previois conv
        x_pre = self.downsample2(x_pre)
        pre_sa = x_cur.mul(self.pre_sa(x_pre))

        x=self.cbam(x_cur)
        x=x+pre_sa+x_cur
        x_LocAndGlo = self.fuse(x)
        return x_LocAndGlo
  



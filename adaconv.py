import torch
from torch import nn
import typing
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from losses import calc_mean_std
from math import ceil, floor


class KernelPredictor(nn.Module):
    def __init__(self, in_channels, n_groups, out_channels, style_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if not out_channels is None else in_channels
        self.w_channels = style_channels
        self.n_groups = in_channels//n_groups
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) / 2
        self.spatial = nn.Conv2d(style_channels,
                                 in_channels * self.out_channels // self.n_groups,
                                 kernel_size=kernel_size,
                                 padding=(ceil(padding), ceil(padding)),
                                 padding_mode='zeros')
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      self.out_channels * self.out_channels,
                      kernel_size=1)
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      self.out_channels,
                      kernel_size=1)
        )

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
            m.requires_grad = True

    def forward(self, w):
        w_spatial = self.spatial(w)
        w_spatial = w_spatial.reshape(len(w),
                                      self.out_channels,
                                      self.in_channels // self.n_groups,
                                      self.kernel_size, self.kernel_size)

        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.reshape(len(w),
                                          self.out_channels,
                                          self.out_channels,
                                          1, 1)

        bias = self.bias(w)
        bias = bias.reshape(len(w),
                            self.out_channels)

        return w_spatial, w_pointwise, bias

class AdaConv(nn.Module):
    def __init__(self, in_channels, n_groups, s_d = 64, c_out = None, kernel_size=3, norm = True):
        super().__init__()
        self.kernel_predictor = KernelPredictor(in_channels, n_groups, in_channels, s_d, kernel_size)
        self.conv = AdaConv2d(in_channels, c_out = c_out, kernel_size=kernel_size, n_groups = n_groups, norm=norm)
    def forward(self, style_enc, x):
        w_spatial, w_pointwise, bias = self.kernel_predictor(style_enc)
        x = self.conv(x, w_spatial, w_pointwise, bias)
        return x

class AdaConv2d(nn.Module):
    def __init__(self, in_channels, c_out = None, kernel_size=3, n_groups=None, norm = True):
        super().__init__()
        self.n_groups = in_channels if n_groups is None else in_channels//n_groups
        self.in_channels = in_channels
        self.out_channels = c_out if not c_out is None else in_channels
        self.norm = norm

        padding = (kernel_size - 1) / 2

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(ceil(padding), floor(padding)),
                              padding_mode='reflect')


    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        if self.norm:
            x = F.instance_norm(x)

        # F.conv2d does not work with batched filters (as far as I can tell)...
        # Hack for inputs with > 1 sample
        ys = []
        for i in range(len(x)):
            y = self._forward_single(x[i:i + 1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0)

        ys = self.conv(ys)
        return ys

    def _forward_single(self, x, w_spatial, w_pointwise, bias):
        # Only square kernels
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1) / 2
        pad = (ceil(padding), floor(padding), ceil(padding), floor(padding))

        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, bias=bias)
        return x
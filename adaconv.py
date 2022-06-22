import torch
from torch import nn
import typing
import torch.nn.functional as F
from losses import calc_mean_std
import math


class AdaConv(nn.Module):
    def __init__(self, c_in:int, n_g_denominator:int, batch_size:int = 8, s_d: int = 512, norm:bool=True, c_out=None, kernel_size=4):
        super(AdaConv, self).__init__()
        self.n_groups = (c_in//n_g_denominator)
        self.c_out = c_out if not c_out is None else c_in
        self.batch_groups = batch_size *(c_in // n_g_denominator)
        self.out_groups = batch_size * (self.c_out // n_g_denominator)
        self.c_in = c_in
        self.s_d = s_d
        self.kernel_size = kernel_size
        if (kernel_size-1) % 2 == 0:
            if kernel_size == 1:
                self.pad = nn.Identity()
                kernel_size = 3
            else:
                padding = int((kernel_size - 1) / 2)
                padding = (padding,)*4
                self.pad = nn.ReflectionPad2d(padding)
        else:
            tl = math.ceil((kernel_size - 1) / 2)
            br = math.floor((kernel_size - 1) / 2)
            padding = (tl, br, tl, br)
            self.pad = nn.ReflectionPad2d(padding)
        self.norm = F.instance_norm if norm else nn.Identity()
        self.depthwise_kernel_conv = nn.Sequential(
            self.pad,
            nn.Conv2d(self.s_d,
                   self.c_in * (self.c_in//self.n_groups),
                   kernel_size=kernel_size,))

        self.pointwise_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pw_cn_kn = nn.Conv2d(self.s_d, self.c_in * self.c_out, kernel_size=1)
        self.pw_cn_bias = nn.Conv2d(self.s_d, self.c_out, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
            m.requires_grad=True

    def forward(self, style_encoding: torch.Tensor, predicted: torch.Tensor):
        N = style_encoding.shape[0]
        depthwise = self.depthwise_kernel_conv(style_encoding)
        depthwise = depthwise.view(N*self.c_in, self.c_in // self.n_groups, self.kernel_size, self.kernel_size)
        s_d = self.pointwise_avg_pool(style_encoding)
        pointwise_kn = self.pw_cn_kn(s_d).view(N*self.c_out, self.c_in, 1, 1)
        pointwise_bias = self.pw_cn_bias(s_d).view(N*self.c_out)

        a, b, c, d = predicted.size()
        #predicted = self.project_in(predicted)
        if self.norm:
            #predicted = F.instance_norm(predicted)
            predicted = predicted * torch.rsqrt(torch.mean(predicted ** 2, dim=1, keepdim=True) + 1e-8)
        predicted = predicted.view(1,a*b,c,d)
        content_out = nn.functional.conv2d(self.pad(predicted),
                                     weight=depthwise,
                                     stride=1,
                                     groups=self.batch_groups
                                     )
        content_out = nn.functional.conv2d(content_out,stride=1,
                weight=pointwise_kn,
                bias=pointwise_bias,
                groups = N)
        content_out = content_out.permute([1, 0, 2, 3]).view(a, self.c_out, c, d)

        return content_out
import torch
from torch import nn
import typing
import torch.nn.functional as F
from losses import calc_mean_std


class AdaConv(nn.Module):
    def __init__(self, c_in:int, p:int, s_d: int = 512, norm:bool=True):
        super(AdaConv, self).__init__()
        self.n_groups = c_in//p
        self.pointwise_groups = s_d//p
        self.c_out = c_in
        self.c_in = c_in
        self.style_groups = (s_d//p)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.norm = norm
        self.depthwise_kernel_conv = nn.Conv2d(s_d, self.c_out * (self.c_in//self.n_groups), kernel_size=2,stride=2,padding=1,padding_mode='reflect')

        self.pointwise_avg_pool = nn.Sequential(
            nn.AvgPool2d(2,2),
            nn.AvgPool2d(2,2))
        self.pw_cn_kn = nn.Conv2d(s_d, self.c_out*(self.c_out//self.n_groups), kernel_size=1)
        self.pw_cn_bias = nn.Conv2d(s_d, self.c_out, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.01)
            m.requires_grad=True

    def forward(self, style_encoding: torch.Tensor, predicted: torch.Tensor):
        N = style_encoding.shape[0]
        depthwise = self.depthwise_kernel_conv(style_encoding)
        depthwise = depthwise.view(N*self.c_out, self.c_in // self.n_groups, 3, 3)
        s_d = self.pointwise_avg_pool(style_encoding)
        pointwise_kn = self.pw_cn_kn(s_d).view(N*self.c_out, self.c_out // self.n_groups, 1, 1)
        pointwise_bias = self.pw_cn_bias(s_d).view(N*self.c_out)

        a, b, c, d = predicted.size()
        if self.norm:
            mean,std = calc_mean_std(predicted)
            predicted = predicted - mean
            predicted = predicted / std

        predicted = predicted.view(1,a*b,c,d)
        content_out = nn.functional.conv2d(
                nn.functional.conv2d(self.pad(predicted),
                                     weight=depthwise,
                                     stride=1,
                                     groups=self.n_groups*a
                                     ),
                stride=1,
                weight=pointwise_kn,
                bias=pointwise_bias,
                groups=self.n_groups*a)
        content_out = content_out.permute([1, 0, 2, 3]).view(a,b,c,d)
        return content_out
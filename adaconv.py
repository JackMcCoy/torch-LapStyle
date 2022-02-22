import torch
from torch import nn
import typing
import torch.nn.functional as F
from losses import calc_mean_std


class AdaConv(nn.Module):
    def __init__(self, c_in:int, p:int, batch_size:int = 8, s_d: int = 512, norm:bool=True, c_out=None, kernel_size=5):
        super(AdaConv, self).__init__()
        self.n_groups = (c_in//p)
        self.batch_groups = batch_size * (c_in // p)
        self.pointwise_groups = s_d//p
        self.c_out = c_out if not c_out is None else c_in
        self.c_in = c_in
        self.style_groups = (s_d//p)
        self.pad = nn.ReflectionPad2d((2, 2, 2, 2))
        self.norm = norm
        self.depthwise_kernel_conv = nn.Sequential(
            nn.Conv2d(s_d,s_d,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(s_d, self.c_out * (self.c_in//self.n_groups), kernel_size=3, padding=1 if kernel_size==5 else 0, padding_mode='reflect'),
            nn.LeakyReLU())

        self.pointwise_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))
        self.pw_cn_kn = nn.Sequential(
            nn.Conv2d(s_d, s_d, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(s_d, self.c_out*(self.c_out//self.n_groups), kernel_size=1),
            nn.LeakyReLU())
        self.pw_cn_bias = nn.Sequential(
            nn.Conv2d(s_d, s_d, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(s_d, self.c_out, kernel_size=1),
            nn.LeakyReLU())
        self.relu=nn.LeakyReLU()
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
        depthwise = depthwise.view(N*self.c_out, self.c_in // self.n_groups, 5, 5)
        s_d = self.pointwise_avg_pool(style_encoding)
        pointwise_kn = self.pw_cn_kn(s_d).view(N*self.c_out, self.c_out // self.n_groups, 1, 1)
        pointwise_bias = self.pw_cn_bias(s_d).view(N*self.c_out)

        a, b, c, d = predicted.size()
        if self.norm:
            predicted = F.instance_norm(predicted)

        predicted = predicted.view(1,a*b,c,d)
        content_out = nn.functional.conv2d(
                nn.functional.conv2d(self.pad(predicted),
                                     weight=depthwise,
                                     stride=1,
                                     groups=self.batch_groups
                                     ),
                stride=1,
                weight=pointwise_kn,
                bias=pointwise_bias,
                groups=self.batch_groups)
        content_out = content_out.permute([1, 0, 2, 3]).view(a,b,c,d)
        return content_out
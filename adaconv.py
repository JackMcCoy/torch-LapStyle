import torch
from torch import nn
import typing
import torch.nn.functional as F
from losses import calc_mean_std


class AdaConv(nn.Module):
    def __init__(self, c_in:int, p:int, batch_size:int = 8, s_d: int = 512, norm:bool=True, c_out=None, kernel_size=5):
        super(AdaConv, self).__init__()
        self.c_out = c_out if not c_out is None else c_in
        self.c_in = c_in
        self.n_groups = (c_out // p)
        self.kernel_size = kernel_size
        self.batch_groups = batch_size * (c_in // p)
        self.pointwise_groups = s_d // p
        self.style_groups = (s_d//p)
        pad = 2 if kernel_size==5 else 1
        self.pad = nn.ReflectionPad2d((pad, pad, pad, pad))
        self.norm = norm
        self.depthwise_kernel_conv = nn.Sequential(
            nn.Conv2d(s_d,self.c_in,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.c_in, self.c_out * (self.c_in//self.n_groups), kernel_size=3, padding=1 if kernel_size==5 else 0, padding_mode='reflect'))

        self.pointwise_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))
        print(s_d)
        print(self.c_in)
        print(self.c_out)
        print(self.n_groups)
        print('____________')
        self.pw_cn_kn = nn.Sequential(
            nn.Conv2d(s_d, self.c_in, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.c_in, self.c_out*(self.c_out//self.n_groups), kernel_size=1))
        self.pw_cn_bias = nn.Sequential(
            nn.Conv2d(s_d, self.c_in, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.c_in, self.c_out, kernel_size=1))
        self.relu=nn.ReLU()
        #self.project_in = nn.Sequential(
        #    nn.Conv2d(self.c_in, self.c_in, kernel_size=1),
        #    nn.LeakyReLU()
        #)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.01)
            m.requires_grad=True

    def forward(self, style_encoding: torch.Tensor, predicted: torch.Tensor):
        N = style_encoding.shape[0]
        print(style_encoding.shape)
        depthwise = self.depthwise_kernel_conv(style_encoding)
        print(depthwise.shape)
        depthwise = depthwise.view(N*self.c_out, self.c_in // self.n_groups, self.kernel_size, self.kernel_size)
        print(depthwise.shape)
        s_d = self.pointwise_avg_pool(style_encoding)
        print(s_d.shape)
        pointwise_kn = self.pw_cn_kn(s_d).view(N*self.c_out, self.c_out // self.n_groups, 1, 1)
        pointwise_bias = self.pw_cn_bias(s_d).view(N*self.c_out)

        a, b, c, d = predicted.size()
        #predicted = self.project_in(predicted)
        if self.norm:
            #predicted = F.instance_norm(predicted)
            predicted = predicted * torch.rsqrt(torch.mean(predicted ** 2, dim=1, keepdim=True) + 1e-8)

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
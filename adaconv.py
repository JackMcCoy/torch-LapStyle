import torch
from torch import nn
import typing
from losses import calc_mean_std

class AdaConv(nn.Module):
    def __init__(self, c_in:int, p:int, s_d: int = 512, norm:bool=True):
        super(AdaConv, self).__init__()
        self.n_groups = int(c_in//p)
        self.pointwise_groups = s_d//p
        self.c_out = c_in
        self.c_in = c_in
        self.style_groups = (s_d//p)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.norm = norm
        self.depthwise_kernel_conv = nn.Sequential(
            nn.Conv2d(s_d, self.c_out * (self.c_in//self.n_groups), kernel_size=2),
            nn.Unflatten(1,(self.c_out, self.c_in // self.n_groups))
        )
        self.pointwise_avg_pool = nn.Sequential(
            nn.AvgPool2d(2,2),
            nn.AvgPool2d(2,2))
        self.pw_cn_kn = nn.Sequential(
            nn.Conv2d(s_d, self.c_out*(self.c_out//self.n_groups), kernel_size=1),
            nn.Unflatten(1,(self.c_out, self.c_out//self.n_groups))
        )
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
        s_d = self.pointwise_avg_pool(style_encoding)
        pointwise_kn = self.pw_cn_kn(s_d)
        pointwise_bias = self.pw_cn_bias(s_d).view(N,self.c_out)

        a, b, c, d = predicted.size()
        if self.norm:
            mean = predicted.mean(dim=(2, 3), keepdim=True)
            predicted = predicted - mean
            predicted = predicted * torch.rsqrt(predicted.square().mean(dim=(2, 3), keepdim=True) + 1e-5)
        content_out = torch.empty_like(predicted)
        for i in range(a):
            content_out[i] = nn.functional.conv2d(
                nn.functional.conv2d(self.pad(predicted[i].unsqueeze(0)),
                                             weight=depthwise[i],
                                             stride=1,
                                             groups=self.n_groups
                                             ),
                                 stride = 1,
                                 weight=pointwise_kn[i],
                                 bias=pointwise_bias[i],
                                 groups=self.n_groups).squeeze()
        return content_out
import torch
from torch import nn
from function import calc_mean_std
import typing

class AdaConv(nn.Module):
    def __init__(self, c_in:int, p:int, batch_size: typing.Optional[int], s_d: int = 512):
        super(AdaConv, self).__init__()
        self.n_groups = int(c_in//p)
        self.pointwise_groups = s_d//p
        self.c_out = c_in
        self.c_in = c_in
        self.style_groups = (s_d//p)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.depthwise_kernel_conv = nn.Conv2d(s_d, self.c_out * (self.c_in//self.n_groups), kernel_size=2)
        self.pointwise_avg_pool = nn.AvgPool2d(4)
        self.pw_cn_kn = nn.Conv2d(s_d, self.c_out*(self.c_out//self.n_groups), kernel_size=1)
        self.pw_cn_bias = nn.Conv2d(s_d, self.c_out, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 1e-9)
            m.requires_grad=True
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 1-e9)

    def forward(self, style_encoding: torch.Tensor, predicted: torch.Tensor, norm: bool):
        depthwise = self.depthwise_kernel_conv(style_encoding)
        epthwise = depthwise.view(N, self.c_out, self.c_in // self.n_groups, 3, 3)
        s_d = self.pointwise_avg_pool(style_encoding)
        pointwise_kn = self.pw_cn_kn(s_d).view(N, self.c_out, self.c_out//self.n_groups, 1, 1)
        pointwise_bias = self.pw_cn_bias(s_d).view(N,self.c_out)

        a, b, c, d = predicted.size()
        if norm:
            content_mean, content_std = calc_mean_std(predicted)
            content_mean = content_mean.view(a, 1, 1, 1).expand(a,b,c,d)
            content_std = content_std.view(a, 1, 1, 1).expand(a,b,c,d)
            predicted = (predicted - content_mean) / content_std

        for i in range(a):
            predicted[i] = nn.functional.conv2d(
                nn.functional.conv2d(self.pad(predicted[i].unsqueeze(0)),
                                             weight=depthwise[i],
                                             stride=1,
                                             groups=self.n_groups
                                             ),
                                 stride = 1,
                                 weight=pointwise_kn[i],
                                 bias=pointwise_bias[i],
                                 groups=self.n_groups).squeeze()
        return predicted
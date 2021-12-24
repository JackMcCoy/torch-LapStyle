import torch
from torch import nn
from function import calc_mean_std

class AdaConv(nn.Module):
    def __init__(self, c_in, p, batch_size, s_d = 512):
        super(AdaConv, self).__init__()
        self.n_groups = c_in//p
        self.pointwise_groups = s_d//p
        self.c_out = c_in
        self.c_in = c_in
        self.style_groups = (s_d//p)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.depthwise_kernel_conv = nn.Conv2d(s_d, self.c_out * (self.c_in//self.n_groups), 3)
        self.pointwise_avg_pool = nn.AvgPool2d(3)
        self.pw_cn_kn = nn.Conv2d(s_d, self.c_out*(self.c_out//self.n_groups), 1)
        self.pw_cn_bias = nn.Conv2d(s_d, self.c_out, 1)
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

    def forward(self, style_encoding, predicted, norm):
        N, ch, h, w = predicted.shape
        conv_out = []
        depthwise = self.depthwise_kernel_conv(style_encoding)
        depthwise = depthwise.view(N,self.c_out, self.c_in//self.n_groups, 3, 3)
        s_d = self.pointwise_avg_pool(style_encoding)
        pointwise_kn = self.pw_cn_kn(s_d)
        pointwise_kn = pointwise_kn.view(N, self.c_out, self.c_out//self.n_groups, 1, 1)
        pointwise_bias = self.pw_cn_bias(s_d).view(N,self.c_out)
        if norm:
            content_mean, content_std = calc_mean_std(predicted)
            content_mean = content_mean.view(N, 1, 1, 1).expand(N, ch, h, w)
            content_std = content_std.view(N, 1, 1, 1).expand(N, ch, h, w)
            predicted = (predicted - content_mean) / content_std

        depth = nn.functional.conv2d(self.pad(predicted),
                                         weight=depthwise,
                                         stride = 1,
                                         groups=self.n_groups)
        conv_out =nn.functional.conv2d(depth, weight=pointwise_kn,
                                         bias=pointwise_bias,
                                         stride=1,
                                         groups=self.n_groups)
        return conv_out
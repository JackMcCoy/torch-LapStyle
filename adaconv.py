import torch
from torch import nn

class KernelPredictor(nn.Module):
    def __init__(self, c_in, c_out, p):
        super(KernelPredictor, self).__init__()
        self.n_groups = c_in/(c_in/p)
        self.pointwise_groups = c_out/(c_in/p)
        self.c_out = c_out
        self.depthwise_kernel_conv = nn.Conv2d(512, self.n_groups, 2)
        self.pointwise_avg_pool = nn.AvgPool2d(4)
        self.pw_cn_kn = nn.Conv2d(512, self.n_groups, 1)
        self.pw_cn_bias = nn.Conv2d(512, c_out, 1)

    def forward(self, style_encoding):
        N = style_encoding.shape[0]
        depthwise = self.depthwise_kernel_conv(style_encoding).unsqueeze(1).expand(N,self.c_out, self.n_groups, 3, 3)
        s_d = self.pointwise_avg_pool(style_encoding)
        pointwise_1_kn = self.pw_cn_kn(s_d)
        pointwise_bias = self.pw_cn_bias(s_d).squeeze()
        return (depthwise, pointwise_1_kn, pointwise_bias)
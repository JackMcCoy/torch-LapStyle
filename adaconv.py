import torch
from torch import nn
from function import calc_mean_std

class AdaConv(nn.Module):
    def __init__(self, ch_in, p):
        super(AdaConv, self).__init__()
        self.kernel_predictor = KernelPredictor(ch_in, ch_in, p)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, style_encoding, style_in, content_in):
        depthwise, pointwise_kn, pointwise_bias = self.kernel_predictor(style_encoding)
        spatial_conv_out = []
        N = style_encoding.shape[0]
        size = content_in.size()
        content_mean, content_std = calc_mean_std(content_in)
        normalized_feat = (content_in - content_mean.expand(
            size)) / content_std.expand(size)
        predicted = self.pad(normalized_feat)
        for i in range(N):
            spatial_conv_out.append(nn.functional.conv2d(predicted[i,:,:,:].unsqueeze(0),
                                       weight = depthwise[i]*pointwise_kn[i],
                                       bias = pointwise_bias[i],
                                       groups = self.kernel_predictor.n_groups))
        predicted = torch.cat(spatial_conv_out,0)
        return normalized_feat * predicted

class KernelPredictor(nn.Module):
    def __init__(self, c_in, c_out, p):
        super(KernelPredictor, self).__init__()
        self.n_groups = c_in//p
        self.pointwise_groups = c_out//p
        self.c_out = c_out
        self.c_in = c_in
        self.depthwise_kernel_conv = nn.Sequential(
            nn.Conv2d(512, self.c_in//self.n_groups, 2),
            nn.Sigmoid())
        self.pointwise_avg_pool = nn.AvgPool2d(4)
        self.pw_cn_kn = nn.Sequential(
            nn.Conv2d(512, self.c_out//self.pointwise_groups, 1),
            nn.Sigmoid())
        self.pw_cn_bias = nn.Sequential(
            nn.Conv2d(512, c_out, 1),
            nn.Sigmoid())

    def forward(self, style_encoding):
        N = style_encoding.shape[0]
        depthwise = self.depthwise_kernel_conv(style_encoding).unsqueeze(1).expand(N,self.c_out, self.c_in//self.n_groups, 3, 3)
        s_d = self.pointwise_avg_pool(style_encoding)
        pointwise_1_kn = self.pw_cn_kn(s_d).unsqueeze(1).expand(N, self.c_out, self.c_out//self.pointwise_groups, 1, 1)
        pointwise_bias = self.pw_cn_bias(s_d).squeeze()
        return depthwise, pointwise_1_kn, pointwise_bias
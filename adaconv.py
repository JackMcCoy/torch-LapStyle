import torch
from torch import nn

class AdaConv(nn.Module):
    def __init__(self, ch_in, p):
        super(AdaConv, self).__init__()
        self.kernel_predictor = KernelPredictor(ch_in, ch_in, p)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, style_encoding, content_in):
        depthwise, pointwise_kn, pointwise_bias = self.kernel_predictor(style_encoding)
        spatial_conv_out = []
        N = style_encoding.shape[0]
        content_in = self.pad(content_in)
        for i in range(N):
            depth = nn.functional.conv2d(content_in[i,:,:,:].unsqueeze(0),
                                       weight = depthwise[i],
                                       groups = self.kernel_predictor.n_groups)
            spatial_conv_out.append(nn.functional.conv2d(depth,
                                                         weight = pointwise_kn[i],
                                                         bias = pointwise_bias[i]))
        return torch.cat(spatial_conv_out,0)


class KernelPredictor(nn.Module):
    def __init__(self, c_in, c_out, p):
        super(KernelPredictor, self).__init__()
        self.n_groups = c_in//p
        self.pointwise_groups = c_out//p
        self.c_out = c_out
        self.c_in = c_in
        print(self.n_groups)
        self.depthwise_kernel_conv = nn.Conv2d(512, self.n_groups, 2)
        self.pointwise_avg_pool = nn.AvgPool2d(4)
        self.pw_cn_kn = nn.Conv2d(512, self.pointwise_groups, 1)
        self.pw_cn_bias = nn.Conv2d(512, c_out, 1)

    def forward(self, style_encoding):
        N = style_encoding.shape[0]
        depthwise = self.depthwise_kernel_conv(style_encoding).unsqueeze(1).expand(N,self.c_out, self.c_in//self.n_groups, 3, 3)
        s_d = self.pointwise_avg_pool(style_encoding)
        pointwise_1_kn = self.pw_cn_kn(s_d).unsqueeze(1).expand(N, self.c_out, self.c_out//self.pointwise_groups, 1, 1)
        pointwise_bias = self.pw_cn_bias(s_d).squeeze()
        return depthwise, pointwise_1_kn, pointwise_bias
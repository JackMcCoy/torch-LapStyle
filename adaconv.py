import torch
from torch import nn
from function import calc_mean_std

class AdaConv(nn.Module):
    def __init__(self, ch_in, p, s_d = 512):
        super(AdaConv, self).__init__()
        self.s_d = s_d
        self.kernel_predictor = KernelPredictor(ch_in, ch_in, p, s_d)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.n_groups = ch_in//p
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
            m.requires_grad = True
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, style_encoding, content_in, norm: bool=True):
        depthwise, pointwise_kn, pointwise_bias = self.kernel_predictor(style_encoding)
        spatial_conv_out = []
        N = style_encoding.shape[0]
        if norm:
            size = content_in.size()
            content_mean, content_std = calc_mean_std(content_in)

            content_in = (content_in - content_mean.expand(
                    size)) / content_std.expand(size)
        predicted = self.pad(content_in)
        predicted = torch.vsplit(predicted, N)
        depthwise = torch.unbind(depthwise, 0)
        pointwise_kn = torch.unbind(pointwise_kn, 0)
        pointwise_bias = torch.unbind(pointwise_bias, 0)

        for a,b,c,d in zip(predicted, depthwise, pointwise_kn, pointwise_bias):

            depth = nn.functional.conv2d(a,
                                         weight=b,
                                         groups=self.n_groups)
            spatial_conv_out.append(nn.functional.conv2d(depth,
                                                         weight=c,
                                                         bias=d,
                                                         groups=self.n_groups))
        predicted = torch.stack(spatial_conv_out,0)
        return predicted

class KernelPredictor(nn.Module):
    def __init__(self, c_in, c_out, p, s_d):
        super(KernelPredictor, self).__init__()
        self.n_groups = c_in//p
        self.pointwise_groups = s_d//p
        self.c_out = c_out
        self.c_in = c_in
        self.style_groups = (s_d//p)
        self.depthwise_kernel_conv = nn.Conv2d(s_d, self.c_out * (self.c_in//self.n_groups), 2)
        self.pointwise_avg_pool = nn.AvgPool2d(4)
        self.pw_cn_kn = nn.Conv2d(s_d, self.c_out*(self.c_out//self.n_groups), 1)
        self.pw_cn_bias = nn.Conv2d(s_d, c_out, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
            m.requires_grad=True
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, style_encoding):
        N = style_encoding.shape[0]

        depthwise = self.depthwise_kernel_conv(style_encoding)
        depthwise = depthwise.reshape((N,self.c_out, self.c_in//self.n_groups, 3, 3))
        s_d = self.pointwise_avg_pool(style_encoding)
        pointwise_1_kn = self.pw_cn_kn(s_d)
        pointwise_1_kn = pointwise_1_kn.reshape((N, self.c_out, self.c_out//self.n_groups, 1, 1))
        pointwise_bias = self.pw_cn_bias(s_d)
        pointwise_bias = pointwise_bias.squeeze()
        return depthwise, pointwise_1_kn, pointwise_bias
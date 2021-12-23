import torch
from torch import nn
from function import calc_mean_std

class AdaConv(nn.Module):
    def __init__(self, ch_in, p, batch_size, s_d = 512):
        super(AdaConv, self).__init__()
        self.s_d = s_d
        self.kernel_predictor = KernelPredictor(ch_in, ch_in, p, s_d, batch_size)
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

    def forward(self, style_encoding, predicted, norm: bool=True):
        conv_out = []
        depthwise, pointwise_kn, pointwise_bias = self.kernel_predictor(style_encoding)
        N, ch, h, w = predicted.shape
        if norm:
            content_mean, content_std = calc_mean_std(predicted)
            content_mean = content_mean.expand(N, ch, h, w)
            content_std = content_std.expand(N, ch, h, w)
            predicted = (predicted - content_mean) / content_std
        predicted = predicted.view(N,1,ch,h,w)

        for idx in range(N):
            depth = nn.functional.conv2d(self.pad(predicted[idx]),
                                         weight=depthwise[idx],
                                         groups=self.n_groups)
            conv_out.append(nn.functional.conv2d(depth,
                                                         weight=pointwise_kn[idx],
                                                         bias=pointwise_bias[idx],
                                                         groups=self.n_groups))
        conv_out = torch.cat(conv_out,0)
        return conv_out

class KernelPredictor(nn.Module):
    def __init__(self, c_in, c_out, p, s_d, batch_size):
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
        self.depthwise = nn.Parameter(torch.zeros(batch_size, self.c_out, self.c_in//self.n_groups, 3, 3))
        self.pw_kn = nn.Parameter(torch.zeros(batch_size, self.c_out, self.c_out//self.n_groups, 1, 1))
        self.pw_bias =  nn.Parameter(torch.zeros(batch_size,c_out))
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
        depthwise = depthwise.view(N,self.c_out, self.c_in//self.n_groups, 3, 3)
        s_d = self.pointwise_avg_pool(style_encoding)
        pointwise_1_kn = self.pw_cn_kn(s_d)
        pointwise_1_kn = pointwise_1_kn.view(N, self.c_out, self.c_out//self.n_groups, 1, 1)
        pointwise_bias = self.pw_cn_bias(s_d)
        pointwise_bias = pointwise_bias.squeeze()
        with torch.no_grad():
            self.depthwise.copy_(depthwise)
            self.pw_kn.copy_(pointwise_1_kn)
            self.pw_bias.copy_(pointwise_bias)
        return self.depthwise, self.pw_kn, self.pw_bias
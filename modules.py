import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch
import typing
from function import normalized_feat, calc_mean_std


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(dim, dim, kernel_size=3),
                                        nn.ReLU(),
                                        nn.Conv2d(dim, dim, kernel_size=1))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 1e-9)
            m.requires_grad = True
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 1e-9)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class RiemannNoise(nn.Module):

    def __init__(self, size:int, channels:int):
        super(RiemannNoise, self).__init__()
        self.size = size
        self.spatial_params = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.ones(size, size))),
                                        nn.Parameter(nn.init.normal_(torch.ones(size, size))),
                                        nn.Parameter(nn.init.constant_(torch.ones(1, ), .5)),
                                        nn.Parameter(nn.init.constant_(torch.ones(1, ), .5))])
        self.noise = torch.zeros(1,device=torch.device('cuda:0'))
        self.size=size
        self.relu = nn.ReLU()


    def set_random(self):
        self.noise = self.zero_holder.normal_()

    def forward(self, x):
        #self.cuda_states = torch.utils.checkpoint.get_device_states(x)
        N, c, h, w = x.shape
        A, b, alpha, r = self.spatial_params


        s = torch.sum(self.relu(-x), dim=1, keepdim=True)
        s = s - s.mean(dim=(2, 3), keepdim=True)
        s_max = torch.abs(s).amax(dim=(2, 3), keepdim=True)
        s = s / (s_max + 1e-8)
        s = s * A + b
        s = torch.tile(s, (1, c, 1, 1))
        sp_att_mask = (1 - alpha) + alpha * s
        sp_att_mask = sp_att_mask / (torch.linalg.norm(sp_att_mask, dim=0, keepdims=True) + 1e-8)

        x = r*sp_att_mask * x + r * sp_att_mask * (self.noise.repeat(N,c,h,h).normal_())
        return x


class SpectralResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel,padding, downsample=False):
        super(SpectralResBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size = kernel,padding=padding,padding_mode='reflect')
        self.relu = nn.LeakyReLU(0.2)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size = kernel,padding=padding,padding_mode='reflect')
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, kernel_size= 1, stride = 1, padding = 0)
        else:
            self.c_sc = nn.Identity()

    def init_spectral_norm(self):
        self.conv_1 = spectral_norm(self.conv_1)
        self.conv_2 = spectral_norm(self.conv_2)

    def forward(self, in_feat):
        x = self.conv_1(in_feat)
        x = self.relu(x)
        x = self.conv_2(x)
        if self.downsample:
            x = nn.functional.avg_pool2d(x, 2)
        if self.learnable_sc:
            x2 = self.c_sc(in_feat)
            if self.downsample:
                x2 = nn.functional.avg_pool2d(x2, 2)
        else:
            x2=0
        out = x+x2
        return out


class Bias(nn.Module):
    def __init__(self, channels):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(nn.init.normal_(torch.ones(channels,1,1),.5))
    def forward(self, x):
        x = x+self.bias
        return x

class ConvBlock(nn.Module):

    def __init__(self, dim1, dim2,noise=0):
        super(ConvBlock, self).__init__()
        layers = [nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(dim1, dim2, kernel_size=3, bias=False)]
        if noise>0:
            layers.append(RiemannNoise(noise, dim2))
        layers.extend([Bias(dim2),nn.ReLU()])
        self.conv_block = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            m.requires_grad = True
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 1e-9)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class Attention(nn.Module):
    def __init__(self, num_features):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.key_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.value_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.uniform_(self.query_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.uniform_(self.key_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.value_conv.weight)
        nn.init.uniform_(self.value_conv.bias, 0.0, 1.0)

    def forward(self, content_feat, style_feat):
        Query = self.query_conv(normalized_feat(content_feat))
        Key = self.key_conv(normalized_feat(style_feat))
        Value = self.value_conv(style_feat)
        batch_size, channels, height_c, width_c = Query.size()
        Query = Query.view(batch_size, -1, width_c * height_c).permute(0, 2, 1)
        batch_size, channels, height_s, width_s = Key.size()
        Key = Key.view(batch_size, -1, width_s * height_s)
        Attention_Weights = self.softmax(torch.bmm(Query, Key))

        Value = Value.view(batch_size, -1, width_s * height_s)
        Output = torch.bmm(Value, Attention_Weights.permute(0, 2, 1))
        Output = Output.view(batch_size, channels, height_c, width_c)
        return Output


class SAFIN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.shared_weight = nn.Parameter(torch.Tensor(num_features), requires_grad=True)
        self.shared_bias = nn.Parameter(torch.Tensor(num_features), requires_grad=True)
        self.shared_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.gamma_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.beta_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.attention = Attention(num_features)
        self.relu = nn.ReLU()
        nn.init.ones_(self.shared_weight)
        nn.init.zeros_(self.shared_bias)
        nn.init.xavier_uniform_(self.gamma_conv.weight)
        nn.init.uniform_(self.gamma_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.beta_conv.weight)
        nn.init.uniform_(self.beta_conv.bias, 0.0, 1.0)

    def forward(self, content_feat, style_feat, output_shared=False):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_feat = self.attention(content_feat, style_feat)
        style_gamma = self.relu(self.gamma_conv(style_feat))
        style_beta = self.relu(self.beta_conv(style_feat))
        content_mean, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        shared_affine_feat = normalized_feat * self.shared_weight.view(1, self.num_features, 1, 1).expand(size) + \
                             self.shared_bias.view(1, self.num_features, 1, 1).expand(size)
        if output_shared:
            return shared_affine_feat
        output = shared_affine_feat * style_gamma + style_beta
        return output

def get_wav(in_channels, pool=True):
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool: net = nn.Conv2d
    else: net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def reshape(self, x, y):
        assert len(x.shape)==len(y.shape)==4
        y = y[:, :, :x.shape[2], :x.shape[3]]
        return y

    def forward(self, ll, lh, hl, hh):
        lh = self.reshape(ll, lh); hl = self.reshape(ll, hl); hh = self.reshape(ll, hh)
        return self.LL(ll) + self.LH(lh) + self.HL(hl) + self.HH(hh)
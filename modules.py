import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch
import typing
from function import normalized_feat
from adaconv import AdaConv


class ThumbInstanceNorm(nn.Module):
    def __init__(self, out_channels=None, affine=True):
        super(ThumbInstanceNorm, self).__init__()
        self.thumb_mean = None
        self.thumb_std = None
        self.collection = True
        if affine == True:
            self.weight = nn.Parameter(torch.ones(size=(1, out_channels, 1, 1), requires_grad=True))
            self.bias = nn.Parameter(torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True))

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, x, thumb=None):
        if self.training:
            thumb_mean, thumb_std = self.calc_mean_std(thumb)
            x = (x - thumb_mean) / thumb_std * self.weight + self.bias
            return x
        else:
            if self.collection:
                thumb_mean, thumb_std = self.calc_mean_std(x)
                self.thumb_mean = thumb_mean
                self.thumb_std = thumb_std
            x = (x - self.thumb_mean) / self.thumb_std * self.weight + self.bias
            return x


class ResBlock(nn.Module):
    def __init__(self, dim, hw=0, noise=False):
        super(ResBlock, self).__init__()
        modules = [nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(dim, dim, kernel_size=3)]
        #if noise == True:
        #    modules.append(RiemannNoise(hw))
        self.conv_block = nn.Sequential(*modules,
                                        nn.LeakyReLU(),
                                        nn.Conv2d(dim, dim, kernel_size=1))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            if m.kernel_size==3:
                nn.init.kaiming_normal_(m.weight.data, .01)
            else:
                nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
            m.requires_grad = True

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class FusedConvNoiseBias(nn.Module):
    def __init__(self, ch_in, ch_out, hw, scale_change, noise = True):
        super(FusedConvNoiseBias, self).__init__()
        self.resize = nn.Identity()
        self.noise = nn.Identity()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        if scale_change == 'up':
            self.resize = nn.Upsample(scale_factor=2, mode='nearest')
        elif scale_change == 'down':
            self.resize = nn.AvgPool2d(2, stride=2)
        if noise==True:
            self.noise = RiemannNoise(hw)
        self.bias = nn.Parameter(nn.init.constant_(torch.ones(ch_out, ), .01))
        self.act = nn.LeakyReLU()
        self.res_scale = torch.rsqrt((torch.ones(1,device=device)*2))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, a = .01)
            m.requires_grad = True
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias,0)

    def forward(self, x):
        x = self.resize(x)
        out = self.conv(x)
        out = self.noise(out)
        out = out + self.bias.view(1,-1,1,1)
        out = self.act(out)
        if self.ch_in == self.ch_out:
            out = (x + out) * self.res_scale
        return out


class RiemannNoise(nn.Module):

    def __init__(self, size:int, device):
        super(RiemannNoise, self).__init__()
        self.size = size
        self.spatial_params = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.ones(size, size))),
                                        nn.Parameter(nn.init.normal_(torch.ones(size, size))),
                                        nn.Parameter(nn.init.constant_(torch.ones(1, ), .5)),
                                        nn.Parameter(nn.init.constant_(torch.ones(1, ), .5))])
        self.noise = torch.zeros(1, device=device)
        self.noise.requires_grad = False
        self.size=size
        self.relu = nn.ReLU()


    def set_random(self):
        self.noise = self.zero_holder.normal_()

    def forward(self, x):
        #self.cuda_states = torch.utils.checkpoint.get_device_states(x)
        N, c, h, w = x.shape
        A, b, alpha, r = self.spatial_params


        s = torch.sum(x, dim=1, keepdim=True)
        s = s - s.mean(dim=(2, 3), keepdim=True)
        s_max = s.abs().amax(dim=(2, 3), keepdim=True)
        s = s / (s_max + 1e-8)
        s = (s + 1) / 2
        s = s * A + b
        s = torch.tile(s, (1, c, 1, 1))
        sp_att_mask = (1 - alpha) + alpha * s
        sp_att_mask = sp_att_mask / (torch.linalg.norm(sp_att_mask, dim=1, keepdims=True) + 1e-8)

        x = r*sp_att_mask * x + r * sp_att_mask * (self.noise.repeat(N,c,h,h).normal_())
        return x


class SpectralResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel,padding, downsample=False):
        super(SpectralResBlock, self).__init__()
        self.conv_1 = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size = kernel,padding=padding,padding_mode='reflect'))
        self.relu = nn.LeakyReLU()
        self.conv_2 = spectral_norm(nn.Conv2d(out_ch, out_ch, kernel_size = kernel,padding=padding,padding_mode='reflect'))
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        self.c_sc = nn.Conv2d(in_ch, out_ch, kernel_size= 1, stride = 1, padding = 0)

    def forward(self, in_feat):
        x = self.conv_1(in_feat)
        x = self.relu(x)
        x = self.conv_2(x)
        if self.downsample:
            x = nn.functional.avg_pool2d(x, 2)
        '''
        if self.downsample:
            x2 = self.c_sc(in_feat)
            x2 = nn.functional.avg_pool2d(x2, 2)
        else:
            x2 = 0
        '''
        return x


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
                  nn.Conv2d(dim1, dim2, kernel_size=3),
                  nn.ReLU()]

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

'''
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
'''

def PixelShuffleUp(channels):
    return nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                         nn.Conv2d(channels, channels*4, kernel_size=3),
                         nn.PixelShuffle(2),
                         nn.Conv2d(channels, channels, kernel_size=1)
                         )

def Downblock():
    return nn.Sequential(  # Downblock
        nn.Conv2d(6, 128, kernel_size=1),
        nn.LeakyReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, kernel_size=3, stride=1),
        nn.LeakyReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, kernel_size=3, stride=1),
        nn.LeakyReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, kernel_size=3, stride=2),
        nn.LeakyReLU(),
        # Resblock Middle
        ResBlock(64)
    )

def adaconvs(batch_size,s_d):
    return nn.ModuleList([
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(128, 2, batch_size, s_d=s_d),
            AdaConv(128, 2, batch_size, s_d=s_d)])


class StyleEncoderBlock(nn.Module):
    def __init__(self, ch):
        super(StyleEncoderBlock, self).__init__()
        self.net = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(ch, ch, kernel_size=3),
        nn.AvgPool2d(2, stride=2),
        nn.LeakyReLU())
    def forward(self, x):
        x = self.net(x)
        return x

def Upblock():
    return nn.ModuleList([nn.Sequential(nn.Conv2d(64, 256, kernel_size=1),
                                 nn.LeakyReLU(),
                                 nn.PixelShuffle(2),
                                 nn.Conv2d(64, 64, kernel_size=1),
                                 nn.ReLU(),
                                 nn.ReflectionPad2d((1, 1, 1, 1)),
                                 nn.Conv2d(64, 64, kernel_size=3),
                                 nn.LeakyReLU()),
                   nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                 nn.Conv2d(64, 128, kernel_size=3),
                                 nn.LeakyReLU()),
                   nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                 nn.Conv2d(128, 128, kernel_size=3),
                                 nn.LeakyReLU()),
                   nn.Sequential(nn.Conv2d(128, 3, kernel_size=1)
                                 )])


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
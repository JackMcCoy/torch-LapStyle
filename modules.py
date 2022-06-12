import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch
import typing
import math
from function import normalized_feat
from gaussian_diff import gaussian
from adaconv import AdaConv
import numpy as np
from torch.nn import functional as F
from fused_act import FusedLeakyReLU


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            N,C,h,w = inp.shape
            out = F.conv2d(self.pad(inp), weight=self.filt, stride=self.stride, groups=C)
            return out

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


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
        self.res_scale = torch.rsqrt((torch.ones(1,device='cuda:0')*2))
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


class GaussianNoise(nn.Module):

    def __init__(self):
        super(GaussianNoise, self).__init__()
        self.noise_strength = nn.Parameter(nn.init.constant_(torch.zeros(1,), 0))

    def forward(self, x):
        #self.cuda_states = torch.utils.checkpoint.get_device_states(x)
        noise = torch.empty_like(x,device='cuda').normal_()
        return x + self.noise_strength * noise


class RiemannNoise(nn.Module):

    def __init__(self, size:int):
        super(RiemannNoise, self).__init__()
        self.size = size
        self.spatial_params = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.ones(size, size))),
                                        nn.Parameter(nn.init.normal_(torch.ones(size, size))),
                                        nn.Parameter(nn.init.constant_(torch.ones(1, 1), .5)),
                                        nn.Parameter(nn.init.constant_(torch.ones(1, 1), .5))])
        self.noise = torch.zeros(1, device='cuda:0')
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
        s = s - s.mean(dim=(2, 3)).view(N,1,1,1)
        s_max = s.abs().amax(dim=(2, 3)).view(N,1,1,1)
        s = s / (s_max + 1e-8)
        s = (s + 1)/2
        s = A * s + b
        sp_att_mask = alpha + (1 - alpha) * s
        sp_att_mask = sp_att_mask * torch.rsqrt(torch.mean(torch.square(sp_att_mask), axis=[2, 3], keepdims=True) + 1e-8)
        x = r*sp_att_mask * x + r * sp_att_mask * (self.noise.repeat(N,c,h,h).normal_())
        return x


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class SpectralResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel,padding, downsample=False):
        super(SpectralResBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, padding_mode='reflect')
        self.relu = nn.LeakyReLU()
        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel, padding=padding, padding_mode='reflect')
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        self.c_sc = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def init_spectral_norm(self):
        self.conv_1 = spectral_norm(self.conv_1)
        self.conv_2 = spectral_norm(self.conv_2)

    def forward(self, in_feat):
        x = self.relu(in_feat)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        if self.downsample:
            x = nn.functional.avg_pool2d(x, 2)
        x2 = self.c_sc(in_feat)
        if self.downsample:
            x2 = nn.functional.avg_pool2d(x2, 2)
        else:
            x2=0
        x = x+x2
        return x


class Bias(nn.Module):
    def __init__(self, channels):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(nn.init.constant_(torch.ones(channels,1,1),0))
    def forward(self, x):
        x = x+self.bias
        return x

def ConvMixer(h, depth, kernel_size=9, patch_size=7):
    Seq, ActBn = nn.Sequential, lambda x: Seq(x, nn.GELU(), nn.BatchNorm2d(h))
    Residual = type('Residual', (Seq,), {'forward': lambda self, x: self[0](x) + x})
    return Seq(ActBn(nn.Conv2d(h, h, patch_size, stride=patch_size)),
    [Seq(Residual(ActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding="same"))),
    ActBn(nn.Conv2d(h, h, 1))) for i in range(depth)])


class ConvBlock(nn.Module):

    def __init__(self, dim1, dim2,kernel_size=3,padding=1,scale_change='', padding_mode='reflect', noise=False, nn_blur=False):
        super(ConvBlock, self).__init__()
        self.resize=nn.Identity()
        self.skip = nn.Identity()
        self.blurpool = nn.Identity()
        if scale_change == 'up':
            self.blurpool = BlurPool(dim2, pad_type='reflect', filt_size=4, stride=1, pad_off=0)
            if nn_blur:
                self.resize=StyleNERFUpsample(dim2)
            else:
                self.resize = nn.Upsample(scale_factor=2, mode='nearest')
        elif scale_change == 'down':
            self.blurpool = BlurPool(dim2, pad_type='reflect', filt_size=4, stride=2, pad_off=0)
            self.skip = nn.Sequential(
                nn.Conv2d(dim1, dim2, kernel_size=1, bias=not noise),
                #nn.Upsample(scale_factor=.5, mode='nearest')
            )
        if scale_change != 'down' and dim2 != dim1:
            self.skip = nn.Conv2d(dim1, dim2, kernel_size=1, bias=not noise)
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim1, dim2, kernel_size=kernel_size,padding=padding, padding_mode=padding_mode),
            #nn.GroupNorm(32,dim2),
            nn.LeakyReLU(),
            #nn.BatchNorm2d(dim2),
            nn.Conv2d(dim2, dim2, kernel_size = kernel_size, padding=padding, padding_mode=padding_mode, bias= not noise),
            self.blurpool
            )
        self.use_noise=noise
        if noise:
            self.noise = GaussianNoise()
            self.relu = FusedLeakyReLU(dim2)
        else:
            self.groupnorm = nn.GroupNorm(32,dim2)
            self.relu = nn.LeakyReLU()
        self.skip = nn.Sequential(self.skip,self.blurpool)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if not m.bias is None:
                nn.init.constant_(m.bias.data, 0.01)
            m.requires_grad = True
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.01)

    def forward(self, x):
        out = self.conv_block(x)
        skip = self.skip(x)
        if self.use_noise:
            out = self.resize(out+skip)
            out = self.noise(out)
            out = self.relu(out)
        else:
            out = self.groupnorm(out+skip)
            out = self.relu(out)
            out = self.resize(out)
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


class StyleNERFUpsample(nn.Module):
    def __init__(self, dim):
        super(StyleNERFUpsample, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(dim, dim * 8, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim*8, dim*4, kernel_size=1),
        )
        self.blurpool = BlurPool(dim,stride=1)
        self.conv = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,padding=1,padding_mode='reflect'),
            nn.LeakyReLU()
        )
        self.pad = nn.ReflectionPad2d((1,1,1,1))
    def forward(self, x):
        xa = self.adapter(x)
        x = torch.pixel_shuffle(x.repeat(1,4,1,1)+xa, 2)
        x = self.blurpool(x)
        x = self.conv(x)
        return x



class StyleEncoderBlock(nn.Module):
    def __init__(self, ch, kernel_size=3):
        super(StyleEncoderBlock, self).__init__()
        padding = kernel_size//2
        self.net = nn.Sequential(
        nn.Conv2d(ch, ch, kernel_size=kernel_size, padding=padding, padding_mode='reflect'),
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


def sobel(window_size):
    assert (window_size % 2 != 0)
    ind = window_size // 2
    matx = []
    maty = []
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gx_ij = 0
            else:
                gx_ij = i / float(i * i + j * j)
            row.append(gx_ij)
        matx.append(row)
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gy_ij = 0
            else:
                gy_ij = j / float(i * i + j * j)
            row.append(gy_ij)
        maty.append(row)

    # matx=[[-3, 0,+3],
    # 	  [-10, 0 ,+10],
    # 	  [-3, 0,+3]]
    # maty=[[-3, -10,-3],
    # 	  [0, 0 ,0],
    # 	  [3, 10,3]]
    if window_size == 3:
        mult = 2
    elif window_size == 5:
        mult = 20
    elif window_size == 7:
        mult = 780

    matx = np.array(matx) * mult
    maty = np.array(maty) * mult
    matx = torch.tensor(matx,device='cuda',dtype=torch.float32)
    maty = torch.tensor(maty, device='cuda', dtype=torch.float32)

    return matx, maty


def create_window(window_size, channel):
    windowx, windowy = sobel(window_size)
    windowx, windowy = windowx.unsqueeze(0).unsqueeze(0), windowy.unsqueeze(0).unsqueeze(0)
    windowx = windowx.expand(channel, 1, window_size, window_size)
    windowy = windowy.expand(channel, 1, window_size, window_size)
    # print windowx
    # print windowy

    return windowx, windowy


def gradient(img, windowx, windowy, window_size, padding, channel):
    if channel > 1:  # do convolutions on each channel separately and then concatenate
        gradx = torch.ones(img.shape, device='cuda')
        grady = torch.ones(img.shape, device='cuda')
        for i in range(channel):
            gradx[:, i, :, :] = F.conv2d(img[:, i, :, :].unsqueeze(0), windowx, padding=padding, groups=1).squeeze(
                0)  # fix the padding according to the kernel size
            grady[:, i, :, :] = F.conv2d(img[:, i, :, :].unsqueeze(0), windowy, padding=padding, groups=1).squeeze(0)

    else:
        gradx = F.conv2d(img, windowx, padding=padding, groups=1)
        grady = F.conv2d(img, windowy, padding=padding, groups=1)

    return gradx, grady


class SobelGrad(torch.nn.Module):
    def __init__(self, window_size=3, padding=1):
        super(SobelGrad, self).__init__()
        self.window_size = window_size
        self.padding = padding
        self.channel = 1  # out channel
        self.windowx, self.windowy = create_window(window_size, self.channel)

    def forward(self, pred):
        (batch_size, channel, _, _) = pred.size()

        pred_gradx, pred_grady = gradient(pred, self.windowx, self.windowy, self.window_size, self.padding, channel)

        return pred_gradx, pred_grady


class ETF(nn.Module):
    def __init__(self, kernel_radius, iter_time, dir_num):
        super(ETF, self).__init__()
        self.kernel_size = kernel_radius * 2 + 1
        self.kernel_radius = kernel_radius
        self.iter_time = iter_time
        self.pad = nn.ReflectionPad2d((self.kernel_radius,self.kernel_radius,self.kernel_radius,self.kernel_radius))
        self.dir_num = dir_num
        self.theta = torch.tensor([math.pi*.5],device='cuda')
        self.sobel = SobelGrad()

    def Ws(self, x):
        kernels = torch.ones((*x.shape, self.kernel_size, self.kernel_size), device=x.device)
        # radius = central = (self.kernel_size-1)/2
        # for i in range(self.kernel_size):
        #     for j in range(self.kernel_size):
        #         if (i-central)**2+(i-central)**2 <= radius**2:
        #              self.flow_field[x][y]
        return kernels

    def Wm(self, x, gradient_norm):
        kernels = torch.ones((*x.shape, self.kernel_size, self.kernel_size), device=x.device)

        eta = 1  # Specified in paper
        (N,C,h, w) = x.shape
        gradient_norm = self.pad(gradient_norm)
        x = gradient_norm[:,:,self.kernel_radius:-self.kernel_radius, self.kernel_radius:-self.kernel_radius]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                y = gradient_norm[:,:,i:i + h, j:j + w]
                kernels[:,:,:, :, i, j] = (1 / 2) * (1 + torch.tanh(eta * (y - x)))
        return kernels

    def Wd(self, x, y):
        kernels = torch.ones((*x.shape, self.kernel_size, self.kernel_size), device=x.device)

        (N,C,h, w) = x.shape
        x = self.pad(x)
        y = self.pad(y)

        X_x = x[:,:,self.kernel_radius:-self.kernel_radius, self.kernel_radius:-self.kernel_radius]
        X_y = y[:,:,self.kernel_radius:-self.kernel_radius, self.kernel_radius:-self.kernel_radius]

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                Y_x = x[:,:,i:i + h, j:j + w]
                Y_y = y[:,:,i:i + h, j:j + w]
                kernels[:,:,:,:, i, j] = X_x * Y_x + X_y * Y_y

        return torch.abs(kernels), torch.sign(kernels)

    def forward(self, img):
        B,C,h,w = img.shape

        img_normal = img/img.amax(dim=(2,3),keepdim=True)

        x_der = []
        y_der = []
        for i in range(img_normal.shape[1]):
            x_, y_ = self.sobel(img_normal[:,i,:,:].unsqueeze(1))
            x_der.append(x_)
            y_der.append(y_)
        x_der = torch.cat(x_der,1)
        y_der = torch.cat(y_der,1)
        x_der = torch.clamp(x_der, min = 1e-12)
        y_der = torch.clamp(y_der, min = 1e-12)

        gradient_magnitude = torch.sqrt(x_der ** 2.0 + y_der ** 2.0)
        gradient_norm = gradient_magnitude / gradient_magnitude.amax(dim=(2,3),keepdim=True)

        x_norm = x_der / (gradient_magnitude)
        y_norm = y_der / (gradient_magnitude)

        # rotate 90 degrees counter-clockwise
        x_norm, y_norm = x_norm * torch.cos(self.theta) - y_norm * torch.sin(self.theta),\
                         y_norm * torch.cos(self.theta) + x_norm * torch.sin(self.theta)
        Ws = self.Ws(x_norm)
        Wm = self.Wm(x_norm, gradient_norm)
        for iter_time in range(self.iter_time):
            Wd, phi = self.Wd(x_norm, y_norm)
            kernels = phi * Ws * Wm * Wd

            x_magnitude = self.pad((gradient_norm * x_norm))
            # print(x_magnitude.min())
            y_magnitude = self.pad((gradient_norm * y_norm))

            x_patch = torch.nn.functional.unfold(x_magnitude, (self.kernel_size, self.kernel_size))
            y_patch = torch.nn.functional.unfold(y_magnitude, (self.kernel_size, self.kernel_size))

            x_patch = x_patch.view(B,C,self.kernel_size, self.kernel_size, -1)
            y_patch = y_patch.view(B,C,self.kernel_size, self.kernel_size, -1)

            x_patch = x_patch.permute(0, 1, 4, 2, 3)
            y_patch = y_patch.permute(0, 1, 4, 2, 3)
            kernels = kernels.view(B,C,-1,self.kernel_size,self.kernel_size)
            x_result = (x_patch * kernels).sum(-1).sum(-1)
            y_result = (y_patch * kernels).sum(-1).sum(-1)

            magnitude = torch.sqrt(x_result ** 2.0 + y_result ** 2.0)
            x_norm = x_result / magnitude
            y_norm = y_result / magnitude
            x_norm = x_norm.view(B,C,h,w)
            y_norm = y_norm.view(B, C, h, w)

        tan = -y_norm / x_norm
        angle = torch.atan(tan)
        angle = F.instance_norm((180 * angle / torch.pi))
        return angle

    def save(self, x, y):
        x = x.squeeze()
        y = y.squeeze()
        x[x == 0] += 1e-12

        tan = -y / x
        angle = torch.atan(tan)
        angle = 180 * angle / math.pi

        length = 180 / self.dir_num
        for i in range(self.dir_num):
            if i == 0:
                minimum = -90
                maximum = -90 + length / 2
                mask1 = 255 * (((angle > minimum) + (angle == minimum)) * (angle < maximum))
                maximum = 90
                minimum = 90 - length / 2
                mask2 = 255 * ((angle > minimum) + (angle == minimum))
                mask = mask1 + mask2
                return mask
            else:
                minimum = -90 + (i - 1 / 2) * length
                maximum = minimum + length
                mask = 255 * (((angle > minimum) + (angle == minimum)) * (angle < maximum))
                return mask


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
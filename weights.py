import torch
import torch.nn as nn
import typing
import numpy as np

activation_std = 0.5893595616022745

def init_(t, dim = None):
    dim = dim if dim is not None else t.shape[-1]
    std = 1. / math.sqrt(dim)
    return torch.nn.init.normal_(t, mean=0, std=std)

def conv_weight(in_features: int, out_features: int, kernel_size: int, groups: int, std: float):
    return nn.Parameter(nn.init.xavier_normal_(torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size, groups=groups).weight))

def rnoise_weight(size):
    return nn.ParameterList([nn.Parameter(nn.init.normal_(torch.ones(size, size))),
                                        nn.Parameter(nn.init.normal_(torch.ones(size, size))),
                                        nn.Parameter(nn.init.constant_(torch.ones(1, ), .5)),
                                        nn.Parameter(nn.init.constant_(torch.ones(1, ), .5))])

def bias(dim):
    return nn.Parameter(nn.init.constant_(torch.zeros(dim, ), 0))

def adaconv_weight(s_d, channels, n_groups,):
    params = []
    params.append(nn.ParameterList([conv_weight(s_d, channels * (channels // n_groups), 2, 1, activation_std), bias(channels * (channels // n_groups))]))
    params.append(nn.ParameterList([conv_weight(s_d, channels * (channels // n_groups), 1, 1, activation_std), bias(channels * (channels // n_groups))]))
    params.append(nn.ParameterList([conv_weight(s_d, channels, 1, 1, activation_std),bias(channels)]))
    return nn.ModuleList(params)

def resblock_weight(ch):
    params = []
    params.append(nn.ParameterList([conv_weight(ch, ch, 3, 1, activation_std), bias(ch)]))
    params.append(nn.ParameterList([conv_weight(ch, ch, 1, 1, activation_std), bias(ch)]))
    return nn.ModuleList(params)

def fused_conv_noise_weights(ch_in, ch_out, noise=False, noise_size=256):
    params = []
    params.append(nn.ParameterList([conv_weight(ch_in, ch_out, 3, 1, activation_std)]))
    if noise:
        params.append(rnoise_weight(noise_size))
    params.append(nn.ParameterList([bias(ch_out)]))
    if ch_in != ch_out:
        params.append(nn.ParameterList([conv_weight(ch_in, ch_out, 3, 1, activation_std),bias(ch_out)]))
    return nn.ModuleList(params)

def downblock_weights():
    params = []
    for i,j,k,l in [(6, 128, False, 0), (128, 128, False, 0), (128, 64, False, 0), (64, 64, False, 0), (64, 64, False, 0)]:
        params.append(fused_conv_noise_weights(i, j, noise=k, noise_size=l))
    return nn.ModuleList(params)

def upblock_weights():
    params = []
    for i, j, k, l in [(64, 64, False, 0), (64, 128, False, 0), (128, 128, False, 0), (128, 3, False, 0)]:
        params.append(fused_conv_noise_weights(i, j, noise=k, noise_size=l))
    params.append(nn.ParameterList([conv_weight(3, 3, 1, 1, activation_std),bias(3)]))
    return nn.ModuleList(params)

def up_adaconv_weights(s_d):
    params=[]
    params.append(adaconv_weight(s_d,64, 1))
    params.append(adaconv_weight(s_d, 64, 1))
    params.append(adaconv_weight(s_d, 128, 2))
    return nn.ModuleList(params)

def style_encoder_block(ch):
    params = []
    params.append(nn.ParameterList([conv_weight(ch,ch,3,1,activation_std),bias(ch)]))
    params.append(nn.ParameterList([conv_weight(ch,ch,1,1,activation_std),bias(ch)]))
    return nn.ModuleList(params)

def style_encoder_weights(ch, s_d):
    params = []
    for i in range(3):
        params.append(style_encoder_block(ch))
    params.append(nn.ParameterList([nn.Parameter(torch.normal(torch.zeros(s_d * dim, 8192),
                                                          torch.ones(s_d * dim, 8192))),bias(s_d * dim)]))
    return nn.ModuleList(params)
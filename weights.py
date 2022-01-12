import torch
import torch.nn as nn
import typing
import numpy as np

activation_std = 0.5893595616022745

def orthonormal(inp: typing.Union[torch.Tensor, torch.nn.Parameter, typing.List[int]], gain: float):
    original_input = inp
    if isinstance(inp, list):
        inp = torch.zeros(inp)
    if isinstance(inp, torch.nn.Parameter):
        inp = inp.data
    flat_shape = (inp.shape[0], np.prod(inp.shape[1:]))
    a = torch.rand(flat_shape)
    u, _, v = torch.linalg.svd(a, full_matrices=False)
    inp.copy_((u if u.shape == flat_shape else v).reshape(inp.shape).mul(gain).to(device=inp.device, dtype=inp.dtype))
    if isinstance(original_input, list):
        return torch.nn.Parameter(inp)
    return original_input

def init_(t, dim = None):
    dim = dim if dim is not None else t.shape[-1]
    std = 1. / math.sqrt(dim)
    return torch.nn.init.normal_(t, mean=0, std=std)

def conv_weight(in_features: int, out_features: int, kernel_size: int, groups: int, std: float):
    return orthonormal(torch.nn.Conv2d(in_features, out_features, (kernel_size,), groups=groups).weight, 1 / std)

def conv1d_weight(in_features: int, out_features: int, kernel_size: int, groups: int, std: float):
    return orthonormal(torch.nn.Conv1d(in_features, out_features, (kernel_size,), groups=groups).weight, 1 / std)

def rnoise_weight(size):
    return nn.ParameterList([nn.Parameter(nn.init.normal_(torch.ones(size, size))),
                                        nn.Parameter(nn.init.normal_(torch.ones(size, size))),
                                        nn.Parameter(nn.init.constant_(torch.ones(1, ), .5)),
                                        nn.Parameter(nn.init.constant_(torch.ones(1, ), .5))])

def bias():
    return nn.Parameter(nn.init.constant_(torch.ones(1, ), 0))

def adaconv_weight(s_d, channels, n_groups,):
    params = []
    params.append(nn.ParameterList([conv_weight(s_d, channels * (channels // n_groups), 2, 1, activation_std), bias()]))
    params.append(nn.ParameterList([conv_weight(s_d, channels * (channels // n_groups), 1, 1, activation_std), bias()]))
    params.append(nn.ParameterList([conv_weight(s_d, channels, 1, 1, activation_std),bias()]))
    params.append(bias())
    return nn.ParameterList(params)

def resblock_weight(ch):
    params = []
    params.append(nn.ParameterList([conv_weight(ch, ch, 3, 1, activation_std), bias()]))
    params.append(nn.ParameterList([conv_weight(ch, ch, 1, 1, activation_std), bias()]))
    return nn.ModuleList(params)

def fused_conv_noise_weights(ch_in, ch_out, noise=False, noise_size=256):
    params = []
    params.append(nn.ParameterList([conv_weight(ch_in, ch_out, 3, 1, activation_std)]))
    if noise:
        params.append(rnoise_weight(noise_size))
    params.append(nn.ParameterList([bias()]))
    if ch_in != ch_out:
        params.append(nn.ParameterList([conv_weight(ch_in, ch_out, 3, 1, activation_std),bias()]))
    return nn.ModuleList(params)

def downblock_weights():
    params = []
    for i,j,k,l in [(6, 128, False, 0), (128, 128, False, 0), (128, 64, False, 0), (64, 64, False, 0), (64, 64, False, 0)]:
        params.append(fused_conv_noise_weights(i, j, noise=k, noise_size=l))
    return nn.ModuleList(params)

def upblock_weights():
    params = []
    for i, j, k, l in [(64, 64, False, 0), (64, 128, False, 0), (128, 128, False, 0), (128, 3, False, 0),
                       (64, 64, False, 0)]:
        params.append(fused_conv_noise_weights(i, j, noise=k, noise_size=l))
    params.append(nn.ParameterList([conv_weight(3, 3, 1, 1, activation_std),bias()]))
    return nn.ModuleList(params)

def up_adaconv_weights(s_d):
    params=[]
    params.append(adaconv_weight(s_d,64, 1))
    params.append(adaconv_weight(s_d, 64, 1))
    params.append(adaconv_weight(s_d, 128, 2))
    return nn.ParameterList(params)

def style_encoder_weights(s_d, dim=16):
    params = []
    params.append(torch.nn.Parameter(torch.normal(torch.zeros(s_d * dim, s_d * dim),
                                                          torch.ones(s_d * dim, s_d * dim))))
    params.append(bias())
    return nn.ParameterList(params)
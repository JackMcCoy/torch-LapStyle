import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(inp, weight, groups, use_pad=True,bias=None):
    if use_pad:
        inp = F.pad(inp, (1, 1, 1, 1), mode='reflect')
    return F.conv2d(inp, weight, groups=groups, bias=bias)

def conv1d(inp, weight, groups, use_pad=True,bias=None):
    if use_pad:
        inp = F.pad(inp, 1, mode='reflect')
    return F.conv1d(inp, weight, groups=groups, bias=bias)

def fused_conv_noise_bias(inp, weights, scale_change='',noise=False):
    if scale_change == 'up':
        resized = F.upsample(inp, scale_factor=2, mode='nearest')
    elif scale_change == 'down':
        resized = F.avg_pool2d(inp, 2, stride=2)
    else:
        resized = inp
    out = conv(resized, weights[0][0], 1)
    if noise:
        out = rnoise(out, weights[1])
    # note: including noise will create order conflict
    out = out + weights[1][0]
    out = F.leaky_relu(out)
    if len(weights)>3:
        resized = conv(resized, weights[3][0], 1, bias=weights[3][1])
    return (out + resized) * torch.rsqrt((torch.ones(1,device='cuda:0')*2))

def adaconv(input, weights, style_encoding, n_groups, ch):
    depthwise_w, pw_cn_w, pw_cn_b = weights
    N = style_encoding.shape[0]
    depthwise = conv(style_encoding, depthwise_w[0], 1, use_pad=False, bias=depthwise_w[1])
    depthwise = depthwise.view(N, ch, ch // n_groups, 3, 3)
    s_d = F.avg_pool2d(style_encoding, 4)
    pointwise_kn = conv(s_d,pw_cn_w[0], 1, use_pad=False, bias=pw_cn_w[1]).view(N, ch, ch//n_groups, 1, 1)
    pointwise_bias = conv(s_d,pw_cn_b[0], 1, use_pad=False, bias=pw_cn_b[1]).view(N, ch)

    a, b, c, d = input.size()
    mean = input.mean(dim=(2,3), keepdim=True)
    predicted = input -mean
    predicted = predicted * torch.rsqrt(predicted.square().mean(dim=(2,3), keepdim=True)+1e-5)
    content_out = torch.empty_like(predicted, device='cuda:0')
    for i in range(a):
        content_out[i] = nn.functional.conv2d(
            nn.functional.conv2d(self.pad(predicted[i].unsqueeze(0)),
                                         weight=depthwise[i],
                                         stride=1,
                                         groups=n_groups
                                         ),
                             stride = 1,
                             weight=pointwise_kn[i],
                             bias=pointwise_bias[i],
                             groups=n_groups).squeeze()
    return content_out

def downblock(inp, weights):
    out = fused_conv_noise_bias(inp, weights[0])
    out = fused_conv_noise_bias(out, weights[1])
    out = fused_conv_noise_bias(out, weights[2])
    out = fused_conv_noise_bias(out, weights[3], scale_change='down')
    out = fused_conv_noise_bias(out, weights[4])
    return out

def upblock_w_adaconvs(inp, style_encoding, weights, adaconv_weights, adaconv_param_list=[(1, 64),(1, 64),(2, 128)]):
    out = adaconv(inp,adaconv_weights[0],style_encoding, *adaconv_param_list[0])
    out = fused_conv_noise_bias(out, weights[0], scale_change='up')
    out = adaconv(out,adaconv_weights[1],style_encoding, *adaconv_param_list[1])
    out = fused_conv_noise_bias(out, weights[1])
    out = adaconv(out, adaconv_weights[2], style_encoding, *adaconv_param_list[2])
    out = fused_conv_noise_bias(out, weights[2])
    out = fused_conv_noise_bias(out, weights[3])
    out = conv(out,weights[4][0],1,bias=weights[4][1])
    return out

def style_projection(inp, weights, s_d):
    N = inp.shape[0]
    out = inp.flatten(1)
    out = F.linear(out,weights[0],bias=weights[1])
    out = F.leaky_relu(out.reshape(N, s_d, 4, 4))
    return out

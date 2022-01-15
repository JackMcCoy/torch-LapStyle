import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        ctx.save_for_backward(inp)
        return inp

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        inp, = ctx.saved_tensors
        inp.mean().backward()

def conv(inp, weight, groups, use_pad=True,bias=None):
    if use_pad:
        inp = F.pad(inp, (1, 1, 1, 1), mode='reflect')
    return F.conv2d(inp, weight, groups=groups, bias=bias)

def conv1d(inp, weight, groups, use_pad=True,bias=None):
    if use_pad:
        inp = F.pad(inp, 1, mode='reflect')
    return F.conv1d(inp, weight, groups=groups, bias=bias)

def rnoise(x, params):
    N, c, h, w = x.shape
    A, b, alpha, r = params

    s = torch.sum(x, dim=1, keepdim=True)
    s = s - s.mean(dim=(2,3), keepdim=True)
    s_max = s.abs().amax(dim=(2,3), keepdim=True)
    s = s / (s_max + 1e-8)
    s = (s + 1) / 2
    s = s * A + b
    s = torch.tile(s, (1, c, 1, 1))
    sp_att_mask = (1 - alpha) + alpha * s
    sp_att_mask = sp_att_mask / (torch.linalg.norm(sp_att_mask, dim=1, keepdims=True) + 1e-8)
    zero = torch.zeros(1, device='cuda')
    x = r * sp_att_mask * x + r * sp_att_mask * (zero.repeat(N, c, h,w).normal_())
    return x

def fused_conv_noise_bias(inp, weights, scale_change='',noise=False):
    if scale_change == 'up':
        resized = F.upsample(inp, scale_factor=2, mode='nearest')
    elif scale_change == 'down':
        resized = F.avg_pool2d(inp, 2, stride=2)
    else:
        resized = inp
    out = conv(resized, weights[0][0], 1)
    if noise:
        out = rnoise(out, weights[2])
    # note: including noise will create order conflict
    out = out + weights[1][0].view(1,-1,1,1)
    out = F.leaky_relu(out)
    if weights[0][0].shape[0] != weights[0][0].shape[1]:
        resized = conv(resized, weights[-1][0], 1, bias=weights[-1][1])
    return (out + resized)

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
            nn.functional.conv2d(F.pad(predicted[i].unsqueeze(0),(1, 1, 1, 1), mode='reflect'),
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
    out = fused_conv_noise_bias(out, weights[3], scale_change='down',)
    out = fused_conv_noise_bias(out, weights[4])
    return out

def upblock_w_adaconvs(inp, style_encoding, weights, adaconv_weights, adaconv_param_list=[(1, 64),(1, 64),(2, 128)]):
    out = fused_conv_noise_bias(inp, weights[0], scale_change='up')
    out = fused_conv_noise_bias(out, weights[1])
    out = fused_conv_noise_bias(out, weights[2])
    out = fused_conv_noise_bias(out, weights[3])
    out = conv(out,weights[4][0],1,bias=weights[4][1],use_pad=False)
    return out

def style_projection(inp, weights, s_d):
    for i in weights[:-1]:
        inp = conv(inp,i[0][0], 1, bias=i[0][1]).relu()
        inp = F.avg_pool2d(inp, kernel_size=2, stride=2)
        inp = conv(inp,i[1][0], 1, bias=i[1][1]).relu()
    inp = inp.flatten(1)
    inp = F.linear(inp, weights[-1][0], bias=weights[-1][1])
    b = inp.shape[0]
    inp = inp.reshape(b, s_d, 4, 4)
    return inp

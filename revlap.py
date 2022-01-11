from adaconv import AdaConv
from net import style_encoder_block, ResBlock
from modules import RiemannNoise, PixelShuffleUp, StyleEncoderBlock, FusedConvNoiseBias
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import typing, math, copy
from functools import partial
from einops.layers.torch import Rearrange
from revlib.utils import module_list_to_momentum_net
import revlib


class MomentumNetStem(torch.nn.Module):
    def __init__(self, wrapped_module: torch.nn.Module, beta: float):
        super(MomentumNetStem, self).__init__()
        self.wrapped_module = wrapped_module
        self.beta = beta

    def forward(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.wrapped_module(inp * self.beta, *args, **kwargs)


class MomentumNetSide(torch.nn.Module):
    def __init__(self, beta: float):
        super(MomentumNetSide, self).__init__()
        self.beta = beta

    def forward(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return inp * self.beta

def style_encoder_block(s_d):
    return nn.Sequential(
            nn.Linear(s_d * 16, s_d * 16),
            nn.ReLU()
        )

def downblock():
    return nn.Sequential(FusedConvNoiseBias(6, 128, 256, 'none', noise=False),
                          FusedConvNoiseBias(128, 128, 256, 'none', noise = False),
                          FusedConvNoiseBias(128, 64, 256, 'none', noise=False),
                          FusedConvNoiseBias(64, 64, 128, 'down', noise=False),
                          ResBlock(64, hw= 128, noise = True),)

def upblock():
    return nn.ModuleList([
                          FusedConvNoiseBias(64, 64, 256, 'up'),
                          FusedConvNoiseBias(64, 128, 256, 'none', noise=False),
                          FusedConvNoiseBias(128, 128, 256, 'none'),
                          nn.Sequential(
                              FusedConvNoiseBias(128, 3, 256, 'none', noise=False),
                              nn.Conv2d(3, 3, kernel_size=1))
                      ]
    )

def adaconvs(batch_size,s_d):
    return nn.ModuleList([
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(128, 2, batch_size, s_d=s_d)])


def cropped_coupling_forward(total_height, height, layer_num, other_stream: torch.Tensor, fn_out: torch.Tensor):
    fn_out = revlib.core.split_tensor_list(fn_out)

    layer_res = 512*2**height # 512
    up_f = 256*2**((total_height-1)-height) #256
    row_num = layer_res // 256 # 2
    lr = math.floor(layer_num / row_num) # 0 -> 0
    # 1 -> 0    2->1    3 ->
    lc = layer_num % row_num # 0 -> 0   1 -> 1
    # 2 -> 0    3 -> 1

    if isinstance(fn_out, torch.Tensor):
        combined = other_stream[:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)] \
                   + fn_out[:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)]
        other_stream[:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)] = combined
        #print(f'{layer_num} forward - {up_f * lc}: {up_f * (lc + 1)}, {up_f * lr}: {up_f * (lr + 1)}')
        return other_stream

    combined = other_stream[:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)] \
               + fn_out[0][:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)]
    other_stream[:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)] = combined

    return [other_stream]\
           + fn_out[1]

def cropped_coupling_inverse(total_height, height, layer_num, output: torch.Tensor, fn_out: torch.Tensor):
    fn_out = revlib.core.split_tensor_list(fn_out)

    layer_res = 512 * 2 ** height
    up_f = 256 * 2 ** ((total_height-1) - height)
    row_num = layer_res // 256
    lr = math.floor(layer_num / row_num)
    lc = layer_num % row_num

    if isinstance(fn_out, torch.Tensor):
        diff = output[:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)] \
               - fn_out[:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)]
        output[:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)] = diff
        #print(f'{layer_num} backward - {up_f * lc}: {up_f * (lc + 1)}, {up_f * lr}: {up_f * (lr + 1)}')
        return output
    diff = output[:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)] \
           - fn_out[0][:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)]
    output[:, :, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)] = diff
    return [output]\
           + fn_out[1]

lap_weight = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
lap_weight = torch.Tensor(lap_weight).to(torch.device('cuda:0'))
lap_weight.requires_grad = False


class Sequential_Worker(nn.Module):
    def __init__(self, init_scale:float, layer_height, num, max_res,working_res, batch_size,s_d):
        super(Sequential_Worker, self).__init__()
        self.init_scale = init_scale
        self.style_projection = style_encoder_block(s_d)
        self.downblock = downblock()
        self.upbloack = upblock()
        self.adaconvs = adaconvs(batch_size, s_d)
        self.working_res = working_res
        self.s_d = 512
        self.max_res =max_res
        self.layer_height = layer_height
        self.num = num

        # row_num == col_num, as these are squares

    def get_layer_rows(self, layer_res):
        row_num = layer_res // self.working_res
        layer_row = math.floor(self.num / row_num)
        layer_col = self.num % row_num
        return layer_row, layer_col, row_num

    def crop_to_working_area(self, x, layer_row, layer_col):
        return x[:,:,self.working_res*layer_col:self.working_res*(layer_col+1),self.working_res*layer_row:self.working_res*(layer_row+1)]

    def reinsert_work(self, x, out, layer_row, layer_col):

        x[:, :, self.working_res * layer_col:self.working_res * (layer_col + 1),
        self.working_res * layer_row:self.working_res * (layer_row + 1)] = out
        return out

    def resize_to_res(self, x, layer_res):
        return F.interpolate(x, layer_res, mode='nearest')

    def crop_style_thumb(self, x, layer_res, row, col, row_num):
        style_col = col if col % 2 == 0 else col - 1
        style_row = row
        if row + 1 >= row_num:
            style_row -= 1
        scaled = F.interpolate(x, layer_res//2, mode='nearest')
        scaled = scaled[:,:,self.working_res*style_col:self.working_res*(style_col+1),self.working_res*style_row:self.working_res*(style_row+1)]
        return scaled

    def return_to_full_res(self, x):
        return F.interpolate(x, self.max_res, mode='nearest')

    def momentum(self, init_scale, layer_num):
        out = copy.copy(self)
        out.init_scale = init_scale
        out.num = layer_num
        return out

    def forward(self, x, ci, style):
        # x = input in color space
        # out = laplacian (residual) space
        layer_res = 512*2**self.layer_height
        row, col, row_num = self.get_layer_rows(layer_res)

        x = self.resize_to_res(x, layer_res)
        ci = self.resize_to_res(ci,layer_res)
        out = self.crop_to_working_area(x, row, col)
        lap = self.crop_to_working_area(ci, row, col)

        lap = F.conv2d(F.pad(lap, (1,1,1,1), mode='reflect'), weight = lap_weight, groups = 3)
        out = torch.cat([out, lap], dim=1)

        N,C,h,w = style.shape
        style = style.flatten(1)
        style = self.style_projection(style)
        style = style.reshape(N, self.s_d, 4, 4)
        out = self.downblock(out)
        for idx, (ada, learnable) in enumerate(zip(self.adaconvs, self.upblock)):
            if idx > 0:
                out = ada(style, out)
            out = learnable(out)
        out = upblock[-1](out)
        out = self.reinsert_work(x, out, row, col)
        out = self.return_to_full_res(out)
        return out


class LapRev(nn.Module):
    def __init__(self, max_res, working_res, batch_size, s_d, momentumnet_beta):
        super(LapRev, self).__init__()
        self.max_res = max_res
        self.momentumnet_beta = momentumnet_beta
        self.working_res = working_res
        height = max_res//working_res

        self.num_layers = [(h,i) for h in range(height) for i in range(int((2**h)/.25))]
        coupling_forward = [c for h, i in self.num_layers for c in (partial(cropped_coupling_forward, height, h, i),)*2]
        coupling_inverse = [c  for h, i in self.num_layers for c in (partial(cropped_coupling_inverse, height, h, i),)*2]
        cells = [Sequential_Worker(1., i, 0, self.max_res,256, batch_size, s_d) for i in range(height)]

        modules = nn.ModuleList([cells[height].momentum((1 - self.momentumnet_beta) / self.momentumnet_beta ** i,
                                                                                   layer_num) for i, (height, layer_num) in enumerate(self.num_layers)])

        self.layers = module_list_to_momentum_net(modules,
                                                  beta=self.momentumnet_beta,
                                                  coupling_forward = coupling_forward,
                                                  coupling_inverse = coupling_inverse,
                                                  target_device='cuda:0')

    def forward(self, input:torch.Tensor, ci:torch.Tensor, style:torch.Tensor):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        #input = F.interpolate(input, self.max_res, mode='nearest').repeat(1,2,1,1).data.to(torch.device('cuda:0'))
        #input.requires_grad = True
        out = F.interpolate(input, self.max_res, mode='nearest')

        for idx, layer in zip(self.num_layers,self.layers):
            out = layer(out,ci, style.data)
        return out
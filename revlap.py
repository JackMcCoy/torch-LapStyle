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

def calc_crop_indices(layer_height,layer_num,total_height):
    layer_res = 512 * 2 ** layer_height
    up_f = 256 * 2 ** ((total_height - 1) - layer_height)  # 256
    row_num = layer_res // 256  # 2
    lr = math.floor(layer_num / row_num)  # 0 -> 0
    # 1 -> 0    2->1    3 ->
    lc = layer_num % row_num
    ci1 = up_f * lc
    ci2 = up_f * (lc + 1)
    ri1 = up_f * lr
    ri2 = up_f * (lr + 1)
    return ci1, ci2, ri1, ri2


class MomentumNetStem(torch.nn.Module):
    def __init__(self, wrapped_module: torch.nn.Module, beta: float, layer_height:int, layer_num:int, total_height:int):
        super(MomentumNetStem, self).__init__()
        self.wrapped_module = wrapped_module
        self.beta = beta
        self.layer_num = layer_num
        self.ci1, self.ci2, self.ri1, self.ri2 = calc_crop_indices(layer_height,layer_num,total_height)
        
    def forward(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        inp = self.wrapped_module(inp, *args, **kwargs)
        #print(f'{self.layer_num}')
        y = inp.clone()
        y[:,:,self.ci1:self.ci2,self.ri1:self.ri2] = y[:,:,self.ci1:self.ci2,self.ri1:self.ri2]*self.beta
        return y


class MomentumNetSide(torch.nn.Module):
    def __init__(self, beta: float, layer_height:int, layer_num:int, total_height:int):
        super(MomentumNetSide, self).__init__()
        self.beta = beta
        self.ci1, self.ci2, self.ri1, self.ri2 = calc_crop_indices(layer_height, layer_num, total_height)

    def forward(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y = inp.clone()
        y[:,:,self.ci1:self.ci2,self.ri1:self.ri2] = y[:,:,self.ci1:self.ci2,self.ri1:self.ri2] * self.beta
        return y

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
    ci1, ci2, ri1, ri2 = calc_crop_indices(height,layer_num,total_height)

    if not isinstance(fn_out, list):
        y = fn_out.clone()
        y[:, :, ci1: ci2, ri1: ri2]= \
            other_stream[:, :, ci1: ci2, ri1: ri2]+\
                   y[:, :, ci1: ci2, ri1: ri2]
        return y

    other_stream[:, :, ci1: ci2, ri1: ri2].add_(
               fn_out[0][:, :, ci1: ci2, ri1: ri2])
    return [other_stream]\
           + fn_out[1]

def cropped_coupling_inverse(total_height, height, layer_num, output: torch.Tensor, fn_out: torch.Tensor):
    ci1, ci2, ri1, ri2 = calc_crop_indices(height,layer_num,total_height)

    if isinstance(fn_out, list):
        y = output.clone()
        y[:, :, ci1: ci2, ri1: ri2]= \
            y[:, :, ci1: ci2, ri1: ri2] -\
               fn_out[:, :, ci1: ci2, ri1: ri2]
        return y
    output[:, :, ci1: ci2, ri1: ri2].subtract_(
           fn_out[0][:, :, ci1: ci2, ri1: ri2])
    return [output]\
           + fn_out[1]


class Sequential_Worker(nn.Module):
    def __init__(self, init_scale:float, layer_height, num, max_res,working_res, batch_size,s_d):
        super(Sequential_Worker, self).__init__()
        self.init_scale = init_scale
        self.working_res = working_res
        self.s_d = 512
        self.max_res =max_res
        self.layer_height = layer_height
        self.num = num
        self.style_projection = style_encoder_block(s_d)
        self.downblock = downblock()
        self.upblock = upblock()
        self.adaconvs = adaconvs(batch_size, s_d)
        self.lap_weight = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
        self.lap_weight = torch.Tensor(lap_weight).to(torch.device('cuda:0'))
        self.lap_weight.requires_grad = False
        # row_num == col_num, as these are squares

    def get_layer_rows(self, layer_res):
        row_num = layer_res // self.working_res
        layer_row = math.floor(self.num / row_num)
        layer_col = self.num % row_num
        return layer_row, layer_col, row_num

    def crop_to_working_area(self, x, layer_row, layer_col):
        return x[:,:,self.working_res*layer_col:self.working_res*(layer_col+1),self.working_res*layer_row:self.working_res*(layer_row+1)]

    def reinsert_work(self, x, out, layer_row, layer_col):
        y = x.clone()
        y[:, :, self.working_res * layer_col:self.working_res * (layer_col + 1),
        self.working_res * layer_row:self.working_res * (layer_row + 1)] = y[:, :, self.working_res * layer_col:self.working_res * (layer_col + 1),
        self.working_res * layer_row:self.working_res * (layer_row + 1)]+out
        return y

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

    def copy(self, layer_num):
        out = copy.copy(self)
        out.num = layer_num
        return out

    def forward(self, x, ci, style):
        # x = input in color space
        # out = laplacian (residual) space
        layer_res = 512*2**self.layer_height
        row, col, row_num = self.get_layer_rows(layer_res)
        if layer_res != self.max_res:
            x = self.resize_to_res(x, layer_res)
            ci = self.resize_to_res(ci,layer_res)
        out = self.crop_to_working_area(x, row, col)
        lap = self.crop_to_working_area(ci, row, col)
        lap = F.conv2d(F.pad(lap, (1,1,1,1), mode='reflect'), weight = self.lap_weight, groups = 3)
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
        out = self.upblock[-1](out)
        out = self.reinsert_work(x, out, row, col)
        if layer_res != self.max_res:
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
        coupling_inverse = [c for h, i in self.num_layers for c in (partial(cropped_coupling_inverse, height, h, i),)*2]
        coupling_inverse.reverse()
        cells = [Sequential_Worker(1., i, 0, self.max_res,256, batch_size, s_d) for i in range(height)]

        modules = nn.ModuleList([cells[height].copy(layer_num) for height, layer_num in self.num_layers])
        momentum_modules = []
        for idx, (mod,(h,i)) in enumerate(zip(modules,self.num_layers)):
            momentum_modules.append(MomentumNetStem(mod, self.momentumnet_beta ** h, h,i,height))
            momentum_modules.append(MomentumNetSide((1 - self.momentumnet_beta) / self.momentumnet_beta ** (h + 1), h,i,height))
        self.momentumnet = revlib.ReversibleSequential(*momentum_modules,split_dim=0,memory_mode = revlib.MemoryModes.no_savings,coupling_forward=coupling_forward,coupling_inverse=coupling_inverse,target_device='cuda')
        '''
        secondary_branch_buffer = []
        stem = list(momentumnet.stem)[:-1]
        modules = [
            revlib.core.SingleBranchReversibleModule(secondary_branch_buffer, wrapped_module=mod.wrapped_module.wrapped_module,
                                         coupling_forward=mod.wrapped_module.coupling_forward,
                                         coupling_inverse=mod.wrapped_module.coupling_inverse,
                                         memory_savings=mod.memory_savings, target_device=mod.target_device,
                                         cache=mod.cache, first=idx == 0, last=idx == len(stem)-1)
            for idx, mod in enumerate(stem)]
        out_modules = [revlib.core.MergeCalls(modules[i], modules[i + 1], collate_fn=lambda y, x: [y] + x[0][1:])
                       for i in range(0, len(stem)-1, 2)]
        out_modules.append(modules[-1])
        #for i in range(0,len(modules),2):
        #    out_modules.append(modules[i])
        self.layers = nn.ModuleList(out_modules)
        '''
    def forward(self, input:torch.Tensor, ci:torch.Tensor, style:torch.Tensor):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        N,C,h,w = input.shape
        #input = F.interpolate(input, self.max_res, mode='nearest').repeat(1,2,1,1).data.to(torch.device('cuda:0'))
        #input.requires_grad = True
        input = F.interpolate(input, self.max_res, mode='nearest')
        out = input.repeat(2,1,1,1)
        out = self.momentumnet(out,ci, style)

        out = out[N:,:, :,:]
        return out
from adaconv import AdaConv
from net import style_encoder_block, ResBlock
from modules import RiemannNoise, PixelShuffleUp, StyleEncoderBlock, FusedConvNoiseBias
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import typing, math
from functools import partial
from einops.layers.torch import Rearrange
from revlib.utils import module_list_to_momentum_net

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


def cropped_coupling_forward(total_height, height, layer_num, other_stream: torch.Tensor, fn_out: torch.Tensor) -> TENSOR_OR_LIST:
    fn_out = revlib.core.split_tensor_list(fn_out)

    layer_res = 512*2**height
    up_f = 256*2**(total_height-height)
    row_num = layer_res // 256
    lr = math.floor(layer_num / row_num)
    lc = layer_num % row_num

    if isinstance(fn_out, torch.Tensor):
        return other_stream + fn_out
    return [other_stream[:,:, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)]\
            + fn_out[0][:,:, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)]]\
           + fn_out[1][:,:, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)]


def additive_coupling_inverse(height, layer_num, output: torch.Tensor, fn_out: torch.Tensor) -> TENSOR_OR_LIST:
    fn_out = revlib.core.split_tensor_list(fn_out)

    layer_res = 512 * 2 ** height
    up_f = 256 * 2 ** (total_height - height)
    row_num = layer_res // 256
    lr = math.floor(layer_num / row_num)
    lc = layer_num % row_num
    if isinstance(fn_out, torch.Tensor):
        return output - fn_out
    return [output[:,:, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)]\
            - fn_out[0][:,:, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)]]\
           + fn_out[1][:,:, up_f * lc: up_f * (lc + 1), up_f * lr: up_f * (lr + 1)]

lap_weight = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
lap_weight = torch.Tensor(lap_weight).to(torch.device('cuda:0'))
lap_weight.requires_grad = False

class Sequential_Worker(nn.Module):
    def __init__(self, layer_height, num, max_res,working_res, batch_size,s_d):
        super(Sequential_Worker, self).__init__()
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
        self.working_res * layer_row:self.working_res * (layer_row + 1)] += out
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

    def forward(self, x, params, ci, style):
        # x = input in color space
        # out = laplacian (residual) space
        style_projection,downblock, upblock,adaconvs = params
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
        style = style_projection(style)
        style = style.reshape(N, self.s_d, 4, 4)
        out = downblock(out)
        for idx, (ada, learnable) in enumerate(zip(adaconvs, upblock)):
            if idx > 0:
                out = ada(style, out)
            out = learnable(out)
        out = upblock[-1](out)
        out = self.reinsert_work(x, out, row, col)
        out = self.return_to_full_res(out)
        return out


class LapRev(nn.Module):
    def __init__(self, max_res, working_res, batch_size, s_d):
        super(LapRev, self).__init__()
        self.max_res = max_res
        self.working_res = working_res
        height = max_res//working_res
        self.num_layers = [(h,i) for h in range(height) for i in range(int((2**h)/.25))]
        coupling_forward = [partial(cropped_coupling_forward, height, h, i) for h, i in self.num_layers]
        coupling_inverse = [partial(cropped_coupling_inverse, height, h, i) for h, i in self.num_layers]
        self.params = nn.ModuleList([nn.ModuleList([style_encoder_block(s_d), downblock(),upblock(),adaconvs(batch_size, s_d)]) for h in range(height)])
        self.layers = module_list_to_momentum_net(nn.ModuleList([Sequential_Worker(*i,self.max_res,256, batch_size, s_d) for i in self.num_layers]),
                                                  beta=.5,
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
            height, num = idx
            out = layer(out, self.params[height],ci, style.data)
        return out
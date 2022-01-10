from adaconv import AdaConv
from net import style_encoder_block, ResBlock
from modules import RiemannNoise, PixelShuffleUp, StyleEncoderBlock, FusedConvNoiseBias
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import typing, math
from einops.layers.torch import Rearrange
from revlib.utils import module_list_to_momentum_net

def additive_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return upsample(other_stream),  + fn_out


def additive_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return output - downsample(fn_out)

def style_encoder_block(s_d):
    return nn.Sequential(StyleEncoderBlock(512),
            StyleEncoderBlock(512),
            StyleEncoderBlock(512)), nn.Sequential(
            nn.Linear(8192, s_d * 16),
            nn.ReLU()
        )




def Down_and_Up():
    return nn.ModuleList([FusedConvNoiseBias(6, 128, 256, 'none', noise=False),
                          FusedConvNoiseBias(128, 128, 256, 'none'),
                          FusedConvNoiseBias(128, 64, 256, 'none', noise=False),
                          FusedConvNoiseBias(64, 64, 128, 'down', noise=False),
                          ResBlock(64),
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
            nn.Identity(),
            AdaConv(128, 2, batch_size, s_d=s_d),
            AdaConv(128, 2, batch_size, s_d=s_d),
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(128, 2, batch_size, s_d=s_d)])

blank_canvas = torch.zeros(4,3,512,512, device='cuda:0')
lap_weight = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
lap_weight = torch.Tensor(lap_weight).to(torch.device('cuda:0'))
lap_weight.requires_grad = False

class Sequential_Worker(nn.Module):
    def __init__(self, max_res,working_res, batch_size,s_d):
        super(Sequential_Worker, self).__init__()
        self.working_res = working_res
        self.s_d = 512
        self.max_res =max_res

        # row_num == col_num, as these are squares

    def get_layer_rows(self, layer_num, layer_res):
        row_num = layer_res // self.working_res
        layer_row = math.floor(layer_num / row_num)
        layer_col = layer_num % row_num
        return layer_row, layer_col, row_num

    def crop_to_working_area(self, x, layer_row, layer_col):
        return x[:,:,self.working_res*layer_col:self.working_res*(layer_col+1),self.working_res*layer_row:self.working_res*(layer_row+1)]

    def reinsert_work(self, x, out, layer_row, layer_col):
        z = blank_canvas.clone()
        z[:, :, self.working_res * layer_col:self.working_res * (layer_col + 1),
        self.working_res * layer_row:self.working_res * (layer_row + 1)] += out
        return x

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

    def forward(self, x, params, ci, layer_height, num, enc_):
        # x = input in color space
        # out = laplacian (residual) space
        style_encoding,style_projection,down_and_up,adaconvs = params
        layer_res = 512*2**layer_height
        row, col, row_num = self.get_layer_rows(num, layer_res)
        thumb = self.crop_style_thumb(x, layer_res, row, col, row_num)
        thumb = enc_(thumb)['r4_1']
        x = self.resize_to_res(x, layer_res)
        ci = self.resize_to_res(ci,layer_res)
        out = self.crop_to_working_area(x, row, col)
        lap = self.crop_to_working_area(ci, row, col)

        lap = F.conv2d(F.pad(lap, (1,1,1,1), mode='reflect'), weight = lap_weight, groups = 3)
        out = torch.cat([out, lap], dim=1)

        N,C,h,w = thumb.shape
        style = style_encoding(thumb)
        style = style.flatten(1)
        style = style_projection(style)
        style = style.reshape(N, self.s_d, 4, 4)

        for idx, (ada, learnable) in enumerate(zip(adaconvs, down_and_up)):
            if idx > 0:
                out = ada(style, out)
            out = learnable(out)
        out = down_and_up[-1](out)
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
        self.params = nn.ModuleList([nn.ModuleList([*style_encoder_block(s_d), Down_and_Up(),adaconvs(batch_size, s_d)]) for h in range(height)])
        self.layers = module_list_to_momentum_net(nn.ModuleList([Sequential_Worker(self.max_res,256, batch_size, s_d) for i in self.num_layers]),target_device='cuda:0')

    def forward(self, input:torch.Tensor, ci:torch.Tensor, enc_:torch.nn.Module):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        #input = F.interpolate(input, self.max_res, mode='nearest').repeat(1,2,1,1).data.to(torch.device('cuda:0'))
        #input.requires_grad = True
        out = input
        out = F.interpolate(out, self.max_res, mode='nearest')
        for idx, layer in zip(self.num_layers,self.layers):
            height, num = idx
            out = layer(out, self.params[height],ci, height, num, enc_)
        return out
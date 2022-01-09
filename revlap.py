from adaconv import AdaConv
from net import style_encoder_block, ResBlock
from modules import RiemannNoise, PixelShuffleUp, Upblock, Downblock, adaconvs, StyleEncoderBlock
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


blank_canvas = torch.zeros(4,3,512,512)
class Sequential_Worker(nn.Module):
    def __init__(self, max_res,working_res, batch_size,s_d):
        super(Sequential_Worker, self).__init__()
        self.working_res = working_res
        self.s_d = 3
        self.max_res =max_res
        self.style_encoding = nn.Sequential(
            StyleEncoderBlock(self.s_d),
            StyleEncoderBlock(self.s_d),
            StyleEncoderBlock(self.s_d),
            StyleEncoderBlock(self.s_d)
        )
        self.style_projection = nn.Sequential(
            nn.Linear(192, self.s_d * 16),
            nn.ReLU()
        )
        self.downblock = nn.Sequential(*Downblock())


        self.adaconvs = nn.ModuleList(adaconvs(batch_size, s_d=self.s_d))
        self.upblock = nn.ModuleList(Upblock())
        self.lap_weight = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
        self.lap_weight = torch.Tensor(self.lap_weight).to(torch.device('cuda:0'))
        self.lap_weight.requires_grad = False
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
        self.working_res * layer_row:self.working_res * (layer_row + 1)] = out
        return x

    def resize_to_res(self, x, layer_res):
        return F.interpolate(x, layer_res, mode='nearest')

    def crop_style_thumb(self, x, layer_res, row, col, row_num):
        style_col = col if col % 2 == 0 else col - 1
        style_row = row
        if row + 1 > row_num:
            style_row -= 1
        scaled = F.interpolate(x, layer_res//2, mode='nearest')
        scaled = scaled[:,:,self.working_res*style_col:self.working_res*(style_col+1),self.working_res*style_row:self.working_res*(style_row+1)]
        return scaled

    def return_to_full_res(self, x):
        return F.interpolate(x, self.max_res, mode='nearest')

    def forward(self, x, ci, layer_height, num):
        # x = input in color space
        # out = laplacian (residual) space
        layer_res = 512*2**layer_height
        row, col, row_num = self.get_layer_rows(num, layer_res)
        thumb = self.crop_style_thumb(x, layer_res, row, col, row_num)

        x = self.resize_to_res(x, layer_res)
        ci = self.resize_to_res(ci,layer_res)
        out = self.crop_to_working_area(x, row, col)
        lap = self.crop_to_working_area(ci, row, col)

        lap = F.conv2d(F.pad(lap, (1,1,1,1), mode='reflect'), weight = self.lap_weight, groups = 3)
        out = torch.cat([out, lap], dim=1)
        out = self.downblock(out)

        N,C,h,w = thumb.shape
        style = self.style_encoding(thumb)
        style = style.flatten(1)
        style = self.style_projection(style)
        style = style.reshape(N, self.s_d, 4, 4)

        for ada, learnable in zip(self.adaconvs, self.upblock):
            out = ada(style, out, norm=True)
            out = learnable(out)
        out = self.reinsert_work(x, out, row, col)
        out = self.return_to_full_res(out)
        return out


class LayerHolders(nn.Module):
    def __init__(self, max_res: int, working_res: int, layer_num: int, batch_size: int, s_d: int):
        """Uses square-valued resolutions"""
        super(LayerHolders, self).__init__()
        self.max_res = max_res
        self.working_res = working_res
        self.layer_num = layer_num
        self.internal_layer_res = 512*2**layer_num
        self.num_layers_per_side = self.internal_layer_res // 256
        self.module_patches = Sequential_Worker(256, self.internal_layer_res, batch_size,s_d)

    def resize_to_res(self, x):
        return F.interpolate(x, self.internal_layer_res, mode='nearest')

    def return_to_full_res(self, x):
        return F.interpolate(x, self.max_res, mode='nearest')

    def forward(self, x, ci, style):

        #out = self.resize_to_res(x).repeat(1,2,1,1).data
        out = self.resize_to_res(x)
        ci = self.resize_to_res(ci)
        for i in range(self.num_layers_per_side**2):
            out = self.module_patches(out, ci, style, i)
        out = self.return_to_full_res(out)
        print(out.shape)
        return out


class LapRev(nn.Module):
    def __init__(self, max_res, working_res, batch_size, s_d):
        super(LapRev, self).__init__()
        self.max_res = max_res
        self.working_res = working_res
        height = max_res//working_res
        self.num_layers = [(h,i) for h in range(height) for i in range(int((2**h)/.25))]
        self.layers = module_list_to_momentum_net(nn.ModuleList([Sequential_Worker(self.max_res,256, batch_size, s_d) for i in self.num_layers]),target_device='cuda:0')

    def forward(self, input:torch.Tensor, ci:torch.Tensor, style:torch.Tensor):
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
            out = layer(out, ci, height, num)
        return out
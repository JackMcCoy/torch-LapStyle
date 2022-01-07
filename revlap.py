import copy
from adaconv import AdaConv
from net import style_encoder_block, ResBlock
from modules import RiemannNoise, PixelShuffleUp, Upblock, Downblock, adaconvs
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import typing, math
from einops.layers.torch import Rearrange
from revlib.utils import sequential_to_momentum_net

def additive_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return upsample(other_stream),  + fn_out


def additive_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return output - downsample(fn_out)


class RevisorLap(nn.Module):
    def __init__(self, batch_size:int,levels:int= 1):
        super(RevisorLap, self).__init__()
        self.layers = nn.ModuleList([])
        self.levels = levels
        for i in range(levels):
            self.layers.append(RevisionNet(batch_size,  i))
    def forward(self, x: torch.Tensor, ci: torch.Tensor, style: torch.Tensor):
        for layer in self.layers:
            x = layer(x, ci, style)
        return x


class Sequential_Worker(nn.Module):
    def __init__(self, working_res, layer_res, batch_size,s_d, layer_num):
        super(Sequential_Worker, self).__init__()
        self.layer_num = 0
        self.working_res = working_res
        self.layer_res = layer_res
        self.s_d = s_d
        self.layer_num = layer_num
        self.downblock = nn.Sequential(*Downblock())
        self.adaconvs = nn.ModuleList(adaconvs(batch_size, s_d=self.s_d))
        self.upblock = nn.ModuleList(Upblock())
        self.lap_weight = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
        self.lap_weight = torch.Tensor(self.lap_weight).to(torch.device('cuda:0'))
        self.lap_weight.requires_grad = False
        # row_num == col_num, as these are squares

    def get_layer_rows(self, layer_num):
        row_num = self.layer_res // self.working_res
        layer_row = math.ceil(layer_num / row_num) - 1
        layer_col = self.layer_num % row_num
        return layer_row, layer_col

    def crop_to_working_area(self, x, layer_row, layer_col):
        return x[:,:,self.working_res*layer_col:working_res*(layer_col+1),self.working_res*layer_row:self.working_res*(layer_row+1)]

    def reinsert_work(self, x, out, layer_row, layer_col):
        x[:, :, self.working_res * layer_col:working_res * (layer_col + 1),
        self.working_res * layer_row:self.working_res * (layer_row + 1)] = out
        return x

    def forward(self, x, ci, style):
        # x = input in color space
        # out = laplacian (residual) space
        row, col = self.get_layer_rows(self.layer_num)
        out = crop_to_working_area(x, row, col)
        lap = crop_to_working_area(ci, row, col)
        lap = F.conv2d(F.pad(lap, (1,1,1,1), mode='reflect'), weight = self.lap_weight, groups = 3)
        out = torch.cat([out, lap], dim=0)

        out = self.downblock(out)
        for ada, learnable in zip(self.adaconvs, self.upblock):
            out = ada(style, out)
            out = learnable(out)

        out = reinsert_work(x, out, row, col)
        return out


class LayerHolders(nn.Module):
    def __init__(self, max_res: int, working_res: int, layer_num: int, batch_size: int, s_d: int):
        """Uses square-valued resolutions"""
        super(LayerHolders, self).__init__()
        self.max_res = max_res
        self.working_res = working_res
        self.layer_num = layer_num
        self.internal_layer_res = working_res*2**layer_num
        self.num_layers_per_side = self.internal_layer_res // self.working_res
        self.module_patches = sequential_to_momentum_net(nn.Sequential(*[Sequential_Worker(working_res, self.internal_layer_res, batch_size,s_d, i) for i in range(self.num_layers_per_side**2)]),target_device='cuda')

    def resize_to_res(self, x, layer_num):
        intermediate_size = self.working_res*2**layer_num
        return F.interpolate(x, intermediate_size, mode='nearest')

    def return_to_full_res(self, x):
        return F.interpolate(x, self.max_res, mode='nearest')

    def forward(self, x, ci, style):
        out = resize_to_res(x, self.layer_num).repeat(1,2,1,1).to(torch.device('cuda:0'))
        ci = resize_to_res(ci, self.layer_num).to(torch.device('cuda:0'))
        print(x)
        print(ci)
        print(style)
        style = style.to(torch.device('cuda:0'))
        out = self.module_patches(out, ci, style)
        out = self.return_to_full_res(out)
        return out


class LapRev(nn.Module):
    def __init__(self, max_res, working_res, batch_size, s_d):
        super(LapRev, self).__init__()
        self.max_res = max_res
        self.working_res = working_res
        num_layers = max_res//working_res//2
        self.layers = sequential_to_momentum_net(nn.Sequential(*[LayerHolders(max_res, working_res, i, batch_size, s_d) for i in range(num_layers)]), target_device='cuda')

    def forward(self, input:torch.Tensor, ci:torch.Tensor, style:torch.Tensor):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        input = F.interpolate(input, self.max_res, mode='nearest').repeat(1,2,1,1).data.to(torch.device('cuda:0'))
        input.requires_grad = True
        ci = ci.to(torch.device('cuda:0'))
        out = self.layers(input, ci, style)
        return out
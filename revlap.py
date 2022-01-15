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
from weights import upblock_weights, up_adaconv_weights, downblock_weights, adaconv_weight, style_encoder_weights
from shared_modules import AuxLoss, upblock_w_adaconvs, downblock, style_projection
from function import adaptive_instance_normalization as adain

'''
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
'''

def get_mask(inp, height, layer_num):
    N,C,h,w = inp.shape
    side = 2**(height+1)
    x = torch.zeros_like(inp)
    x = x.view(N, C, side, h // side, side, w // side)
    x = torch.permute(x, (0, 2, 4, 1, 3, 5)).reshape(N, -1, C, h // side, w // side)
    x[:,layer_num,:,:,:]+=1
    x = x.reshape((N, side, side, C, h // side, w // side)).permute(0, 3, 1, 4, 2, 5).reshape(N, C, h, w)
    return x

def cropped_coupling_forward(height, layer_num, other_stream: torch.Tensor, fn_out: torch.Tensor):
    fn_out = revlib.core.split_tensor_list(fn_out)

    mask = get_mask(fn_out,height,layer_num)
    if isinstance(fn_out, torch.Tensor):
        return other_stream + (fn_out*mask)

    return [other_stream + (fn_out[0]*mask)] + fn_out[1]

def cropped_coupling_inverse(height, layer_num, output: torch.Tensor, fn_out: torch.Tensor):
    fn_out = revlib.core.split_tensor_list(fn_out)

    mask = get_mask(fn_out,height,layer_num)

    if isinstance(fn_out, torch.Tensor):
        return output - (fn_out * mask)
    return [output - (fn_out[0] * mask)] + fn_out[1]


class Sequential_Worker(nn.Module):
    def __init__(self, init_scale:float, layer_height, num, max_res,working_res, batch_size,s_d):
        super(Sequential_Worker, self).__init__()
        self.init_scale = init_scale
        self.working_res = working_res
        self.s_d = 512
        self.max_res =max_res
        self.layer_height = layer_height
        self.num = num
        self.lap_weight = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
        self.lap_weight = torch.Tensor(self.lap_weight).to(torch.device('cuda'))
        self.upblock_w = upblock_weights()
        self.adaconv_w = up_adaconv_weights(self.s_d)
        self.downblock_w = downblock_weights()

        # row_num == col_num, as these are squares

    def copy(self, layer_num):
        out = copy.copy(self)
        out.num = layer_num
        return out

    def forward(self, x, ci, input):
        # x = input in color space
        # out = laplacian (residual) space

        out = patch_calc(x, ci, input, self.layer_height, self.working_res, self.max_res, self.num,
               self.lap_weight, self.s_d,
               self.downblock_w, self.upblock_w, self.adaconv_w)
        return out

def patch_calc(x, ci, input, layer_height, working_res, max_res, num,
               lap_weight, s_d,
               downblock_w, upblock_w, adaconv_w):
    layer_res = 512*2**layer_height
    if layer_res != max_res:
        x = resize_to_res(x, layer_res, working_res)
        ci = resize_to_res(ci,layer_res, working_res)
    out = crop_to_working_area(x, layer_height, num)
    lap = crop_to_working_area(ci, layer_height, num)
    with torch.no_grad():
        lap = F.conv2d(F.pad(lap, (1,1,1,1), mode='reflect'), weight = lap_weight, groups = 3)
        out = torch.cat([out, lap], dim=1)
    input = F.interpolate(input, 256, mode='nearest')
    out = downblock(out, downblock_w)
    inp_downblock = downblock(input, downblock_w)
    out = adain(out, inp_downblock)
    out = upblock_w_adaconvs(out,style,upblock_w,adaconv_w)

    out = reinsert_work(x, out, layer_height, num)
    if layer_res != max_res:
        out = return_to_full_res(out, max_res)
    return out

def get_layer_rows(layer_res, working_res, num):
    row_num = layer_res // working_res
    layer_row = math.floor(num / row_num)
    layer_col = num % row_num
    return layer_row, layer_col, row_num

def crop_to_working_area(x, height, num):
    N, C, h, w = x.shape
    side = 2 ** (height + 1)
    x = x.view(N, C, side, h // side, side, w // side)
    x = torch.permute(x, (0, 2, 4, 1, 3, 5)).reshape(N, -1, C, h // side, w // side)
    return x[:,num,:,:,:]

def reinsert_work(x, out, height, num):
    N, C, h, w = x.shape
    side = 2 ** (height + 1)
    x = x.view(N, C, side, h // side, side, w // side)
    x = torch.permute(x, (0, 2, 4, 1, 3, 5)).reshape(N, -1, C, h // side, w // side)
    a,b,c,d = out.shape
    y = torch.cat([x[:,:num,:,:,:],out.view(a,1,b,c,d),x[:,num+1:,:,:,:]],1)
    y = y.reshape((N, side, side, C, h // side, w // side)).permute(0, 3, 1, 4, 2, 5).reshape(N, C, h, w)
    return y

def resize_to_res(x, layer_res):
    return F.interpolate(x, layer_res, mode='nearest')

def crop_style_thumb(x, layer_res, row, col, row_num, working_res):
    style_col = col if col % 2 == 0 else col - 1
    style_row = row
    if row + 1 >= row_num:
        style_row -= 1
    scaled = F.interpolate(x, layer_res//2, mode='nearest')
    if layer_res == 512:
        return scaled
    scaled = scaled[:,:,working_res*style_col:working_res*(style_col+1),working_res*style_row:working_res*(style_row+1)]
    return scaled

def return_to_full_res(x, max_res):
    return F.interpolate(x, max_res, mode='nearest')



class LapRev(nn.Module):
    def __init__(self, max_res, working_res, batch_size, s_d, momentumnet_beta):
        super(LapRev, self).__init__()
        self.max_res = max_res
        self.momentumnet_beta = momentumnet_beta
        self.working_res = working_res
        height = max_res//working_res


        self.num_layers = [(h,i) for h in range(height) for i in range(int((2**h)/.25))]
        coupling_forward = [partial(cropped_coupling_forward, h, i) for h, i in self.num_layers]
        coupling_inverse = [partial(cropped_coupling_inverse, h, i) for h, i in self.num_layers]

        cell = Sequential_Worker(1., 0, 0, self.max_res,256, batch_size, s_d)
        self.layers = revlib.ReversibleSequential(*[cell.copy(layer_num) for height, layer_num in self.num_layers],split_dim=0,coupling_forward=coupling_forward,coupling_inverse=coupling_inverse, memory_mode=revlib.core.MemoryModes.autograd_function)

    def forward(self, input:torch.Tensor, ci:torch.Tensor):
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
        out = self.layers(out, ci.data, input,layerwise_args_kwargs=None)

        out = torch.cat([out[:N,:,:,:256],out[N:,:,:,256:]],3)
        return out
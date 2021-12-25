import copy
from adaconv import AdaConv
from net import style_encoder_block, ResBlock
from modules import RiemannNoise
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange


def additive_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return upsample(other_stream),  + fn_out


def additive_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return output - downsample(fn_out)


class RevisorLap(nn.Module):
    def __init__(self, batch_size,levels= 1):
        super(RevisorLap, self).__init__()
        self.layers = nn.ModuleList([])
        self.levels = levels
        for i in range(levels):
            self.layers.append(RevisionNet(batch_size,  i))
    def forward(self, x, ci, style):
        for layer in self.layers:
            x = layer(x, ci, style)
        return x

class RevisionNet(nn.Module):
    def __init__(self, layer_num, batch_size):
        super(RevisionNet, self).__init__()
        self.position_encoding = nn.Embedding(4,4096, max_norm=2)
        self.position_encoding.requires_grad = True
        self.lap_weight = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
        self.lap_weight = torch.Tensor(self.lap_weight).to(torch.device('cuda'))
        self.lap_weight.requires_grad = False
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.downsample = nn.Upsample(scale_factor=.5, mode='bicubic')
        s_d = 256
        self.s_d = 256
        self.content_adaconv = AdaConv(64, 1, batch_size, s_d=s_d)
        self.resblock = ResBlock(64)
        self.rearrange = Rearrange('b c (p1 h) (p2 w) -> (b p1 p2) c h w',p1=2,p2=2)
        self.unarrange = Rearrange('(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1=2, p2=2)
        self.adaconvs = nn.ModuleList([
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(128, 2, batch_size, s_d=s_d),
            AdaConv(128, 2, batch_size, s_d=s_d)])

        self.relu = nn.ReLU()
        self.layer_num = layer_num

        self.DownBlock = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(6, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(),)
        self.UpBlock = nn.ModuleList([nn.Sequential(RiemannNoise(128),
                                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(64, 256, kernel_size=3),
                                                    nn.LeakyReLU(),
                                                    nn.PixelShuffle(2),
                                                    RiemannNoise(256),
                                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(64, 64, kernel_size=3),
                                                    nn.LeakyReLU(),),
                                      nn.Sequential(RiemannNoise(256),
                                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(64, 128, kernel_size=3),
                                                    nn.LeakyReLU(),),
                                      nn.Sequential(RiemannNoise(256),
                                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(128, 128, kernel_size=3),
                                                    nn.LeakyReLU(),),
                                      nn.Sequential(RiemannNoise(256),
                                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(128, 3, kernel_size=3)
                                                    )])


    '''
    def recursive_controller(self, x, ci, thumbnail, enc_):
        holder = []
        base_case = False
        thumbnail_style = None
        if x.shape[-1] == 512:
            base_case = True
            thumbnail_style = self.thumbnail_style_calc(thumbnail, enc_)

        for i, c in zip(x.chunk(2,dim=2), ci.chunk(2,dim=2)):
            for j, c2 in zip(i.chunk(2,dim=3), c.chunk(2,dim=3)):
                if not base_case:
                    holder.append(self.recursive_controller(j, c2, j, enc_))
                else:
                    holder.append(self.generator(j, c2, thumbnail_style))
        holder = torch.cat((torch.cat([holder[0],holder[2]],dim=2),
                            torch.cat([holder[1],holder[3]],dim=2)),dim=3)
        return holder
    '''

    def recursive_controller(self, x, ci, style):
        N,C,h,w = style.shape
        x = self.rearrange(x)
        ci = self.rearrange(ci)
        style = style.view(1,N,C,h,w).expand(4,N,C,h,w)
        style = style.reshape((4*N,C,h,w))
        idx = torch.arange(4).view(4,1).expand(4,N).reshape(N*4).to(torch.device('cuda'))
        out = self.generator(x, ci, style, idx)
        out = self.unarrange(out)
        return out



    def generator(self, x, ci, style, idx):
        ci =  F.conv2d(F.pad(ci.detach(), (1,1,1,1), mode='reflect'), weight = self.lap_weight, groups = 3).to(torch.device('cuda'))
        out = torch.cat([x, ci], dim=1)

        out = self.DownBlock(out)
        out = self.resblock(out)
        N,C,h,w = style.shape
        style = style * self.position_encoding(idx).view(N,C,h,w)
        for adaconv, learnable in zip(self.adaconvs, self.UpBlock):
            out = out + adaconv(style, out, norm=True)
            out = learnable(out)
        out = out + x
        return out

    def forward(self, input, ci, style):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        input = self.upsample(input)
        if ci.shape[-1] != 512:
            scaled_ci = F.interpolate(ci, size=512*2**self.layer_num, mode='bicubic', align_corners=True).detach()
        else:
            scaled_ci = ci
        out = self.recursive_controller(input, scaled_ci, style)
        return out
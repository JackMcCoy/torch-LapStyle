import copy
from adaconv import AdaConv
from net import style_encoder_block, ResBlock
from modules import RiemannNoise
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def additive_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return upsample(other_stream),  + fn_out


def additive_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return output - downsample(fn_out)


class RevisorLap(nn.Module):
    def __init__(self, batch_size,levels= 1):
        super(RevisorLap, self).__init__()
        self.layers = nn.ModuleList([])
        self.levels = levels
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners='True')

        for i in range(levels):
            self.layers.append(RevisionNet(batch_size,  i))

    def forward(self, x, enc_, ci, style):
        for layer in self.layers:
            x = layer(x, enc_, ci, style)
        return x

class RevisionNet(nn.Module):
    def __init__(self, layer_num, batch_size):
        super(RevisionNet, self).__init__()
        self.lap_weight = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
        self.lap_weight = torch.Tensor(self.lap_weight).to(torch.device('cuda'))
        self.lap_weight.requires_grad = False
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.downsample = nn.Upsample(scale_factor=.5, mode='bicubic')
        s_d = 128
        self.s_d = 128
        self.content_adaconv = AdaConv(64, 1, batch_size, s_d=s_d)
        self.resblock = ResBlock(64)
        self.adaconvs = nn.ModuleList([
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(64, 1, batch_size, s_d=s_d),
            AdaConv(128, 2, batch_size, s_d=s_d),
            AdaConv(128, 2, batch_size, s_d=s_d)])

        self.relu = nn.ReLU()
        self.layer_num = layer_num

        self.DownBlock = nn.Sequential(RiemannNoise(256),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),)
        self.UpBlock = nn.ModuleList([nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(6, 256, kernel_size=3, groups=2),
                                                    nn.ReLU(),
                                                    nn.PixelShuffle(2),
                                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(64, 64, kernel_size=3),
                                                    nn.ReLU(),),
                                      nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(64, 128, kernel_size=3),
                                                    nn.ReLU(),),
                                      nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(128, 128, kernel_size=3),
                                                    nn.ReLU(),),
                                      nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
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

    def recursive_controller(self, x, ci, enc_, style):
        holder = []
        for i, c in zip(torch.split(x,512, dim=2), torch.split(ci,512, dim=2)):
            for j, c2 in zip(torch.split(i, 512, dim=3), torch.split(c, 512, dim=3)):
                #thumbnail_style = self.thumbnail_style_calc(j, enc_)
                mini_holder = []
                for s, cs in zip(torch.split(j,256,dim=2),torch.split(c2,256,dim=2)):
                    for s2, cs2 in zip(torch.split(s,256,dim=3),torch.split(cs,256,dim=3)):
                        mini_holder.append((self.generator(s2, cs2, style, enc_)))
                holder.append(torch.cat((torch.cat([mini_holder[0],mini_holder[2]],dim=2),
                            torch.cat([mini_holder[1],mini_holder[3]],dim=2)),dim=3))
        if len(holder)==1:
            return holder[0]
        holder = torch.cat((torch.cat([holder[0], holder[2]], dim=2),
                                 torch.cat([holder[1], holder[3]], dim=2)), dim=3)
        return holder



    def generator(self, x, ci, style, enc_):

        ci =  F.conv2d(F.pad(scaled_ci.detach(), (1,1,1,1), mode='reflect'), weight = self.lap_weight, groups = 3).to(device)
        out = torch.cat([x, ci], dim=1)

        out = self.DownBlock(out)
        out = self.resblock(out)
        for adaconv, learnable in zip(self.adaconvs, self.UpBlock):
            out = out + adaconv(style, out, norm=True)
            out = learnable(out)
        out = out + x
        return out

    def forward(self, input, enc_, ci, style):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        input = self.upsample(input)
        scaled_ci = F.interpolate(ci, size=512*2**self.layer_num, mode='bicubic', align_corners=True).detach()
        out = self.recursive_controller(input, scaled_ci, enc_, style)
        return out
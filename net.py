import torch.nn as nn
import torch

from gaussian_diff import xdog, make_gaussians
from typing import Dict
from function import adaptive_instance_normalization as adain
from function import calc_mean_std
from modules import ResBlock, ConvBlock
from losses import CalcContentLoss, CalcContentReltLoss, CalcStyleEmdLoss, CalcStyleLoss, GramErrors
from vgg import vgg
from vqgan import VQGANLayers

gaus_1, gaus_2, morph = make_gaussians(torch.device('cuda'))

class Encoder(nn.Module):
    def __init__(self, vggs):
        super(Encoder,(self)).__init__()
        enc_layers = list(vggs.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:])

    def forward(self, x, detach_all=False):
        encodings = {}
        detach_if_true = lambda x: x if detach_all == False else x.detach()
        x = self.enc_1(x)
        encodings['r1_1'] = detach_if_true(x)
        x = self.enc_2(x)
        encodings['r2_1'] = detach_if_true(x)
        x = self.enc_3(x)
        encodings['r3_1'] = detach_if_true(x)
        x = self.enc_4(x)
        encodings['r4_1'] = detach_if_true(x)
        x = self.enc_5(x)
        encodings['r5_1'] = detach_if_true(x)
        return encodings

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_1 = nn.Sequential(
            ResBlock(512),
            ConvBlock(512,256))

        self.decoder_2 = nn.Sequential(
            ResBlock(256),
            ConvBlock(256,128)
            )
        self.decoder_3 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64)
            )
        self.decoder_4 = nn.Sequential(
            ConvBlock(64, 64),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=3)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, sF, cF):
        t = adain(cF['r4_1'], sF['r4_1'])
        t = self.decoder_1(t)
        t = self.upsample(t)
        t += adain(cF['r3_1'], sF['r3_1'])
        t = self.decoder_2(t)
        t = self.upsample(t)
        t += adain(cF['r2_1'], sF['r2_1'])
        t = self.decoder_3(t)
        t = self.upsample(t)
        t = self.decoder_4(t)
        return t

class DecoderVQGAN(nn.Module):
    def __init__(self, vgg_path):
        super(DecoderVQGAN, self).__init__()
        self.vqgan = VQGANLayers(vgg_path)
        self.vqgan.train()
        self.decoder_1 = nn.Sequential(
            ResBlock(512),
            ConvBlock(512,256))

        self.decoder_2 = nn.Sequential(
            ResBlock(256),
            ConvBlock(256,128)
            )
        self.decoder_3 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64)
            )
        self.decoder_4 = nn.Sequential(
            ConvBlock(64, 64),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=3)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, sF, cF, ci, si):
        t, l = self.vqgan(ci, si)
        print(l)
        t = self.decoder_1(t)
        t = self.upsample(t)
        t += adain(cF['r3_1'], sF['r3_1'])
        t = self.decoder_2(t)
        t = self.upsample(t)
        t += adain(cF['r2_1'], sF['r2_1'])
        t = self.decoder_3(t)
        t = self.upsample(t)
        t = self.decoder_4(t)
        return t, l


class RevisionNet(nn.Module):
    """RevisionNet of Revision module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self, input_nc=6):
        super(RevisionNet, self).__init__()
        DownBlock = []
        DownBlock += [
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(input_nc, 64, kernel_size=3),
            nn.ReLU()
        ]
        DownBlock += [
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU()
        ]

        self.resblock = ResBlock(64)

        UpBlock = []
        UpBlock += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        ]
        UpBlock += [
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=3)
        ]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, input):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        out = self.DownBlock(input)
        out = self.resblock(out)
        res_block = out.clone()
        out = self.UpBlock(out)
        return out, res_block

class Discriminator(nn.Module):
    def __init__(self, depth, num_channels):
        super(Discriminator, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3,num_channels,3,stride=1,padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2)
            )
        self.body = nn.Sequential()
        for i in range(depth - 2):
            self.body.add_sublayer(
                nn.Conv2D(num_channels,
                          num_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            self.body.add_sublayer(nn.BatchNorm2D(num_channels))
            self.body.add_sublayer(nn.LeakyReLU(0.2))
        self.tail = nn.Conv2D(num_channels,
                              1,
                              kernel_size=3,
                              stride=1,
                              padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


mse_loss = GramErrors()
style_remd_loss = CalcStyleEmdLoss()
content_emd_loss = CalcContentReltLoss()
content_loss = CalcContentLoss()
style_loss = CalcStyleLoss()

def calc_losses(stylized, ci, si, cF, sF, encoder, decoder, calc_identity=True, mdog_losses = True):
    stylized_feats = encoder(stylized)
    if calc_identity==True:
        Icc = decoder(cF,cF)
        l_identity1 = content_loss(Icc, ci)
        Fcc = encoder(Icc)
        l_identity2 = 0
        for key in cF.keys():
            l_identity2 += content_loss(Fcc[key], cF[key])

        Iss = decoder(sF, sF)
        l_identity3 = content_loss(Iss, si)
        Fss = encoder(Iss)
        l_identity4 = 0
        for key in cF.keys():
            l_identity4 += content_loss(Fss[key], sF[key])
    else:
        l_identity1 = None
        l_identity2 = None
        l_identity3 = None
        l_identity4 = None
    loss_c = 0
    for key in cF.keys():
        loss_c += content_loss(stylized_feats[key], cF[key],norm=True)
    loss_s = 0
    for key in sF.keys():
        loss_s += style_loss(stylized_feats[key], sF[key])
    content_emd = content_emd_loss(stylized_feats['r3_1'], cF['r3_1']) +\
        content_emd_loss(stylized_feats['r4_1'], cF['r4_1'])
    style_remd = style_remd_loss(stylized_feats['r3_1'], sF['r3_1']) +\
        style_remd_loss(stylized_feats['r4_1'], sF['r4_1'])

    if mdog_losses:
        cX,_ = xdog(ci.detach(),gaus_1,gaus_2,morph,gamma=.9,morph_cutoff=8.85,morphs=1)
        sX,_ = xdog(si.detach(),gaus_1,gaus_2,morph,gamma=.9,morph_cutoff=8.85,morphs=1)
        cXF = encoder(cX)
        sXF = encoder(sX)
        stylized_dog,_ = xdog(torch.clip(stylized,min=0,max=1),gaus_1,gaus_2,morph,gamma=.9,morph_cutoff=8.85,morphs=1)
        cdogF = encoder(stylized_dog)

        mxdog_content = content_loss(stylized_feats['r3_1'], cXF['r3_1'])+content_loss(stylized_feats['r4_1'], cXF['r4_1'])
        mxdog_content_contraint = content_loss(cdogF['r3_1'], cXF['r3_1'])+content_loss(cdogF['r4_1'], cXF['r4_1'])
        mxdog_style = mse_loss(cdogF['r3_1'],sXF['r3_1']) + mse_loss(cdogF['r4_1'],sXF['r4_1'])
        mxdog_losses = mxdog_content * .3 + mxdog_content_contraint *100 + mxdog_style * 1000
    else:
        mxdog_losses = 0

    return loss_c, loss_s, style_remd, content_emd , l_identity1, l_identity2, l_identity3, l_identity4, mxdog_losses


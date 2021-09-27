import torch.nn as nn
import torch

from gaussian_diff import xdog, make_gaussians
from typing import Dict
from function import adaptive_instance_normalization as adain
from function import calc_mean_std
from modules import ResBlock, ConvBlock
from losses import CalcContentLoss, CalcContentReltLoss, CalcStyleEmdLoss, CalcStyleLoss, GramErrors

vgg = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=1),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, kernel_size=3),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, kernel_size=3),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, kernel_size=3),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, kernel_size=3),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, kernel_size=3),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, kernel_size=3),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU()  # relu5-4
)

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

    def forward(self, x):
        encodings = {}
        x = self.enc_1(x)
        encodings['r1_1'] = x
        x = self.enc_2(x)
        encodings['r2_1'] = x
        x = self.enc_3(x)
        encodings['r3_1'] = x
        x = self.enc_4(x)
        encodings['r4_1'] = x
        x = self.enc_5(x)
        encodings['r5_1'] = x
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

def calc_losses(stylized, ci, si, cF, sF, encoder, decoder, calc_identity=True, calc_mdog = True):
    stylized_feats = encoder(stylized)
    if calc_identity==True:
        Icc = decoder(cF,cF)
        l_identity1 = content_loss(Icc, ci)
        Fcc = encoder(Icc)
        l_identity2 = 0
        for key in cF.keys():
            l_identity2 += content_loss(Fcc[key], cF[key])
    else:
        l_identity1 = None
        l_identity2 = None
    loss_c = 0
    for key in cF.keys():
        loss_c += content_loss(stylized_feats[key], cF[key],norm=True)
    loss_s = 0
    for key in sF.keys():
        loss_s += style_loss(stylized_feats[key], sF[key])
    loss_ss = content_emd_loss(stylized_feats['r3_1'], cF['r3_1']) +\
        content_emd_loss(stylized_feats['r4_1'], cF['r4_1'])
    remd_loss = style_remd_loss(stylized_feats['r3_1'], sF['r3_1']) +\
        style_remd_loss(stylized_feats['r4_1'], sF['r4_1'])

    if mdog_losses:
        cX,_ = xdog(self.ci.detach(),gaus_1,gaus_2,morph,gamma=.9,morph_cutoff=8.85,morphs=1)
        sX,_ = xdog(self.si.detach(),gaus_1,gaus_2,morph,gamma=.9,morph_cutoff=8.85,morphs=1)
        cXF = encoder(cX)
        sXF = encoder(sX)
        stylized_dog,_ = xdog(torch.clip(stylized,min=0,max=1),gaus_1,gaus_2,morph,gamma=.9,morph_cutoff=8.85,morphs=1)
        cdogF = encoder(stylized_dog)

        mxdog_content = calc_content_loss(stylized_feats['r31'], cXF['r31'])+calc_content_loss(stylized_feats['r41'], cXF['r41'])
        mxdog_content_contraint = calc_content_loss(cdogF['r31'], cXF['r31'])+calc_content_loss(cdogF['r41'], cXF['r41'])
        mxdog_style = mse_loss(cdogF['r31'],sXF['r31']) + mse_loss(self.cdogF['r41'],sXF['r41'])
        mxdog_losses = mxdog_content * .3 + mxdog_content_contraint *100 + mxdog_style * 1000
    else:
        mxdog_losses = 0

    return loss_c, loss_s, remd_loss, loss_ss, l_identity1, l_identity2, mxdog_losses


import torch.nn as nn
import torch

from gaussian_diff import xdog, make_gaussians
from typing import Dict
from function import adaptive_instance_normalization as adain
from function import calc_mean_std
from modules import ResBlock, ConvBlock
from losses import GANLoss, CalcContentLoss, CalcContentReltLoss, CalcStyleEmdLoss, CalcStyleLoss, GramErrors
from vgg import vgg
from vqgan import VQGANLayers, VectorQuantize

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

class VQGANTrain(nn.Module):
    def __init__(self, vgg_path):
        super(VQGANTrain, self).__init__()
        self.vqgan = VQGANLayers(vgg_path)
        self.vqgan.train()

    def forward(self, ci, si):
        t, l = self.vqgan(ci, si)
        return t, l

class DecoderVQGAN(nn.Module):
    def __init__(self):
        super(DecoderVQGAN, self).__init__()
        self.quantize_4 = VectorQuantize(16, 3200, transformer_size=1)
        self.quantize_3 = VectorQuantize(32, 1200, transformer_size=2)
        self.quantize_2 = VectorQuantize(64, 1280, transformer_size=3)
        self.decoder_1 = nn.Sequential(
            ResBlock(512),
            ConvBlock(512,256),
            ResBlock(256),
            ConvBlock(256, 256)
        )

        self.decoder_2 = nn.Sequential(
            ResBlock(256),
            ConvBlock(256,128),
            ResBlock(128),
            ConvBlock(128, 128)
            )
        self.decoder_3 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 64)
            )
        self.decoder_4 = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=3)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, sF, cF):
        t = adain(cF['r4_1'], sF['r4_1'])
        t, idx, codebook_loss = self.quantize_4(t)
        t = self.decoder_1(t)
        t = self.upsample(t)
        quantized, idx, cbloss = self.quantize_3(adain(cF['r3_1'], sF['r3_1']))
        codebook_loss += cbloss
        t += quantized
        t = self.decoder_2(t)
        t = self.upsample(t)
        quantized, idx, cbloss = self.quantize_2(adain(cF['r2_1'], sF['r2_1']))
        codebook_loss += cbloss
        t+=quantized
        t = self.decoder_3(t)
        t = self.upsample(t)
        t = self.decoder_4(t)
        return t, codebook_loss


class Discriminator(nn.Module):
    def __init__(self, depth=5, num_channels=64):
        super(Discriminator, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3,num_channels,3,stride=1,padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2)
            )
        self.body = []
        for i in range(depth - 2):
            self.body.append(
                nn.Conv2d(num_channels,
                          num_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            self.body.append(nn.BatchNorm2d(num_channels))
            self.body.append(nn.LeakyReLU(0.2))
        self.body = nn.Sequential(*self.body)
        self.tail = nn.Conv2d(num_channels,
                              1,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.ganloss = GANLoss('lsgan')

    def losses(self, real, fake):
        pred_real = self(real)
        loss_D_real = self.ganloss(pred_real, True)
        pred_fake = self(fake)
        loss_D_fake = self.ganloss(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

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

def identity_loss(i, F, encoder, decoder):
    Icc, cb_loss = decoder(F, F)
    l_identity1 = content_loss(Icc, i)
    Fcc = encoder(Icc)
    l_identity2 = 0
    for key in F.keys():
        l_identity2 += content_loss(Fcc[key], F[key])
    return l_identity1, l_identity2, cb_loss

def calc_losses(stylized, ci, si, cF, sF, encoder, decoder, disc_= None, calc_identity=True, mdog_losses = True, disc_loss=True):
    stylized_feats = encoder(stylized)
    if calc_identity==True:
        l_identity1, l_identity2, cb_loss = identity_loss(ci, cF, encoder, decoder)
        l_identity3, l_identity4, cb = identity_loss(si, sF, encoder, decoder)
        cb_loss += cb
        del(cb)
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
    loss_ss = content_emd_loss(stylized_feats['r3_1'], cF['r3_1']) +\
        content_emd_loss(stylized_feats['r4_1'], cF['r4_1'])
    remd_loss = style_remd_loss(stylized_feats['r3_1'], sF['r3_1']) +\
        style_remd_loss(stylized_feats['r4_1'], sF['r4_1'])

    mxdog_losses = 0
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

    loss_Gp_GAN = 0
    if disc_loss:
        pred_fake_p = disc_(stylized)
        loss_Gp_GAN += disc_.ganloss(pred_fake_p, True)

    return loss_c, loss_s, remd_loss, loss_ss, l_identity1, l_identity2, l_identity3, l_identity4, mxdog_losses, cb_loss, loss_Gp_GAN, cX


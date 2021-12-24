import typing
import torch.nn as nn
import torch
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import crop
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np

from gaussian_diff import xdog, make_gaussians
from function import adaptive_instance_normalization as adain
from modules import ResBlock, ConvBlock, SAFIN, WavePool, WaveUnpool, SpectralResBlock, RiemannNoise
from losses import GANLoss, CalcContentLoss, CalcContentReltLoss, CalcStyleEmdLoss, CalcStyleLoss, GramErrors
from einops.layers.torch import Rearrange
from vqgan import VQGANLayers, Quantize_No_Transformer, TransformerOnly
from linear_attention_transformer import LinearAttentionTransformer as Transformer
from adaconv import AdaConv, KernelPredictor
from vector_quantize_pytorch import VectorQuantize

gaus_1, gaus_2, morph = make_gaussians(torch.device('cuda'))

device = torch.device('cuda')

unfold = torch.nn.Unfold(256,stride=256)

def _l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.normal(mean=0,std=1,shape=(1, W.shape(0)))
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.transpose(1,0)), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, W), eps=1e-12)
    sigma = torch.sum(nn.functional.linear(_u, W.transpose(1, 0)) * _v)
    return sigma, _u


class Encoder(nn.Module):
    def __init__(self, vggs):
        super(Encoder,(self)).__init__()
        enc_layers = list(vggs.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        #self.enc_5 = nn.Sequential(*enc_layers[31:44])

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
        #x = self.enc_5(x)
        #encodings['r5_1'] = x
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
        t = t + adain(cF['r3_1'], sF['r3_1'])
        t = self.decoder_2(t)
        t = self.upsample(t)
        t = t + adain(cF['r2_1'], sF['r2_1'])
        t = self.decoder_3(t)
        t = self.upsample(t)
        t = self.decoder_4(t)
        return t


class SwitchableNoise(nn.Module):
    def __init__(self, size, switch=True):
        super(SwitchableNoise, self).__init__()
        if switch:
            self.noise_or_ident = RiemannNoise(size)
        else:
            self.noise_or_ident = nn.Identity()
    def forward(self, x):
        return self.noise_or_ident(x)


class RevisionNet(nn.Module):
    def __init__(self, s_d = 320, batch_size=8, input_nc=6, first_layer=True):
        super(RevisionNet, self).__init__()

        self.style_encoding = nn.Sequential(
            *style_encoder_block(512),
            *style_encoder_block(512),
            *style_encoder_block(512)
        )
        self.s_d = 128
        self.style_projection = nn.Sequential(
            nn.Linear(8192, self.s_d * 16)
        )
        self.style_riemann_noise = RiemannNoise(4)

        self.resblock = ResBlock(64)
        self.first_layer = first_layer
        self.adaconvs = nn.ModuleList([
            AdaConv(64, 1, s_d=s_d),
            AdaConv(64, 1, s_d=s_d),
            AdaConv(128, 2, s_d=s_d),
            AdaConv(128, 2, s_d=s_d)])

        self.relu = nn.ReLU()
        '''
        self.style_reprojection = nn.Sequential(
            nn.Conv2d(s_d, s_d, kernel_size=1),
            nn.LeakyReLU()
        )
        '''


        self.riemann_style_noise = RiemannNoise(4)
        self.DownBlock = nn.Sequential(SwitchableNoise(256, first_layer),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(6, 128, kernel_size=3),
            nn.ReLU(),
            SwitchableNoise(256, first_layer),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            SwitchableNoise(256, first_layer),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            SwitchableNoise(256, first_layer),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            SwitchableNoise(128, first_layer),)
        self.UpBlock = nn.ModuleList([nn.Sequential(SwitchableNoise(128, first_layer),
                                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(64, 256, kernel_size=3),
                                                    nn.ReLU(),
                                                    SwitchableNoise(128, first_layer),
                                                    nn.PixelShuffle(2),
                                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(64, 64, kernel_size=3),
                                                    nn.ReLU(),
                                                    SwitchableNoise(256, first_layer)),
                                      nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(64, 128, kernel_size=3),
                                                    nn.ReLU(),
                                                    SwitchableNoise(256, first_layer)),
                                      nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(128, 128, kernel_size=3),
                                                    nn.ReLU(),
                                                    SwitchableNoise(256, first_layer)),
                                      nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(128, 3, kernel_size=3)
                                                    )])

    def forward(self, input, stylized_feats):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        b = input.shape[0]
        style = self.style_encoding(stylized_feats['r4_1'])
        style = style.flatten(1)
        style = self.style_projection(style)
        style = style.reshape(b, self.s_d, 4, 4)
        style = self.riemann_style_noise(style)

        out = self.DownBlock(input.clone().detach())
        out = self.resblock(out)
        for adaconv, learnable in zip(self.adaconvs, self.UpBlock):
            out = out + adaconv(style, out, norm=True)
            out = learnable(out)
        out = (out + input.clone().detach()[:,:3,:,:])
        return out

class Revisors(nn.Module):
    def __init__(self, levels= 1, state_string = None, batch_size=8):
        super(Revisors, self).__init__()
        self.layers = nn.ModuleList([])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lap_weight = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
        self.lap_weight = torch.Tensor(self.lap_weight).to(device)
        self.crop = RandomCrop(256)
        self.levels = levels
        for i in range(levels):
            self.layers.append(RevisionNet(s_d=128, first_layer= i == 0, batch_size=batch_size))

    def load_states(self, state_string):
        states = state_string.split(',')
        for idx, i in enumerate(states):
            if idx < len(states)-1:
                self.layers[idx].load_state_dict(torch.load(i))

    def forward(self, input, ci, enc_, crop_marks):
        device = torch.device("cuda")
        idx = 0
        size = 256
        for layer in self.layers:
            stylized_feats = enc_(input)
            input = self.upsample(input)
            size *= 2
            scaled_ci = F.interpolate(ci, size=size, mode='bicubic', align_corners=False)
            size_diff = size//256
            for i in range(idx+1):
                tl = (crop_marks[i][0] * 2**(idx-i)).int()
                tr = (tl + (512*2**(idx-1-i))).int()
                bl = (crop_marks[i][1] * 2**(idx-i)).int()
                br = (bl + (512*2**(idx-1-i))).int()
                scaled_ci = scaled_ci[:, :, tl:tr, bl:br]
                size_diff = size_diff *.5
            patch = input[:, :, tl:tr, bl:br]
            lap_pyr = F.conv2d(F.pad(scaled_ci.detach(), (1,1,1,1), mode='reflect'), weight = self.lap_weight, groups = 3).to(device)
            x2 = torch.cat([patch, lap_pyr], dim = 1)
            #if idx == self.levels-1:
            #    for optimizer in optimizers:
            #        optimizer.zero_grad(set_to_none=True)
            input = layer(x2, stylized_feats)
            idx += 1
        return input, scaled_ci, patch

class SingleTransDecoder(nn.Module):
    def __init__(self):
        super(SingleTransDecoder, self).__init__()
        self.embeddings_set = False
        self.transformer = Transformer(dim = 192,
                                            heads = 32,
                                            depth = 24,
                                            max_seq_len = 256,
                                            shift_tokens = True,
                                            reversible = True,
                                            attend_axially = True,
                                            receives_context = True,
                                            n_local_attn_heads = 32,
                                            local_attn_window_size = 256)
        self.ctx_transformer = Transformer(dim=192,
                                       heads=32,
                                       depth=24,
                                       max_seq_len=256,
                                       shift_tokens=True,
                                       reversible=True,
                                       attend_axially=True,
                                       receives_context=True,
                                       n_local_attn_heads=32,
                                       local_attn_window_size=256
                                    )
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',p1=8,p2=8)
        self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=16,w=16, e=8,d=8)
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
        self.transformer_res = ResBlock(3)
        self.transformer_conv = ConvBlock(3, 3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def set_embeddings(self, b, n, d):
        ones = torch.ones((b, n)).int().to(device)
        seq_length = torch.cumsum(ones, axis=1).to(device)
        self.position_ids = (seq_length - ones).to(device)
        self.ctx_pos_embedding = nn.Embedding(n, d).to(device)
        self.pos_embedding = nn.Embedding(n, d).to(device)
        self.embeddings_set = True

    def forward(self, sF, cF, si, ci):
        t = adain(cF['r4_1'], sF['r4_1'])
        t = self.decoder_1(t)
        t = self.upsample(t)
        t = t + adain(cF['r3_1'], sF['r3_1'])
        t = self.decoder_2(t)
        t = self.upsample(t)
        t = t + adain(cF['r2_1'], sF['r2_1'])
        t = self.decoder_3(t)
        t = self.upsample(t)
        t = self.decoder_4(t)
        transformer = self.rearrange(t)
        b, n, _ = transformer.shape
        if not self.embeddings_set:
            self.set_embeddings(b,n,_)
        position_embeddings = self.pos_embedding(self.position_ids.detach())
        ctx_position_embeddings = self.ctx_pos_embedding(self.position_ids.detach())
        style_rearranged = self.rearrange(si)
        content_rearranged = self.rearrange(ci)
        context = self.ctx_transformer(content_rearranged + ctx_position_embeddings, context = style_rearranged + ctx_position_embeddings)
        transformer = self.transformer(transformer + position_embeddings, context = context + position_embeddings)
        transformer = self.decompose_axis(transformer)
        t = t + transformer.data
        t = self.transformer_res(t)
        t = self.transformer_conv(t)
        return t

class VQGANTrain(nn.Module):
    def __init__(self, vgg_path):
        super(VQGANTrain, self).__init__()
        self.vqgan = VQGANLayers(vgg_path)
        self.vqgan.train()

    def forward(self, ci, si):
        t, l = self.vqgan(ci, si)
        return t, l

def style_encoder_block(ch):
    return [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(ch, ch, kernel_size=3),
        nn.ReLU(),
        nn.AvgPool2d(3, padding=1, stride=2)
    ]

class DecoderAdaConv(nn.Module):
    def __init__(self, batch_size = 8):
        super(DecoderAdaConv, self).__init__()

        self.style_encoding = nn.Sequential(
            *style_encoder_block(512),
            *style_encoder_block(512),
            *style_encoder_block(512)
        )
        self.s_d = 512
        self.style_projection = nn.Sequential(
            nn.Linear(8192, self.s_d*25)
        )
        self.kernel_1 = AdaConv(512, 8, batch_size, s_d = self.s_d)
        self.decoder_1 = nn.Sequential(
            RiemannNoise(32),
            ResBlock(512),
            ConvBlock(512, 256))
        self.kernel_2 = AdaConv(256, 4, batch_size, s_d = self.s_d)
        self.decoder_2 = nn.Sequential(
            RiemannNoise(64),
            ResBlock(256),
            ConvBlock(256, 128)
        )
        self.kernel_3 = AdaConv(128, 2, batch_size, s_d = self.s_d)
        self.decoder_3 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64)
        )
        self.kernel_4 = AdaConv(64, 1, batch_size, s_d = self.s_d)
        self.decoder_4 = nn.Sequential(
            ConvBlock(64, 64),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=3)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
            m.requires_grad = True
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, sF: typing.Dict[str, torch.Tensor], cF: typing.Dict[str, torch.Tensor]):
        b, n, h, w = sF['r4_1'].shape
        adaconv_out = {}
        style = self.style_encoding(sF['r4_1'])
        style = style.flatten(1)
        style = self.style_projection(style)
        style = style.reshape(b, self.s_d, 5, 5)
        adaconv_out['r4_1'] = self.kernel_1(style, cF['r4_1'].detach(), norm=True)
        x = self.decoder_1(adaconv_out['r4_1'])
        x = self.upsample(x)
        adaconv_out['r3_1'] =  self.kernel_2(style, cF['r3_1'].detach(), norm=True)
        x = x + adaconv_out['r3_1']
        x = self.decoder_2(x)
        x = self.upsample(x)
        adaconv_out['r2_1'] = self.kernel_3(style, cF['r2_1'].detach(), norm=True)
        x = x + adaconv_out['r2_1']
        x = self.decoder_3(x)
        x = self.upsample(x)
        adaconv_out['r1_1'] = self.kernel_4(style, cF['r1_1'].detach(), norm=True)
        x = x + adaconv_out['r1_1']
        x = self.decoder_4(x)
        return x, style

class DecoderVQGAN(nn.Module):
    def __init__(self):
        super(DecoderVQGAN, self).__init__()
        rc = dict(receives_ctx=True)

        self.quantize_5 = VectorQuantize(8, 860, transformer_size=0, **rc)
        self.quantize_4 = VectorQuantize(16, 860, transformer_size=1)
        self.quantize_3 = VectorQuantize(32, 860, transformer_size=2)
        self.quantize_2 = VectorQuantize(64, 1280, transformer_size=3)
        self.quantize_1 = VectorQuantize(128, 860, transformer_size=4, **rc)

        self.vit = Transformer(192, 4, 256, 16, 192, shift_tokens=True,
                               reversible=True,
                               n_local_attn_heads=8,
                               local_attn_window_size=256,
                               attend_axially=True,
                               ff_chunks=2)

        patch_height, patch_width = (8,8)
        self.rearrange=Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_height, p2 = patch_width)
        self.decompose_axis=Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=16,d=8,e=8)
        self.to_patch_embedding = nn.Linear(256, 192)

        ones = torch.ones((1, 256)).int().to(device)
        seq_length = torch.cumsum(ones, axis=1)
        self.position_ids = seq_length - ones

        self.pos_embedding = nn.Embedding(256, 192)
        self.transformer_relu = nn.ReLU()
        self.transformer_res = ResBlock(3)
        self.transformer_conv = nn.Sequential(
                                ConvBlock(3, 3),
                                ConvBlock(3, 3),
                                ConvBlock(3, 3)
        )
        self.decoder_0 = nn.Sequential(
            ResBlock(512),
            ConvBlock(512, 512)
        )
        self.decoder_1 = nn.Sequential(
            ResBlock(512),
            ConvBlock(512, 256)
        )

        self.decoder_2 = nn.Sequential(
            ResBlock(256),
            ConvBlock(256,128),
            )
        self.decoder_3 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64)
            )
        self.decoder_4 = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 3),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 3, kernel_size=3)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, sF, cF, train_loop = False):
        t, idx, cb_loss = self.quantize_5(cF['r5_1'], sF['r5_1'], train_loop)
        t = self.decoder_0(t)
        t = self.upsample(t)
        quantized, idx, cb = self.quantize_4(cF['r4_1'], sF['r4_1'], train_loop)
        t += quantized.data
        cb_loss += cb.data
        t = self.decoder_1(t)
        t = self.upsample(t)
        quantized, idx, cb = self.quantize_3(cF['r3_1'], sF['r3_1'], train_loop)
        t += quantized.data
        cb_loss += cb.data
        t = self.decoder_2(t)
        t = self.upsample(t)
        quantized, idx, cb = self.quantize_2(cF['r2_1'], sF['r2_1'], train_loop)
        t += quantized.data
        cb_loss += cb.data
        t = self.decoder_3(t)
        t = self.upsample(t)
        quantized, idx, cb = self.quantize_1(cF['r1_1'], sF['r1_1'], train_loop)
        t += quantized.data
        cb_loss += cb.data
        t = self.decoder_4(t)

        quantized = self.rearrange(t)
        position_embedding = self.pos_embedding(self.position_ids.detach())
        quantized = self.vit(quantized + position_embedding)
        quantized = self.decompose_axis(quantized)
        quantized = self.transformer_res(quantized)
        quantized = self.transformer_conv(quantized)
        t += quantized.data
        return t, cb_loss


class Style_Guided_Discriminator(nn.Module):
    def __init__(self, depth=5, num_channels=64, relgan=True, quantize = False, batch_size=5):
        super(Style_Guided_Discriminator, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3,3,3,stride=1,padding=1, padding_mode='reflect'),
            nn.LeakyReLU(.2),
            nn.Conv2d(3, num_channels, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(.2),
            )
        self.body = nn.ModuleList([])
        self.norms = nn.ModuleList([])
        self.s_d = 128
        self.style_encoding = nn.Sequential(
            nn.Conv2d(self.s_d, self.s_d, kernel_size=1),
            nn.LeakyReLU(.2),
        )

        self.style_projection = nn.Sequential(
            nn.Linear(2048, 2048)
        )


        for i in range(depth - 2):
            self.body.append(AdaConv(64, 1, s_d = self.s_d))
            self.norms.append(
                nn.Sequential(nn.LeakyReLU(.2),
                              nn.Conv2d(64, 64, 3, stride=1, padding=1, padding_mode='reflect'),
                              nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),))
        self.tail = nn.Conv2d(num_channels,
                              1,
                              kernel_size=3,
                              stride=1,
                              padding=1, padding_mode='reflect')
        self.ganloss = GANLoss('lsgan', batch_size=batch_size)
        self.relgan = relgan
        self.quantize = quantize


    def losses(self, real, fake, style):
        b, n, h, w = style.shape

        style = self.style_encoding(style.clone().detach())
        style = self.style_projection(style.flatten(1)).reshape(b, self.s_d, 4, 4)

        pred_real = self(real.detach(), style)
        loss_D_real = self.ganloss(pred_real, True)

        pred_fake = self(fake.detach(), style)

        loss_D_fake = self.ganloss(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return (loss_D, style)

    def forward(self, x, style):
        x = self.head(x)
        for i, norm in zip(self.body,self.norms):
            x = i(style, x, norm=False)
            x = norm(x)
        x = self.tail(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, depth=5, num_channels=64, relgan=True):
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
        self.relgan = relgan
        self.true = torch.Tensor([True]).float().to(device)
        self.true.requires_grad = False
        self.false = torch.Tensor([False]).float().to(device)
        self.false.requires_grad = False

    def losses(self, real, fake):
        idx=0
        pred_fake = self(fake)
        if self.relgan:
            pred_fake = pred_fake.view(-1)
        else:
            loss_D_fake = self.ganloss(pred_fake, self.false)
        for i in torch.split(real.detach(),256,dim=2):
            for j in torch.split(i.detach(), 256,dim=3):
                pred_real = self(j)
                if self.relgan:
                    pred_real = pred_real.view(-1)
                    if idx==0:
                        loss_D = (
                                torch.mean((pred_real - torch.mean(pred_fake) - 1) ** 2) +
                                torch.mean((pred_fake - torch.mean(pred_real) + 1) ** 2)
                        )
                    else:
                        loss_D += (
                                torch.mean((pred_real - torch.mean(pred_fake) - 1) ** 2) +
                                torch.mean((pred_fake - torch.mean(pred_real) + 1) ** 2)
                        ).data
                else:
                    loss_D_real = self.ganloss(pred_real, self.true)
                    if idx ==0:
                        loss_D = ((loss_D_real + loss_D_fake) * 0.5)
                    else:
                        loss_D = loss_D + ((loss_D_real + loss_D_fake) * 0.5)
                idx += 1
        return loss_D

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class SNLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', nn.Parameter(torch.normal(mean=torch.Tensor([0.0]),std=torch.Tensor([1.0]))).to(device))

    @property
    def W_(self):
        w_mat = self.weight.reshape((self.weight.shape[0], -1))
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u=_u
        return self.weight / sigma

    def forward(self, input):
        return nn.functional.linear(input, self.W_, self.bias)


class OptimizedBlock(nn.Module):
    def __init__(self, in_channels: int, dim: int, kernel: int, padding: int, downsample: bool=False):
        super(OptimizedBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, dim, kernel_size=kernel, padding=padding,padding_mode='reflect')
        self.relu = nn.LeakyReLU(0.2)
        self.conv_2 = nn.Conv2d(dim, dim, kernel_size=kernel, padding=padding,padding_mode='reflect')
        self.c_sc = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.downsample = nn.AvgPool2d(2) if downsample else nn.Identity()

    def init_spectral_norm(self):
        self.conv_1 = spectral_norm(self.conv_1)
        self.conv_2 = spectral_norm(self.conv_2)
        self.c_sc = spectral_norm(self.c_sc)

    def forward(self, in_feat):
        x = self.conv_1(in_feat)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.downsample(x)
        shortcut = self.downsample(in_feat)
        shortcut = self.c_sc(shortcut)
        x = x + shortcut
        return x

class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

    def forward(self, input: torch.Tensor):
        for module in self:
            input = module(input)
        return input

class SpectralDiscriminator(nn.Module):
    def __init__(self, depth:int=5, num_channels: int=64, relgan:bool=True, batch_size:int=5):
        super(SpectralDiscriminator, self).__init__()
        ch = num_channels
        self.spectral_gan = nn.ModuleList([OptimizedBlock(3, num_channels, 3, 1, downsample=True),
                                          *[SpectralResBlock(ch*2**i, ch*2**(i+1), 5, 2, downsample=True) for i in range(depth-2)],
                                          SpectralResBlock(ch*2**(depth-1), 3, 3, 1, downsample=False)])

        self.relgan = relgan

    def init_spectral_norm(self):
        for layer in self.spectral_gan:
            layer.init_spectral_norm()

    def forward(self, x):
        for layer in self.spectral_gan:
            x = layer(x)
        return x

mse_loss = GramErrors()
style_remd_loss = CalcStyleEmdLoss()
content_emd_loss = CalcContentReltLoss()
content_loss = CalcContentLoss()
style_loss = CalcStyleLoss()

def identity_loss(i, F, encoder, decoder):
    Icc, ada = decoder(F, F)
    l_identity1 = content_loss(Icc, i)
    with torch.no_grad():
        Fcc = encoder(Icc)
    l_identity2 = 0
    for key in F.keys():
        l_identity2 = l_identity2 + content_loss(Fcc[key], F[key])
    return l_identity1, l_identity2

content_layers = ['r1_1','r2_1','r3_1','r4_1']
style_layers = ['r1_1','r2_1','r3_1','r4_1']
gan_first=True
def calc_GAN_loss(real, fake, disc_, ganloss):
    pred_fake = torch.nan_to_num(disc_(fake))
    if disc_.relgan:
        pred_fake = pred_fake.view(-1)
    else:
        loss_D_fake = ganloss(pred_fake, False)
    pred_real = disc_(real)
    if disc_.relgan:
        pred_real = pred_real.view(-1)
        loss_D = (
                torch.mean((pred_real - torch.mean(pred_fake) - 1) ** 2) +
                torch.mean((pred_fake - torch.mean(pred_real) + 1) ** 2)
        )
    else:
        loss_D_real = torch.nan_to_num(ganloss(pred_real, True))
        loss_D = ((loss_D_real + loss_D_fake) * 0.5)
    return loss_D

def calc_patch_loss(stylized_feats, patch_feats):
    patch_loss = content_loss(stylized_feats['r4_1'], patch_feats['r4_1'])
    return patch_loss

tensor_true = torch.Tensor([True]).to(device)
def calc_losses(stylized, ci, si, cF, encoder, decoder, patch_feats=None, disc_= None, disc_style=None, calc_content_style=True, calc_identity=True, mdog_losses = True, disc_loss=True, content_all_layers=False, remd_loss=True, patch_loss=False, sF=None, GANLoss=None, split_style=False):
    stylized_feats = encoder(stylized)
    if calc_identity==True:
        l_identity1, l_identity2 = identity_loss(ci, cF, encoder, decoder)
        l_identity3, l_identity4 = identity_loss(si, sF, encoder, decoder)
    else:
        l_identity1 = 0
        l_identity2 = 0
        l_identity3 = 0
        l_identity4 = 0
        cb_loss = 0
    if content_all_layers:
        #content_layers = ['r1_1', 'r2_1', 'r3_1', 'r4_1', 'r5_1']
        #style_layers = ['r1_1', 'r2_1', 'r3_1', 'r4_1', 'r5_1']
        loss_c = content_loss(stylized_feats['r1_1'], cF['r1_1'].detach())
        for key in content_layers[1:]:
            loss_c += content_loss(stylized_feats[key], cF[key].detach()).data
    else:
        loss_c = content_loss(stylized_feats['r4_1'], cF['r4_1'].detach(), norm=True)
    if split_style:
        sF = []
        b = si.shape[0]
        patches = unfold(si)
        patches = patches.reshape(-1,b, 3, 256, 256)
        for i in patches:
            sF.append(encoder(i.detach()))
    else:
        sF = [sF]
    for idx, s in enumerate(sF):
        if idx == 0:
            loss_s = style_loss(stylized_feats['r1_1'], s['r1_1'].detach())
        else:
            loss_c = content_loss(stylized_feats['r4_1'], cF['r4_1'].detach(), norm=True)
        for idx, s in enumerate(sF):
            if idx == 0:
                loss_s = style_loss(stylized_feats['r1_1'], s['r1_1'].detach())
            else:
                loss_s = loss_s + style_loss(stylized_feats['r1_1'], s['r1_1'].detach())
            for key in style_layers[1:]:
                loss_s = loss_s + style_loss(stylized_feats[key], s[key].detach())
            if remd_loss:
                if idx == 0:
                    style_remd = style_remd_loss(stylized_feats['r3_1'], s['r3_1'].detach()) + \
                                 style_remd_loss(stylized_feats['r4_1'], s['r4_1'].detach())
                style_remd = style_remd + style_remd_loss(stylized_feats['r3_1'], s['r3_1'].detach()) + \
                             style_remd_loss(stylized_feats['r4_1'], s['r4_1'].detach())
    if remd_loss:
        if content_all_layers:
            content_relt = content_emd_loss(stylized_feats['r3_1'], cF['r3_1'].detach())+content_emd_loss(stylized_feats['r4_1'], cF['r4_1'].detach())
        else:
            content_relt = content_emd_loss(stylized_feats['r4_1'], cF['r4_1'].detach())
    else:
        content_relt = 0
        style_remd = 0
    if mdog_losses:
        cX,_ = xdog(ci.detach(),gaus_1,gaus_2,morph,gamma=.9,morph_cutoff=8.85,morphs=1)
        sX,_ = xdog(si.detach(),gaus_1,gaus_2,morph,gamma=.9,morph_cutoff=8.85,morphs=1)
        cXF = encoder(cX)
        sXF = encoder(sX)
        stylized_dog,_ = xdog(torch.clip(stylized,min=0,max=1),gaus_1,gaus_2,morph,gamma=.9,morph_cutoff=8.85,morphs=1)
        cdogF = encoder(stylized_dog)

        mxdog_content = content_loss(stylized_feats['r4_1'], cXF['r4_1'])
        mxdog_content_contraint = content_loss(cdogF['r4_1'], cXF['r4_1'])
        mxdog_style = mse_loss(cdogF['r3_1'],sXF['r3_1']) + mse_loss(cdogF['r4_1'],sXF['r4_1'])
        mxdog_losses = mxdog_content * .3 + mxdog_content_contraint *100 + mxdog_style * 1000
    else:
        mxdog_losses = 0

    if disc_loss:
        fake_loss = disc_(stylized)
        loss_Gp_GAN = GANLoss(fake_loss, True)
    else:
        loss_Gp_GAN = 0

    if patch_loss:
        patch_loss = content_loss(stylized_feats['r4_1'], patch_feats['r4_1'].detach())
    else:
        patch_loss = 0

    return loss_c, loss_s, content_relt, style_remd, l_identity1, l_identity2, l_identity3, l_identity4, mxdog_losses, loss_Gp_GAN, patch_loss


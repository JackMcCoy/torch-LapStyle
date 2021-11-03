import torch.nn as nn
import torch

from gaussian_diff import xdog, make_gaussians
from function import adaptive_instance_normalization as adain
from modules import ResBlock, ConvBlock, SAFIN, WavePool, WaveUnpool
from losses import GANLoss, CalcContentLoss, CalcContentReltLoss, CalcStyleEmdLoss, CalcStyleLoss, GramErrors
from einops.layers.torch import Rearrange
from vqgan import VQGANLayers, Quantize_No_Transformer, TransformerOnly
from linear_attention_transformer import LinearAttentionTransformer as Transformer
from adaconv import AdaConv
from vector_quantize_pytorch import VectorQuantize

gaus_1, gaus_2, morph = make_gaussians(torch.device('cuda'))

device = torch.device('cuda')

class Encoder(nn.Module):
    def __init__(self, vggs):
        super(Encoder,(self)).__init__()
        enc_layers = list(vggs.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_1_2 = nn.Sequential(*enc_layers[4:7])
        self.enc_2 = nn.Sequential(*enc_layers[7:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])

    def forward(self, x, detach_all=False):
        encodings = {}
        x = self.enc_1(x)
        encodings['r1_1'] = x
        x = self.enc_1_2(x)
        encodings['r1_2'] = x
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
        t = t + adain(cF['r3_1'], sF['r3_1'])
        t = self.decoder_2(t)
        t = self.upsample(t)
        t = t + adain(cF['r2_1'], sF['r2_1'])
        t = self.decoder_3(t)
        t = self.upsample(t)
        t = self.decoder_4(t)
        return t

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
    def __init__(self):
        super(DecoderAdaConv, self).__init__()
        '''
        self.vq = VectorQuantize(
            dim = 16,
            codebook_size = 12000,
            kmeans_init=True,
            kmeans_iters=500,
            use_cosine_sim=False,
            threshold_ema_dead_code=2
        )
        '''
        self.style_encoding = nn.Sequential(
            *style_encoder_block(512),
            *style_encoder_block(512),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=3, groups=512),
            nn.LeakyReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.LeakyReLU(),
        )
        self.s_d = 256
        self.style_projection = nn.Sequential(
            nn.Linear(8192, self.s_d*16)
        )
        self.kernel_1 = AdaConv(512, 8, s_d = self.s_d)
        self.decoder_1 = nn.Sequential(
            ResBlock(512),
            ConvBlock(512, 256))
        self.kernel_2 = AdaConv(256, 4, s_d = self.s_d)
        self.decoder_2 = nn.Sequential(
            ResBlock(256),
            ConvBlock(256, 128)
        )
        self.kernel_3 = AdaConv(128, 2, s_d = self.s_d)
        self.decoder_3 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64)
        )
        self.kernel_4 = AdaConv(64, 1, s_d = self.s_d)
        self.kernel_4_2 = AdaConv(64, 1, s_d=self.s_d)
        self.decoder_4_2 = ConvBlock(64, 64)
        self.decoder_4 = nn.Sequential(
            ConvBlock(64, 64),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=3)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, sF, cF):
        b, n, h, w = sF['r4_1'].shape
        adaconv_out = {}
        style = self.style_encoding(sF['r4_1'].detach())
        #style, indices, commit_loss = self.vq(style.flatten(2).float())
        style = self.style_projection(style.flatten(1))
        style = style.reshape(b, self.s_d, 4, 4)
        adaconv_out['r4_1'] = self.kernel_1(style, cF['r4_1'])
        x = self.decoder_1(adaconv_out['r4_1'])
        x = self.upsample(x)
        adaconv_out['r3_1'] =  self.kernel_2(style, cF['r3_1'])
        x += adaconv_out['r3_1'].data
        x = self.decoder_2(x)
        x = self.upsample(x)
        adaconv_out['r2_1'] = self.kernel_3(style, cF['r2_1'])
        x += adaconv_out['r2_1'].data
        x = self.decoder_3(x)
        x = self.upsample(x)
        adaconv_out['r1_2'] = self.kernel_4_2(style, cF['r1_2'])
        x += adaconv_out['r1_2'].data
        x = self.decoder_4_2(x)
        adaconv_out['r1_1'] = self.kernel_4(style, cF['r1_1'])
        x += adaconv_out['r1_1'].data
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
    def __init__(self, depth=5, num_channels=64, relgan=True):
        super(Style_Guided_Discriminator, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3,num_channels,3,stride=1,padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2)
            )
        self.body = []
        self.norms = []
        for i in range(depth - 2):
            self.body.append(AdaConv(64, 1, s_d = 256))
            self.norms.append(nn.Sequential(
                nn.BatchNorm2d(num_channels),
                nn.LeakyReLU(0.2)
            ))
        self.tail = nn.Conv2d(num_channels,
                              1,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.ganloss = GANLoss('lsgan')
        self.relgan = relgan

    def losses(self, real, fake, style):
        pred_real = self(real, style.float())
        pred_fake = self(fake, style.float())
        if self.relgan:
            pred_real = pred_real.view(-1)
            pred_fake = pred_fake.view(-1)
            loss_D = (
                    torch.mean((pred_real - torch.mean(pred_fake) - 1) ** 2) +
                    torch.mean((pred_fake - torch.mean(pred_real) + 1) ** 2)
            )
        else:
            loss_D_real = self.ganloss(pred_real, True)
            loss_D_fake = self.ganloss(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def forward(self, x, style):
        x = self.head(x)
        for idx, i in enumerate(self.body):
            x = i(style, x, norm=False)
            x = self.norms[idx](x)
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

    def losses(self, real, fake):
        pred_real = self(real)
        pred_fake = self(fake)
        if self.relgan:
            pred_real = pred_real.view(-1)
            pred_fake = pred_fake.view(-1)
            loss_D = (
                    torch.mean((pred_real - torch.mean(pred_fake) - 1) ** 2) +
                    torch.mean((pred_fake - torch.mean(pred_real) + 1) ** 2)
            )
        else:
            loss_D_real = self.ganloss(pred_real, True)
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
    Icc, ada, cb = decoder(F, F)
    l_identity1 = content_loss(Icc, i)
    Fcc = encoder(Icc)
    l_identity2 = 0
    for key in F.keys():
        l_identity2 = l_identity2 + content_loss(Fcc[key], F[key]).data
    return l_identity1, l_identity2

content_layers = ['r1_1','r2_1','r3_1','r4_1']
style_layers = ['r1_1','r1_2','r2_1','r3_1','r4_1']

def calc_losses(stylized, ci, si, cF, sF, encoder, decoder, disc_= None, calc_identity=True, mdog_losses = True, disc_loss=True):
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
    loss_c = content_loss(stylized_feats['r4_1'], cF['r4_1'], norm=True)
    loss_s = style_loss(stylized_feats['r1_1'], sF['r1_1'])
    for key in style_layers[1:]:
        loss_s += style_loss(stylized_feats[key], sF[key]).data
    #content_relt = content_emd_loss(stylized_feats['r3_1'], cF['r3_1']) +\
    #    content_emd_loss(stylized_feats['r4_1'], cF['r4_1'])
    #style_remd = style_remd_loss(stylized_feats['r3_1'], sF['r3_1']) +\
    #    style_remd_loss(stylized_feats['r4_1'], sF['r4_1'])
    #content_relt = 0
    #style_remd = 0
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
        mxdog_losses = mxdog_content * .1 + mxdog_content_contraint *100 + mxdog_style * 1000
    else:
        mxdog_losses = 0

    if disc_loss:
        pred_fake_p = disc_(stylized)
        loss_Gp_GAN = disc_.ganloss(pred_fake_p, True).data
    else:
        loss_Gp_GAN = 0

    return loss_c, loss_s, l_identity1, l_identity2, l_identity3, l_identity4, mxdog_losses, loss_Gp_GAN


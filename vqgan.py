import torch
from torch import nn
import torch.nn.functional as F
from vgg import vgg
from einops.layers.torch import Rearrange
from linear_attention_transformer import LinearAttentionTransformer as Transformer
from losses import CalcContentLoss, CalcStyleLoss, CalcContentReltLoss, CalcStyleEmdLoss
from function import adaptive_instance_normalization as adain

device = torch.device('cuda')
'''
VectorQuantize taken from LucidRains repo (https://github.com/lucidrains/vector-quantize-pytorch)
'''
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


device = torch.device('cuda')

class TransformerOnly(nn.Module):
    def __init__(self, transformer_size = 1):
        super().__init__()
        self.embeddings_set = False
        depth = 2
        if transformer_size == 0:
            self.transformer = Transformer(dim=512,
                                           heads=16,
                                           depth=depth,
                                           max_seq_len=64,
                                           shift_tokens=True,
                                           reversible=True,
                                           attend_axially=True,
                                           receives_context=False)
            self.rearrange = Rearrange('b c h w -> b (h w) c')
            self.decompose_axis = Rearrange('b (h w) c -> b c h w', h=8, w=8)
        if transformer_size == 1:
            self.transformer = Transformer(dim=512,
                                           heads=16,
                                           depth=depth,
                                           max_seq_len=256,
                                           shift_tokens=True,
                                           attend_axially=True,
                                           reversible=True,
                                           receives_context=False)
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=1, p2=1)
            self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)', h=16, w=16, e=1, d=1)
        elif transformer_size==2:
            self.transformer = Transformer(dim = 1024,
                                            heads = 16,
                                            depth = depth,
                                            max_seq_len = 256,
                                            shift_tokens = True,
                                            reversible=True,
                                            attend_axially=True,
                                            receives_context=False)
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',p1=2,p2=2)
            self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=16,w=16, e=2,d=2)
        elif transformer_size==3:
            self.transformer = Transformer(dim = 2048,
                                            heads = 16,
                                            depth = depth,
                                            max_seq_len = 256,
                                            shift_tokens = True,
                                            attend_axially=True,
                                            reversible = True,
                                            receives_context=False)
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',p1=4,p2=4)
            self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=16,w=16, e=4,d=4)
        elif transformer_size==4:
            self.transformer = Transformer(dim = 1024,
                                            heads = 16,
                                            depth = depth,
                                            max_seq_len = 1024,
                                            reversible = True,
                                            attend_axially=True,
                                            shift_tokens = True,
                                            receives_context=False)

            self.rearrange=Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = 4, p2 = 4)
            self.decompose_axis=Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=32,w=32,d=4,e=4)

    def set_embeddings(self, b, n, d):
        ones = torch.ones((b, n)).int().to(device)
        seq_length = torch.cumsum(ones, axis=1).to(device)
        self.position_ids = (seq_length - ones).to(device)
        self.pos_embedding = nn.Embedding(n, d).to(device)
        self.embeddings_set = True

    def forward(self, cF, sF):
        target = adain(cF, sF)

        quantize = self.rearrange(target)
        b, n, _ = quantize.shape

        quantize = self.transformer(quantize)
        quantize = self.decompose_axis(quantize)
        return quantize, None, 0

class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        decay = .8,
        commitment = 1.,
        eps = 1e-5,
        n_embed = None,
        transformer_size = 1,
        receives_ctx = False
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment
        self.perceptual_loss = CalcContentLoss()
        self.style_loss = CalcStyleLoss()

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())
        self.embeddings_set = False
        rc = dict(receives_context=False)
        dropout = dict(attn_layer_dropout = 0.1, attn_dropout = 0.1)

        if transformer_size == 0:
            self.transformer = Transformer(dim=512,
                                           heads=32,
                                           depth=8,
                                           max_seq_len=64,
                                           shift_tokens=True,
                                           reversible=True,
                                           **dropout)
            self.rearrange = Rearrange('b c h w -> b (h w) c')
            self.decompose_axis = Rearrange('b (h w) c -> b c h w', h=8, w=8)
            self.normalize = nn.InstanceNorm2d(512)
        if transformer_size == 1:
            self.transformer = Transformer(dim=512,
                                           heads=32,
                                           depth=8,
                                           max_seq_len=256,
                                           shift_tokens=True,
                                           reversible=True,
                                           **dropout)
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=1, p2=1)
            self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)', h=16, w=16, e=1, d=1)
        elif transformer_size==2:
            self.transformer = Transformer(dim = 256,
                                            heads = 32,
                                            depth = 8,
                                            max_seq_len = 256,
                                            shift_tokens = True,
                                            reversible = True, **dropout)
            self.rearrange = Rearrange('b c h w -> b (h w) c')
            self.decompose_axis = Rearrange('b (h w) c -> b c h w',h=32)
        elif transformer_size==3:
            self.transformer = Transformer(dim = 2048,
                                            heads = 32,
                                            depth = 8,
                                            max_seq_len = 256,
                                            shift_tokens = True,
                                            reversible = True, **dropout)
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',p1=4,p2=4)
            self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=16,w=16, e=4,d=4)
            self.normalize = nn.InstanceNorm2d(128, affine=False)
        elif transformer_size==4:
            self.transformer = Transformer(dim = 1024,
                                            heads = 32,
                                            depth = 8,
                                            max_seq_len = 1024,
                                            reversible = True,
                                            shift_tokens = True, **dropout)

            self.rearrange=Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = 4, p2 = 4)
            self.decompose_axis=Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=32,w=32,d=4,e=4)

    def set_embeddings(self, b, n, d):
        ones = torch.ones((b, n)).int().to(device)
        seq_length = torch.cumsum(ones, axis=1).to(device)
        self.position_ids = (seq_length - ones).to(device)
        self.pos_embedding = nn.Embedding(n, d).to(device)
        self.embeddings_set = True

    @property
    def codebook(self):
        return self.embed.transpose(0, 1)

    def forward(self, cF, sF, train_loop=False):
        target = adain(cF, sF)
        quantize = self.rearrange(target)
        b, n, _ = quantize.shape
        if not self.embeddings_set:
            self.set_embeddings(b, n, _)
        position_embeddings = self.pos_embedding(self.position_ids.detach())
        quantize = quantize + position_embeddings

        quantize = self.transformer(quantize)
        quantize = self.decompose_axis(quantize)
        quantize = target + (quantize - target).detach()

        flatten = quantize.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).float()
        embed_ind = embed_ind.view(*cF.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training and train_loop:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        if train_loop:
            loss = self.perceptual_loss(quantize.detach(), target) * self.commitment
            loss += (self.style_loss(quantize.detach(), target) * 10).data
        else:
            loss = 0

        quantize = target + (quantize.detach() - target)

        return quantize, embed_ind, loss

class Quantize_No_Transformer(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        decay = .8,
        commitment = 1.,
        eps = 1e-5,
        n_embed = None,
        transformer_size = 1,
        receives_ctx = False
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment
        self.perceptual_loss = CalcContentLoss()
        self.style_loss = CalcStyleLoss()

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())
        self.embeddings_set = False
        rc = dict(receives_context=receives_ctx)

    @property
    def codebook(self):
        return self.embed.transpose(0, 1)

    def forward(self, cF, sF):
        target = adain(cF, sF)
        flatten = target.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).float()
        embed_ind = embed_ind.view(*cF.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        loss = self.perceptual_loss(quantize.detach(), target) * self.commitment
        loss += (self.style_loss(quantize.detach(), target) * 5).data

        quantize = target + (quantize.detach() - target)

        return quantize, embed_ind, loss

class VQGANLayers(nn.Module):
    def __init__(self, vgg_state_dict):
        super(VQGANLayers, self).__init__()

        self.pkeep = .8
        self.context_mod = vgg
        self.z_mod = vgg

        self.context_mod.load_state_dict(torch.load(vgg_state_dict))
        self.z_mod.load_state_dict(torch.load(vgg_state_dict))

        self.context_mod = self.context_mod[:31]
        self.z_mod = self.z_mod[:31]

        embed_dim = 16
        z_channels = 512
        codebook_size = 1024
        self.quantize_4_z = VectorQuantize(embed_dim, codebook_size)
        self.quantize_4_s = VectorQuantize(embed_dim, codebook_size)
        self.quant_conv_s = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_z = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.transformer_4 = GPT(codebook_size, 1023, 16, 8, embed_dim)
        self.transformer_4.train()
        self.post_quant_conv = torch.nn.Conv2d(1024, z_channels, 1)

    def forward(self, ci, si, training=True):
        zF = self.z_mod(ci)
        sF = self.context_mod(si)

        zF = self.quant_conv_z(zF)
        sF = self.quant_conv_s(sF)

        quant_z, z_indices, loss1 = self.quantize_4_z(zF)
        z_indices = z_indices.view(quant_z.shape[0], -1)
        quant_s, s_indices, loss2 = self.quantize_4_s(sF)
        s_indices = s_indices.view(quant_s.shape[0], -1)
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep * torch.ones(z_indices.shape,
                                                           device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, 32)
            a_indices = mask * z_indices + (1 - mask) * r_indices
        else:
            a_indices = z_indices
        zs = torch.cat([s_indices, a_indices], axis=1)
        target = z_indices
        logits, _ = self.transformer_4(zs[:, :-1])
        logits = logits[:, s_indices.shape[1] - 1:]

        if training:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        else:
            loss = 0
        logits = logits.transpose(1,2)
        logits = logits.reshape((logits.shape[0], 1024, 16, 16))
        logits = self.post_quant_conv(logits)
        return logits, loss, loss1, loss2

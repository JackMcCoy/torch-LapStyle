import torch
from torch import nn
import torch.nn.functional as F
from vgg import vgg
from einops.layers.torch import Rearrange
from linear_attention_transformer import LinearAttentionTransformer as Transformer
from losses import CalcContentLoss

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

class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        decay = .8,
        commitment = 1.,
        eps = 1e-5,
        n_embed = None,
        transformer_size = 1
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment
        self.perceptual_loss = CalcContentLoss()

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

        if transformer_size==1:
            self.transformer = Transformer(dim = 512,
                                            heads = 32,
                                            depth = 16,
                                            max_seq_len = 256,
                                            shift_tokens = True,
                                            attn_layer_dropout = .1,
                                            attn_dropout = .1,
                                            n_local_attn_heads = 2)
            self.pos_embedding = nn.Embedding(512, 256)
            self.rearrange = Rearrange('b c h w -> b c (h w)')
            self.decompose_axis = Rearrange('b c (h w) -> b c h w',h=dim)
        elif transformer_size==2:
            self.transformer = Transformer(dim = 256,
                                            heads = 32,
                                            depth = 16,
                                            max_seq_len = 1024,
                                            shift_tokens = True,
                                            attn_layer_dropout = .1,
                                            attn_dropout = .1,
                                            n_local_attn_heads = 4)
            self.pos_embedding = nn.Embedding(1024, 256)
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) (h w)',p1=2,p2=2)
            self.decompose_axis = Rearrange('b (c e d) (h w) -> b c (h e) (w d)',h=32,w=32, e=2,d=2)
        elif transformer_size==3:
            self.transformer = Transformer(dim = 256,
                                            heads = 16,
                                            depth = 16,
                                            max_seq_len = 2048,
                                            shift_tokens = True,
                                            attn_layer_dropout = .1,
                                            attn_dropout = .1,
                                            n_local_attn_heads = 8)
            self.pos_embedding = nn.Embedding(2048, 256)
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',p1=4,p2=4)
            self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=16,w=16, e=4,d=4)
        elif transformer_size==4:
            self.transformer = Transformer(dim = 256,
                                            heads = 16,
                                            depth = 8,
                                            max_seq_len = 4096,
                                            shift_tokens = True)
            self.pos_embedding = nn.Embedding(4096, 1024)
            self.rearrange=Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = 2, p2 = 2)
            self.decompose_axis=Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=64,w=64,d=2,e=2)


    @property
    def codebook(self):
        return self.embed.transpose(0, 1)

    def forward(self, input):
        dtype = input.dtype
        print(input.shape)
        quantize = self.rearrange(input)
        b, n, _ = quantize.shape
        b, n, _ = quantize.shape

        ones = torch.ones((b, n)).int().to(device)
        seq_length = torch.cumsum(ones, axis=1)
        position_ids = seq_length - ones
        position_ids.stop_gradient = True
        position_embeddings = self.pos_embedding(position_ids)

        quantize = self.decompose_axis(quantize+ position_embeddings)
        print(quantize.shape)
        quantize = input + (quantize - input).detach()
        flatten = quantize.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        loss = F.mse_loss(quantize.detach(), input) * self.commitment

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


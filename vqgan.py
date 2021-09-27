import torch
from torch import nn
import torch.nn.functional as F
from vgg import vgg
from mingpt import GPT

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

class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        decay = 0.8,
        commitment = 1.,
        eps = 1e-5,
        n_embed = None,
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    @property
    def codebook(self):
        return self.embed.transpose(0, 1)

    def forward(self, input):
        dtype = input.dtype
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        print(embed_ind.shape)
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
        quantize = input + (quantize - input).detach()
        return quantize, embed_ind, loss

class VQGANLayers(nn.Module):
    def __init__(self, vgg_state_dict):
        super(VQGANLayers, self).__init__()

        self.context_mod = vgg
        self.z_mod = vgg

        self.context_mod.load_state_dict(torch.load(vgg_state_dict))
        self.z_mod.load_state_dict(torch.load(vgg_state_dict))

        self.context_mod = self.context_mod[:32]
        self.z_mod = self.z_mod[:32]

        self.quantize_4_z = VectorQuantize(2048, 512)
        self.quantize_4_s = VectorQuantize(2048, 512)
        self.transformer_4 = GPT(1024, 512, 24, 16, 1024)

    def forward(self, ci, si, training=True):
        zF = self.z_mod(ci)
        sF = self.context_mod(si)

        quant_z, z_indices, loss1 = self.quantize_4_z(zF)
        quant_s, s_indices, loss2 = self.quantize_4_s(sF)
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep * torch.ones(z_indices.shape,
                                                           device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask * z_indices + (1 - mask) * r_indices
        else:
            a_indices = z_indices
        zs = paddle.concat([s_indices, a_indices], axis=1)
        target = z_indices
        logits, _ = self.transformer_4(zs[:, :-1])
        logits = logits[:, s_indices.shape[1] - 1:]

        if training:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        else:
            loss = 0
        return logits, target, loss


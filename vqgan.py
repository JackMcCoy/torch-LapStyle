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

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())
        self.embeddings_set = False
        rc = dict(receives_context=receives_ctx)

        if transformer_size == 0:
            self.transformer = Transformer(dim=512,
                                           heads=16,
                                           depth=8,
                                           max_seq_len=64,
                                           shift_tokens=True,
                                           reversible=True,
                                           **rc)
            self.rearrange = Rearrange('b c h w -> b (h w) c')
            self.decompose_axis = Rearrange('b (h w) c -> b c h w', h=8, w=8)
            self.normalize = nn.InstanceNorm2d(512)
        if transformer_size == 1:
            self.transformer = Transformer(dim=512,
                                           heads=16,
                                           depth=8,
                                           max_seq_len=256,
                                           shift_tokens=True,
                                           reversible=True,
                                           **rc)
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=1, p2=1)
            self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)', h=16, w=16, e=1, d=1)
            self.normalize = nn.InstanceNorm2d(512)
        elif transformer_size==2:
            self.transformer = Transformer(dim = 1024,
                                            heads = 16,
                                            depth = 8,
                                            max_seq_len = 256,
                                            shift_tokens = True,
                                            reversible = True, **rc)
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',p1=2,p2=2)
            self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=16,w=16, e=2,d=2)
            self.normalize = nn.InstanceNorm2d(256, affine=False)
        elif transformer_size==3:
            self.transformer = Transformer(dim = 2048,
                                            heads = 16,
                                            depth = 8,
                                            max_seq_len = 256,
                                            shift_tokens = True,
                                            reversible = True, **rc)
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',p1=4,p2=4)
            self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=16,w=16, e=4,d=4)
            self.normalize = nn.InstanceNorm2d(128, affine=False)
        elif transformer_size==4:
            self.transformer = Transformer(dim = 1024,
                                            heads = 16,
                                            depth = 8,
                                            max_seq_len = 1024,
                                            reversible = True,
                                            shift_tokens = True, **rc)

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

    def forward(self, cF, sF):
        target = adain(cF, sF)
        quantize = self.normalize(cF)
        inputs = []
        for i in [quantize, sF]:
            quantize = self.rearrange(i)
            b, n, _ = quantize.shape
            if not self.embeddings_set:
                self.set_embeddings(b, n, _)
            position_embeddings = self.pos_embedding(self.position_ids.detach())
            quantize = quantize + position_embeddings
            inputs.append(quantize)

        quantize = self.transformer(inputs[0],context=inputs[1])
        quantize = self.decompose_axis(quantize)

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

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        loss = self.perceptual_loss(quantize.detach(), target) * self.commitment

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

class LinearAttentionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = None,
        bucket_size = 64,
        causal = False,
        ff_chunks = 1,
        ff_glu = False,
        ff_dropout = 0.,
        attn_layer_dropout = 0.,
        attn_dropout = 0.,
        reversible = False,
        blindspot_size = 1,
        n_local_attn_heads = 0,
        local_attn_window_size = 128,
        receives_context = False,
        attend_axially = False,
        pkm_layers = tuple(),
        pkm_num_keys = 128,
        linformer_settings = None,
        context_linformer_settings = None,
        shift_tokens = False
    ):
        super().__init__()
        assert not (causal and exists(linformer_settings)), 'Linformer self attention layer can only be used for non-causal networks'
        assert not exists(linformer_settings) or isinstance(linformer_settings, LinformerSettings), 'Linformer self-attention settings must be a LinformerSettings namedtuple'
        assert not exists(context_linformer_settings) or isinstance(context_linformer_settings, LinformerContextSettings), 'Linformer contextual self-attention settings must be a LinformerSettings namedtuple'

        if type(n_local_attn_heads) is not tuple:
            n_local_attn_heads = tuple([n_local_attn_heads] * depth)

        assert len(n_local_attn_heads) == depth, 'local attention heads tuple must have the same length as the depth'
        assert all([(local_heads <= heads) for local_heads in n_local_attn_heads]), 'number of local attn heads must be less than the maximum number of heads'

        layers = nn.ModuleList([])

        for ind, local_heads in zip(range(depth), n_local_attn_heads):
            layer_num = ind + 1
            use_pkm = layer_num in cast_tuple(pkm_layers)

            parallel_net = Chunk(ff_chunks, FeedForward(dim), along_dim = 1) if not use_pkm else PKM(dim)

            if not exists(linformer_settings):
                attn = SelfAttention(dim, heads, causal, dim_head = dim_head, blindspot_size = blindspot_size, n_local_attn_heads = local_heads, local_attn_window_size = local_attn_window_size, dropout = attn_layer_dropout, attn_dropout= attn_dropout)
            else:
                attn = LinformerSelfAttention(dim, max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, **linformer_settings._asdict())

            if shift_tokens:
                shifts = (1, 0, -1) if not causal else (1, 0)
                attn, parallel_net = map(partial(PreShiftTokens, shifts), (attn, parallel_net))

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, parallel_net)
            ]))

            if attend_axially:
                layers.append(nn.ModuleList([
                    PreNorm(dim, FoldAxially(local_attn_window_size, SelfAttention(dim, heads, causal, dropout = attn_layer_dropout, attn_dropout= attn_dropout))),
                    PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, glu = ff_glu, dropout= ff_dropout), along_dim = 1))
                ]))

            if receives_context:
                if not exists(context_linformer_settings):
                    attn = SelfAttention(dim, heads, dim_head = dim_head, dropout = attn_layer_dropout, attn_dropout= attn_dropout, receives_context = True)
                else:
                    attn = LinformerSelfAttention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout, **context_linformer_settings._asdict())

                layers.append(nn.ModuleList([
                    PreNorm(dim, attn),
                    PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, glu = ff_glu, dropout= ff_dropout), along_dim = 1))
                ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        axial_layer = ((True, False),) if attend_axially else tuple()
        attn_context_layer = ((True, False),) if receives_context else tuple()
        route_attn = ((True, False), *axial_layer, *attn_context_layer) * depth
        route_context = ((False, False), *axial_layer, *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if receives_context else {}
        attn_route_map = {'input_mask': route_attn, 'pos_emb': route_attn}
        self.layers = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        self.pad_to_multiple = lcm(
            1 if not causal else blindspot_size,
            1 if all([(h == 0) for h in n_local_attn_heads]) else local_attn_window_size
        )

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)
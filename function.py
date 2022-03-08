import torch
import torch.nn as nn
import random
import numpy as np
import math
from losses import calc_mean_std


def crop_mark_extract(num_rev,crop_marks,img,level):
    scale = torch.tensor([[2 ** num_rev / 2 ** i] for i in range(num_rev)])
    scaled_crops = crop_marks*scale
    width = scale[level][0]*128
    tx, ty = scaled_crops[:level+1].sum(0)
    return img[:,:,int(tx):int(tx+width),int(ty):int(ty+width)]


class CartesianGrid(nn.Module):
    """Catersian Grid for 2d tensor.
    The Catersian Grid is a common-used positional encoding in deep learning.
    In this implementation, we follow the convention of ``grid_sample`` in
    PyTorch. In other words, ``[-1, -1]`` denotes the left-top corner while
    ``[1, 1]`` denotes the right-botton corner.
    """

    def forward(self, x, **kwargs):
        assert x.dim() == 4
        return self.make_grid2d_like(x, **kwargs)

    def make_grid2d(self, height, width, num_batches=1, requires_grad=False):
        h, w = height, width
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid_x = 2 * grid_x / max(float(w) - 1., 1.) - 1.
        grid_y = 2 * grid_y / max(float(h) - 1., 1.) - 1.
        grid = torch.stack((grid_x, grid_y), 0)
        grid.requires_grad = requires_grad

        grid = torch.unsqueeze(grid, 0)
        grid = grid.repeat(num_batches, 1, 1, 1)

        return grid

    def make_grid2d_like(self, x, requires_grad=False):
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), requires_grad=requires_grad)

        return grid.to(x)

def whiten(cf):
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    cfv = cf.view(c_channels, -1)  # c x (h x w)

    c_mean = torch.mean(cfv, 1) # perform mean for each row
    c_mean = c_mean.unsqueeze(1).expand_as(cfv) # add dim and replicate mean on rows
    cfv = cfv - c_mean # subtract mean element-wise

    c_covm = torch.mm(cfv, cfv.t()).div((c_width * c_height) - 1)  # construct covariance matrix
    c_u, c_e, c_v = torch.svd(c_covm, some=False) # singular value decomposition

    k_c = c_channels
    for i in range(c_channels):
        if c_e[i] < 0.00001:
            k_c = i
            break
    c_d = (c_e[0:k_c]).pow(-0.5)

    w_step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    w_step2 = torch.mm(w_step1, (c_v[:, 0:k_c].t()))
    whitened = torch.mm(w_step2, cfv)
    return whitened

def positionalencoding2d(d_model, height, width, step = 1):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width, device='cuda')
    s_height = height * step
    s_width = width * step
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., s_width, step).unsqueeze(1)
    pos_h = torch.arange(0., s_height, step).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


@torch.no_grad()
def _clip_gradient(model):
    for p in model.parameters():
        if p.grad is None:
            continue
        g_norm = p.grad.norm(2, 0, True).clamp(min=1e-6)
        p_norm = p.norm(2, 0, True).clamp(min=1e-3)
        grad_scale = (p_norm / g_norm * .01).clamp(max=1)
        p.grad.data.copy_(p.grad * grad_scale)

def setup_torch(seed: int):
    torch._C._debug_set_autodiff_subgraph_inlining(False)  # skipcq: PYL-W0212
    torch._C._set_graph_executor_optimize(True)  # skipcq: PYL-W0212
    torch._C._set_backcompat_broadcast_warn(False)  # skipcq: PYL-W0212
    torch._C._set_backcompat_keepdim_warn(False)  # skipcq: PYL-W0212
    torch._C._set_cudnn_enabled(True)  # skipcq: PYL-W0212
    torch._C._set_mkldnn_enabled(True)  # skipcq: PYL-W0212
    torch._C._set_mkldnn_enabled(True)  # skipcq: PYL-W0212
    torch._C._set_cudnn_benchmark(True)  # skipcq: PYL-W0212
    torch._C._set_cudnn_deterministic(False)  # skipcq: PYL-W0212
    torch._C._set_cudnn_allow_tf32(True)  # skipcq: PYL-W0212
    torch._C._set_cublas_allow_tf32(True)  # skipcq: PYL-W0212
    torch._C._jit_set_inline_everything_mode(True)  # skipcq: PYL-W0212

    torch._C._jit_set_profiling_executor(False)  # skipcq: PYL-W0212
    torch._C._jit_set_profiling_mode(False)  # skipcq: PYL-W0212
    torch._C._jit_override_can_fuse_on_cpu(False)  # skipcq: PYL-W0212
    torch._C._jit_override_can_fuse_on_gpu(True)  # skipcq: PYL-W0212
    torch._C._jit_set_texpr_fuser_enabled(True)  # skipcq: PYL-W0212
    torch._C._jit_set_nvfuser_enabled(False)  # skipcq: PYL-W0212

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


def get_embeddings(pos_embeddings,crop_marks):
    size = 256
    embeddings = []
    for idx in range(len(crop_marks)):
        size *= 2
        tl_sum = crop_marks[:idx+1,0].sum()
        bl_sum = crop_marks[:idx+1,1].sum()
        emb = torch.empty(1, size, size, 16, device='cuda:0')
        emb = pos_embeddings(emb)
        embeddings.append(emb[:, tl_sum.to(dtype=torch.long), bl_sum.to(dtype=torch.long), :])
    return embeddings


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def normalized_feat(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(
        size)) / std.expand(size)
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

def init_weights(net,
                 init_type='normal',
                 init_gain=0.02,
                 distribution='normal'):
    """Initialize network weights.
    Args:
        net (nn.Layer): network to be initialized
        init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float): scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                if distribution == 'normal':
                    torch.nn.init.xavier_normal_(m.weight, gain=init_gain)
                else:
                    torch.nn.init.xavier_uniform_(m.weight, gain=init_gain)

            elif init_type == 'kaiming':
                if distribution == 'normal':
                    torch.nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in')
                else:
                    torch.nn.init.kaiming_uniform_(m.weight, a=0.01, mode='fan_in')
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias,0.01)
        elif classname.find(
                'BatchNorm'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight, 1.0, init_gain)
            torch.nn.init.constant_(m.bias, 0.01)
    net.apply(init_func)

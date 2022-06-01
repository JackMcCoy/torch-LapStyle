import torch
import torch.nn as nn
import torch.nn.functional as F
#from geomloss import SamplesLoss
import math

device = torch.device('cuda')

#sinkhorn_loss = SamplesLoss("sinkhorn", p=1, blur=0.03, scaling=0.9)
maxpool = nn.AdaptiveMaxPool2d(64)

@torch.jit.script
def pairwise_distances_cos(a:torch.Tensor, b:torch.Tensor,eps:float = 1e-5):
    a_n, b_n = a.norm(dim=2,p=2)[:, :, None], b.norm(dim=2,p=2)[:, :, None]
    a_norm = a / torch.clip(a_n, min=eps)
    b_norm = b / torch.clip(b_n, min=eps)
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    sim_mt = 1 - sim_mt
    return sim_mt

@torch.jit.script
def pairwise_cos_self(a:torch.Tensor,eps:float = 1e-5):
    a_n = a.norm(dim=2,p=2)[:, :, None]
    a_norm = a / torch.clip(a_n, min=eps)
    sim_mt = torch.bmm(a_norm, a_norm.transpose(1, 2))
    sim_mt = 1 - sim_mt
    return sim_mt

def pairwise_distances_sq_l2(x, y):
    N,C,*_ = x.shape
    x_norm = (x**2).sum(1).view(N,C,1)
    y_norm = (y**2).sum(1).view(N,C,1)
    y_t = torch.transpose(y, 1, 2)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 1e-5, 1e5)/x.size(1)


def cosd_dist(x):
    M = pairwise_distances_cos(x)
    return M

def euc_dist(x,y):
    M = torch.sqrt(pairwise_distances_sq_l2(x, y))
    return M

def rgb_to_yuv(rgb):
    B,C,h,w = rgb.shape

    rgb = (rgb.view(B, C, -1) * torch.tensor([0.157,0.164,0.159],device='cuda').view(1,3,1)) + torch.tensor([0.339, 0.385, 0.465],device='cuda').view(1,3,1)
    #x_min: torch.Tensor = rgb.min(-1)[0].view(B, C, 1)
    #x_max: torch.Tensor = rgb.max(-1)[0].view(B, C, 1)
    #rgb: torch.Tensor = (rgb - x_min) / (x_max - x_min + 1e-6)

    r: torch.Tensor = rgb[..., 0, :]
    g: torch.Tensor = rgb[..., 1, :]
    b: torch.Tensor = rgb[..., 2, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: torch.Tensor = torch.stack([y, u, v], -2)
    return out

def remd_loss(X,Y):
    b = X.shape[0]
    CX_M = cosd_dist(X, Y)

    m1, m1_inds = CX_M.min(2)
    m2, m2_inds = CX_M.min(1)
    remd, remd_ins = torch.cat([m1.mean(1).view(1,b),m2.mean(1).view(1,b)],dim=0).max(1)
    remd = remd.mean()
    return remd

def add_flips(X):
    #    X = X[:,None,:,:]
    X_flip = torch.flip(X, (1,))
    X = torch.stack((X, X_flip), dim=1)
    return X


def CalcStyleEmdLoss(X, Y):
    """Calc Style Emd Loss.
    """
    X, Y = flatten_and_sample(X,Y, shuffle=False)
    #X = X.flatten(2).transpose(1,2).contiguous()
    #Y = Y.flatten(2).transpose(1,2).contiguous()
    try:
        remd = sinkhorn_loss(X,Y).mean()
    except Exception as e:
        print(e)
        remd=0
    return remd

def CalcStyleEmdNoSample(X, Y):
    """Calc Style Emd Loss.
    """
    CX_M = calc_emd_loss(X, Y)
    m1 = CX_M.amin(dim=2)
    m2 = CX_M.amin(dim=1)
    m = torch.cat([m1.mean(dim=0), m2.mean(dim=0)])
    loss_remd = torch.amax(m)
    return loss_remd

def calc_emd_loss(pred, target):
    """calculate emd loss.

    Args:
        pred (Tensor): of shape (N, C, H, W). Predicted tensor.
        target (Tensor): of shape (N, C, H, W). Ground truth tensor.
    """
    b, _, h, w = pred.shape
    pred = pred.reshape([b, -1, w * h])
    pred_norm = torch.sqrt((pred**2).sum(1).reshape([b, -1, 1]))
    pred = pred.transpose(2,1)
    target_t = target.reshape([b, -1, w * h])
    target_norm = torch.sqrt((target**2).sum(1).reshape([b, 1, -1]))
    similarity = torch.bmm(pred, target_t) / pred_norm / target_norm
    dist = 1. - similarity
    return dist

def flatten_and_sample(X, Y, shuffle=True):
    B,C,h,w = X.shape
    choices = h*w
    if choices > 1024:
        if shuffle:
            r = torch.randperm(choices-1,device='cuda')
            X = X.flatten(2)[:,:,r[:1024]].transpose(1,2).contiguous()
            Y = Y.flatten(2)[:,:,r[:1024]].transpose(1,2).contiguous()
        else:
            r = torch.randint(0,choices-1025,(2,))
            X = X.flatten(2)[:, :, r[0]:r[0]+1024].transpose(1, 2).contiguous()
            Y = Y.flatten(2)[:, :, r[1]:r[1]+1024].transpose(1, 2).contiguous()
    else:
        X = X.flatten(2).transpose(1, 2).contiguous()
        Y = Y.flatten(2).transpose(1, 2).contiguous()
    return X, Y

def CalcContentReltLoss(X,Y, eps=1e-5):
    #X = X.flatten(2).transpose(1,2)
    #Y = Y.flatten(2).transpose(1,2)
    X, Y = flatten_and_sample(X,Y)
    # Relaxed EMD
    Mx = cosd_dist(X)
    Mx = Mx / (Mx.sum(1, keepdim=True)+eps)

    My = cosd_dist(Y)
    My = My / (My.sum(1, keepdim=True)+eps)

    d = torch.abs(Mx - My).mean(1) * X.size(1)
    d = d.mean()
    return d

def CalcContentReltNoSample(X,Y, eps=1e-5):
    dM = 1.
    Mx = calc_emd_loss(X, X)
    Mx = Mx / (Mx.sum(1, keepdim=True)+eps)
    My = calc_emd_loss(Y, Y)
    My = My / (My.sum(1, keepdim=True)+eps)
    loss_content = torch.abs(
        dM * (Mx - My)).mean() * Y.shape[2] * Y.shape[3]
    return loss_content

def pixel_loss(X, Y):
    #pred = rgb_to_yuv(pred.flatten(2)[:,:,r[:1024]]).transpose(1,2)
    #target = rgb_to_yuv(target.flatten(2)[:,:,r[:1024]]).transpose(1,2)
    N,C,h,w = X.shape
    # flatten and convert with rgb_to_yuv
    X = F.avg_pool2d(X, kernel_size=4, stride=4).flatten(2).transpose(1,2).contiguous()
    Y = F.avg_pool2d(Y, kernel_size=4, stride=4).flatten(2).transpose(1,2).contiguous()

    try:
        remd = sinkhorn_loss(X, Y).mean()
    except:
        remd = 0
    return remd

class CalcContentLoss():
    """Calc Content Loss.
    """
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def no_norm(self, pred, target):
        return self.mse_loss(pred, target)

    def __call__(self, pred, target):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            norm(Bool): whether use mean_variance_norm for pred and target
        """

        return self.mse_loss(mean_variance_norm(pred),
                                 mean_variance_norm(target))


class CalcStyleLoss():
    """Calc Style Loss.
    """
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, pred, target):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        pred_mean, pred_std = calc_mean_std(pred)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(pred_mean, target_mean) + self.mse_loss(
            pred_std, target_std)


class CharbonnierLoss():
    """Charbonnier Loss (L1).
    Args:
        eps (float): Default: 1e-12.
    """
    def __init__(self, eps=1e-12, reduction='sum'):
        self.eps = eps
        self.reduction = reduction

    def __call__(self, pred, target, **kwargs):
        """Forward Function.
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        if self.reduction == 'sum':
            out = torch.sqrt((pred - target)**2 + self.eps).sum()
        elif self.reduction == 'mean':
            out = torch.sqrt((pred - target)**2 + self.eps).mean()
        else:
            raise NotImplementedError('CharbonnierLoss %s not implemented' %
                                      self.reduction)
        return out


class EdgeLoss():
    def __init__(self):
        k = torch.tensor([[.05, .25, .4, .25, .05]],device=device)
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).tile([3, 1, 1, 1])
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, [kw // 2, kh // 2, kw // 2, kh // 2], mode='reflect')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        with torch.no_grad():
            new_filter[:, :, ::2, ::2] = down * 4  # upsample
            filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def __call__(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self,
                 gan_mode, depth=5, conv_ch=64, batch_size=6):
        """ Initialize the GANLoss class.

        Args:
            gan_mode (str): the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool): label for a real image
            target_fake_label (bool): label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        # when loss weight less than zero return None

        target_real_label = 1.0
        target_fake_label = 0.0
        loss_weight = 1.0

        if loss_weight <= 0:
            return None

        #c = int(conv_ch*2**(depth-2))
        #h = int(256/2**(depth-1))
        c = 1
        self.loss_weight = loss_weight

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)


    def forward(self,
                 prediction,
                 target_is_real: bool):
        """Calculate loss given Discriminator's output and grount truth labels.

        Args:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
            is_updating_D (bool)  - - if we are in updating D step or not

        Returns:
            the calculated loss.
        """
        if target_is_real:
            target_tensor = torch.ones_like(prediction)
        else:
            target_tensor = torch.zeros_like(prediction)
        loss = self.loss(prediction, target_tensor.detach())
        return loss

class GramErrors():
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def gram_matrix(self,input):
        a, b, c, d = input.shape

        features = input.reshape((a * b, c * d))  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product


        return G/(a * b * c * d)

    def __call__(self, pred, target):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        gram_pred = self.gram_matrix(pred)
        gram_target = self.gram_matrix(target)
        return self.mse_loss(gram_pred, gram_target)

def mean_variance_norm(feat):
    """mean_variance_norm.

    Args:
        feat (Tensor): Tensor with shape (N, C, H, W).

    Return:
        Normalized feat with shape (N, C, H, W)
    """
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean) / (std+1e-5)
    return normalized_feat

def calc_mean_std(feat, eps=1e-5):
    """calculate mean and standard deviation.

    Args:
        feat (Tensor): Tensor with shape (N, C, H, W).
        eps (float): Default: 1e-5.

    Return:
        mean and std of feat
        shape: [N, C, 1, 1]
    """
    size = feat.shape
    N, C = size[:2]
    feat_var,feat_mean = torch.var_mean(feat.view(N, C, -1),unbiased=True,dim=2)
    feat_mean = feat_mean.view(N,C,1,1)
    feat_std = (feat_var+eps).sqrt().view(N, C, 1, 1)
    return feat_mean, feat_std

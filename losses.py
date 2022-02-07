import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmatsqrt import MPA_Lya
device = torch.device('cuda')

FastMatSqrt=MPA_Lya.apply



@torch.jit.script
def pairwise_distances_cos(a:torch.Tensor, b:torch.Tensor,eps:float = 1e-5):
    a_n, b_n = a.norm(dim=2,p=2)[:, :, None], b.norm(dim=2,p=2)[:, :, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    sim_mt = 1 - sim_mt
    return sim_mt

def pairwise_distances_sq_l2(x, y):
    N,C,*_ = x.shape
    x_norm = (x**2).sum(1).view(N,C,1)
    y_norm = (y**2).sum(1).view(N,C,1)
    y_t = torch.transpose(y, 1, 2)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 1e-5, 1e5)/x.size(1)


def cosd_dist(x,y):
    M = pairwise_distances_cos(x, y)
    return M

def euc_dist(x,y):
    M = torch.sqrt(pairwise_distances_sq_l2(x, y))
    return M

def rgb_to_yuv(rgb):
    C = torch.tensor([[0.577350,0.577350,0.577350],[-0.577350,0.788675,-0.211325],[-0.577350,-0.211325,0.788675]],device='cuda').view(1,3,3).expand(rgb.shape[0],3,3)
    yuv = torch.bmm(C,rgb)
    return yuv

def remd_loss(X,Y):
    b = X.shape[0]
    CX_M = cosd_dist(X, Y)

    m1, m1_inds = CX_M.min(2)
    m2, m2_inds = CX_M.min(1)
    remd, remd_ins = torch.cat([m1.mean(1).view(1,b),m2.mean(1).view(1,b)],dim=0).max(1)
    remd = remd.mean()
    return remd

def CalcStyleEmdLoss(X, Y):
    """Calc Style Emd Loss.
    """
    b,d = X.shape[:2]
    X = X.flatten(2).transpose(1,2)
    Y = Y.flatten(2).transpose(1,2)

    remd = remd_loss(X,Y)
    return remd

cosinesimilarity = nn.CosineSimilarity()

def calc_emd_loss(pred, target):
    """calculate emd loss.

    Args:
        pred (Tensor): of shape (N, C, H, W). Predicted tensor.
        target (Tensor): of shape (N, C, H, W). Ground truth tensor.
    """

    similarity = cosinesimilarity(pred, target)
    dist = 1. - similarity
    return dist

def CalcContentReltLoss(X,Y, eps=1e-5):
    loss = 0.
    d = X.shape[1]
    X = X.flatten(2).transpose(1,2)
    Y = Y.flatten(2).transpose(1,2)
    # Relaxed EMD
    Mx = cosd_dist(X, X)
    Mx = Mx / Mx.sum(1, keepdim=True)

    My = cosd_dist(Y, Y)
    My = My / My.sum(1, keepdim=True)

    d = torch.abs(Mx - My).mean(1) * X.size(1)
    d = d.mean()
    return d

def pixel_loss(pred, target):
    pred = rgb_to_yuv(pred.flatten(2)).transpose(1,2)
    target = rgb_to_yuv(target.flatten(2)).transpose(1,2)
    remd = remd_loss(pred,target)
    return remd

class CalcContentLoss():
    """Calc Content Loss.
    """
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, pred, target, norm=False):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            norm(Bool): whether use mean_variance_norm for pred and target
        """
        if (norm == False):
            return self.mse_loss(pred, target)
        else:
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
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

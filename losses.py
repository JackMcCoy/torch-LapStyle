import torch.nn as nn
import torch
device = torch.device('cuda')

def pairwise_distances_cos(x, y):
    x_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y**2).sum(1).view(1, -1))
    dist = 1.-torch.mm(x, y_t)/x_norm/y_norm
    return dist

def pairwise_distances_sq_l2(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 1e-5, 1e5)/x.size(1)

def distmat(x, y, cos_d=True):
    if cos_d:
        M = pairwise_distances_cos(x, y)
    else:
        M = torch.sqrt(pairwise_distances_sq_l2(x, y))
    return M

def rgb_to_yuv(rgb):
    C = torch.tensor([[0.577350,0.577350,0.577350],[-0.577350,0.788675,-0.211325],[-0.577350,-0.211325,0.788675]]).to(rgb.device)
    yuv = torch.mm(C,rgb)
    return yuv


class CalcStyleEmdLoss():
    """Calc Style Emd Loss.
    """
    def __init__(self):
        super(CalcStyleEmdLoss, self).__init__()

    @torch.jit.script
    def __call__(self, X, Y):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        d = X.shape[1]

        if d == 3:
            X = rgb_to_yuv(X.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
            Y = rgb_to_yuv(Y.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        else:
            X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
            Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

        # Relaxed EMD
        CX_M = distmat(X, Y, cos_d=True)

        if d == 3: CX_M = CX_M + distmat(X, Y, cos_d=False)

        m1, m1_inds = CX_M.min(1)
        m2, m2_inds = CX_M.min(0)

        remd = torch.max(m1.mean(), m2.mean())
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

class CalcContentReltLoss():
    """Calc Content Relt Loss.
    """
    def __init__(self, eps=1e-5):
        super(CalcContentReltLoss, self).__init__()
        self.eps = eps

    @torch.jit.script
    def __call__(self, X, Y):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        loss = 0.
        d = X.shape[1]
        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        X = X.squeeze().t()
        Y = Y.squeeze().t()

        mu_x = torch.mean(X, 0, keepdim=True)
        mu_y = torch.mean(Y, 0, keepdim=True)
        mu_d = torch.abs(mu_x - mu_y).mean()

        #if 1 in moments
        # print(mu_x.shape)
        loss = loss + mu_d

        #if 2 in moments
        X_c = X - mu_x
        Y_c = Y - mu_y
        X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0] - 1)
        Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0] - 1)

        # print(X_cov.shape)
        # exit(1)

        D_cov = torch.abs(X_cov - Y_cov).mean()
        loss = loss + D_cov
        return loss


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

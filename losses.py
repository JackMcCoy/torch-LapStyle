import torch.nn as nn
import torch
device = torch.device('cuda')
class CalcStyleEmdLoss():
    """Calc Style Emd Loss.
    """
    def __init__(self):
        super(CalcStyleEmdLoss, self).__init__()

    def __call__(self, pred, target):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        CX_M = calc_emd_loss(pred, target)
        m1, _ = CX_M.min(2)
        m2, _ = CX_M.min(1)
        loss_remd = torch.max(torch.mean(m1),torch.mean(m2))
        return loss_remd

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

    def __call__(self, pred, target):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        dM = 1.
        Mx = 1-calc_emd_loss(pred, pred.transpose(3,2))
        Mx = torch.nan_to_num(Mx / (Mx.sum(dim=(1,2), keepdim=True)+self.eps))
        My = 1-calc_emd_loss(target, target.transpose(3,2))
        My = torch.nan_to_num(My / (My.sum(dim=(1,2), keepdim=True)+self.eps))
        loss_content = torch.abs(
            (My.mean(dim=1)-Mx.mean(dim=1))).sum() * 1/(pred.shape[2] * pred.shape[3])**2
        return loss_content


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
            return self.mse_loss(torch.nan_to_num(pred), torch.nan_to_num(target))
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
                 gan_mode, depth=5, conv_ch=64, batch_size=5):
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
        c = 3
        h = 32
        self.target_real = torch.ones(batch_size,c,h,h).to(torch.device('cuda'))
        self.target_fake = torch.zeros(batch_size,c,h,h).to(torch.device('cuda'))
        self.loss_weight = loss_weight

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)


    def __call__(self,
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
            target_tensor = self.target_real
        else:
            target_tensor = self.target_fake
        print(prediction)
        print(prediction.shape)
        print(target_tensor.shape)
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
        gram_pred = torch.clip(self.gram_matrix(pred), min = -1, max = 1)
        gram_target = torch.clip(self.gram_matrix(target), min = -1, max = 1)
        return torch.clip(self.mse_loss(gram_pred, gram_target), min = -1, max = 1)

def mean_variance_norm(feat):
    """mean_variance_norm.

    Args:
        feat (Tensor): Tensor with shape (N, C, H, W).

    Return:
        Normalized feat with shape (N, C, H, W)
    """
    size = feat.shape
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
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
    feat_var = feat.reshape([N, C, -1])
    feat_var = torch.var(feat_var, dim = 2) + eps
    feat_std = torch.sqrt(feat_var)
    feat_std = feat_std.reshape([N, C, 1, 1])
    feat_mean = feat.reshape([N, C, -1])
    feat_mean = feat_mean.mean(2)
    feat_mean = feat_mean.reshape([N, C, 1, 1])
    return feat_mean, feat_std

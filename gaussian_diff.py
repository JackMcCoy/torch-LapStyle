import torch
import numpy as np
import math

def gaussian(kernel_size, sigma,channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.expand((kernel_size,kernel_size))
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], axis=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., axis=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.reshape((1,1, kernel_size, kernel_size))

    return gaussian_kernel

def xdog(im, g, g2,morph_conv,gamma=.94, phi=50, eps=-.5, morph_cutoff=8.88,morphs=1,minmax=False):
    # Source : https://github.com/CemalUnal/XDoG-Filter
    # Reference : XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization
    # Link : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.151&rep=rep1&type=pdf
    #imf1 = paddle.concat(x=[g(paddle.unsqueeze(im[:,0,:,:].detach(),axis=1)),g(paddle.unsqueeze(im[:,1,:,:].detach(),axis=1)),g(paddle.unsqueeze(im[:,2,:,:].detach(),axis=1))],axis=1)

    imf2=torch.zeros_like(im)
    imf1=torch.zeros_like(im)
    imf1.stop_gradient=True
    imf2.stop_gradient=True
    imf2=g2(im)
    imf1=g(im)
    #imf2 = g2(im.detach())
    imdiff = imf1 - gamma * imf2
    imdiff = (imdiff < eps).float() * 1.0  + (imdiff >= eps).float() * (1.0 + torch.tanh(phi * imdiff))
    if type(minmax)==bool:
        min, _ = imdiff.min(axis=3,keepdim=True)
        max, _ = imdiff.max(axis=3,keepdim=True)
        min, _ = min.min(axis=2,keepdim=True)
        max, _ = max.max(axis=2, keepdim=True)
    else:
        min=minmax[0]
        max=minmax[1]
    imdiff -= min.expand_as(imdiff)
    imdiff /= max.expand_as(imdiff)
    if type(minmax)==bool:
        mean = imdiff.mean(axis=[2,3],keepdim=True)
    else:
        mean=minmax[2]
    exmean=mean.expand_as(imdiff)
    for i in range(morphs):
        morphed=morph_conv(imdiff)
        morphed.stop_gradient=True
        passedlow= torch.multiply((imdiff>= exmean).float(),(morphed>= morph_cutoff).float())
    for i in range(morphs):
        passed = morph_conv(passedlow)
        passed= (passed>0).float()
    return passed, [min,max,mean]

def make_gaussians(device):
    symm_gauss_1 = np.repeat(gaussian(11, 1).numpy(), 3, axis=0)
    symm_gauss_2 = np.repeat(gaussian(21, 3).numpy(), 3, axis=0)
    gaussian_filter = torch.nn.Conv2d(3, 3, 11,
                            groups=3, bias=False, stride=1,
                            padding=5, padding_mode='reflect'
                            ).to(device)
    gaussian_filter.weight = torch.nn.parameter.Parameter(torch.Tensor(symm_gauss_1).to(device),requires_grad=False)
    gaussian_filter2 = torch.nn.Conv2d(3, 3, 21,
                            groups=3, bias=False, stride = 1,
                            padding=10, padding_mode='reflect'
                            ).to(device)
    gaussian_filter2.weight = torch.nn.parameter.Parameter(torch.Tensor(symm_gauss_2).to(device),requires_grad=False)
    morph_conv = torch.nn.Conv2d(3, 3, 3, stride= 1, padding=1, groups=3,
                                           padding_mode='reflect', bias=False,
                                           ).to(device)
    torch.nn.init.constant_(morph_conv.weight,1)
    return gaussian_filter, gaussian_filter2, morph_conv
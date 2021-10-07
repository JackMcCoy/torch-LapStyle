import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image
import re, os
import math
import vgg
import net
from function import init_weights
from net import calc_losses
from sampler import InfiniteSamplerWrapper
from functools import partial
from collections import OrderedDict
import numpy as np

torch.autograd.set_detect_anomaly(True)

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform(load_size, crop_size):
    transform_list = [
        transforms.Resize(size=(load_size, load_size)),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        if os.path.isdir(root):
            self.root = root
            self.paths = list(Path(self.root).glob('*'))
        else:
            self.paths = [root]
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

def set_requires_grad(nets, requires_grad=False):
    for param in nets.parameters():
        param.trainable = requires_grad

def adjust_learning_rate(optimizer, iteration_count,args):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--train_model', type=str, default='drafting')
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--load_size', type=int, default=128)
parser.add_argument('--crop_size', type=int, default=128)

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

vgg = vgg.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children()))

if args.train_model=='drafting':

    enc_ = net.Encoder(vgg)
    set_requires_grad(enc_, False)
    dec_ = net.Decoder()
    init_weights(dec_)
    dec_.train()
    enc_.to(device)
    dec_.to(device)

    content_tf = train_transform(args.load_size, args.crop_size)
    style_tf = train_transform(args.load_size, args.crop_size)

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    optimizer = torch.optim.Adam(dec_.parameters(), lr=args.lr)
    for i in tqdm(range(args.max_iter)):
        adjust_learning_rate(optimizer, i,args)
        ci = next(content_iter).to(device)
        si = next(style_iter).to(device)
        cF = enc_(ci, detach_all=True)
        sF = enc_(si, detach_all=True)
        stylized = dec_(sF, cF)
        optimizer.zero_grad()
        losses = calc_losses(stylized, ci, si, cF, sF, enc_, dec_, calc_identity=True)
        loss_c, loss_s, loss_r, loss_ss, l_identity1, l_identity2, l_identity3, l_identity4, mdog = losses
        loss = loss_c * args.content_weight + loss_s * args.style_weight +\
                    l_identity1 * 50 + l_identity2 * 1 +\
                    loss_r * 16 + 10*loss_ss + mdog
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(loss.item())
            print(f'c: {loss_c.item():.3f} s: {loss_s.item():.3f} \
            r: {loss_r.item():.3f} ss: {loss_ss.item():.3f} \
            id1: {l_identity1.item():.3f} id2: {l_identity2.item():.3f} id3: {l_identity3.item():.3f} id4: {l_identity4.item():.3f} \
            mdog: {mdog.item():.3f}')

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)

        if (i + 1) % 100 == 0:
            stylized = stylized.to('cpu')
            for j in range(1):
                save_image(stylized[j], args.save_dir+'/drafting_training_'+str(j)+'_iter'+str(i+1)+'.jpg')

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            print(loss)
            state_dict = dec_.state_dict()
            torch.save(state_dict, save_dir /
                       'decoder_iter_{:d}.pth.tar'.format(i + 1))
    writer.close()

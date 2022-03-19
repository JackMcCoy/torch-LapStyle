import argparse
from pathlib import Path
from revlap import LapRev

import warnings
import copy
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFile
import wandb
import torchvision
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import re, os
import math
import vgg
import net
import random
from function import crop_mark_extract, _clip_gradient, setup_torch, init_weights, PositionalEncoding2D, get_embeddings
from losses import GANLoss
from modules import RiemannNoise
from net import calc_losses, calc_patch_loss, calc_GAN_loss, calc_GAN_loss_from_pred
from sampler import InfiniteSamplerWrapper, SequentialSamplerWrapper, SimilarityRankedSampler
from torch.cuda.amp import autocast, GradScaler
from function import CartesianGrid as Grid
from randaugment import RandAugment

#setup_torch(0)
#warnings.simplefilter("ignore")
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True
#torchvision.set_image_backend('accimage')
ac_enabled = False


invTrans = transforms.Compose([transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)])
invStyleTrans = transforms.Compose([transforms.Normalize(
    mean=[-0.339/0.157, -0.385/0.164, -0.465/0.159],
    std=[1/0.157, 1/0.164, 1/0.159]
)])

content_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
style_normalize = transforms.Normalize(mean=[0.339, 0.385, 0.465],
                             std=[0.157, 0.164, 0.159])

#invTrans = nn.Identity()
def train_transform(load_size, crop_size):
    transform_list = [
        transforms.Resize(size=(load_size, load_size)),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
def style_transform(load_size, crop_size):
    transform_list = [
        transforms.Resize(size=(load_size, load_size)),
        transforms.RandomCrop(crop_size),
        RandAugment(2, 4,prob=.66),
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
        param.requires_grad = requires_grad

def adjust_learning_rate(optimizer, iteration_count,args, disc=False):
    """Imitating the original implementation"""
    lr = args.disc_lr if disc else args.lr
    lr = lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_lr_adjust(optimizer, iteration_count, warmup_start=1e-7, warmup_iters=1000, max_lr = 1e-3, decay=5e-5):
    """Imitating the original implementation"""
    warmup_step = (max_lr - warmup_start) / warmup_iters
    if iteration_count < warmup_iters:
        lr = warmup_start + (iteration_count * warmup_step)
    else:
        lr = max_lr / (1.0 + decay * (iteration_count - warmup_iters))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--train_model', type=str, default='drafting')
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--load_size', type=int, default=128)
parser.add_argument('--style_load_size', type=int, default=128)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--log_every_', type=int, default=10)

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--disc_lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--warmup_iters', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--load_model', type=str, default='none')

# Revision model options
parser.add_argument('--style_augment', type=int, default=0)
parser.add_argument('--content_augment', type=int, default=0)
parser.add_argument('--revision_depth', type=int, default=1)
parser.add_argument('--disc_depth', type=int, default=5)
parser.add_argument('--disc2_depth', type=int, default=5)
parser.add_argument('--disc_channels', type=int, default=64)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--revision_full_size_depth', type=int, default=1)
parser.add_argument('--content_relt', type=float, default=18.5)
parser.add_argument('--style_remd', type=float, default=22.0)
parser.add_argument('--mdog_weight', type=float, default=.66)
parser.add_argument('--thumbnail_loss', type=float, default=.75)
parser.add_argument('--load_rev', type=int, default=0)
parser.add_argument('--load_disc', type=int, default=0)
parser.add_argument('--load_optimizer', type=int, default=0)
parser.add_argument('--disc_quantization', type=int, default=0)
parser.add_argument('--remd_loss', type=int, default=1)
parser.add_argument('--content_style_loss', type=int, default=1)
parser.add_argument('--identity_loss', type=int, default=0)
parser.add_argument('--mdog_loss', type=int, default=0)
parser.add_argument('--patch_loss', type=float, default=1)
parser.add_argument('--gan_loss', type=float, default=2.5)
parser.add_argument('--gan_loss2', type=float, default=2.5)
parser.add_argument('--momentumnet_beta', type=float, default=.9)
parser.add_argument('--disc_update_steps', type=int, default = 1)
parser.add_argument('--fp16', type=int, default=0)
parser.add_argument('--style_contrastive_loss', type=int, default=0)
parser.add_argument('--content_contrastive_loss', type=int, default=0)
parser.add_argument('--s_d', type=int, default=512)
parser.add_argument('--draft_disc', type=int, default=0)
parser.add_argument('--content_all_layers', type=int, default=0)
parser.add_argument('--split_style', type=int, default=0)

args = parser.parse_args()

if args.fp16 ==1:
    ac_enabled=True

args.disc_disc = args.draft_disc == 1
args.split_style = args.split_style == 1
args.content_all_layers = args.content_all_layers == 1
args.content_style_loss = args.content_style_loss == 1

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
wandb.init(config=vars(args))

def build_enc(vgg):
    enc = net.Encoder(vgg)
    set_requires_grad(enc, False)
    enc.train(False)
    return enc

with autocast(enabled=ac_enabled):
    vgg = vgg.vgg

    vgg.load_state_dict(torch.load(args.vgg), strict=False)
    vgg = nn.Sequential(*list(vgg.children()))

    content_tf = style_transform(args.load_size, args.crop_size) if args.content_augment == 1 else train_transform(args.load_size, args.crop_size)
    style_tf = style_transform(args.style_load_size, args.crop_size) if args.style_augment == 1 else train_transform(args.style_load_size, args.crop_size)

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

batch = args.batch_size if args.style_contrastive_loss==0 else args.batch_size//2
content_iter = iter(data.DataLoader(
    content_dataset, batch_size=batch,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads,pin_memory=True))

style_iter = iter(data.DataLoader(
    style_dataset, batch_size=batch,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads,pin_memory=True))

remd_loss = True if args.remd_loss==1 else 0
mdog_loss = True if args.mdog_loss==1 else 0


def build_rev():
    rev = net.RevisionNet(batch_size=args.batch_size, s_d=args.s_d).to(device)
    #if not state is None:
    #    state = torch.load(state)
    #    rev.load_state_dict(state, strict=False)
    init_weights(rev)
    rev.train()

    return rev

def build_revlap(depth, state):
    rev = RevisorLap(args.batch_size, levels=args.revision_depth).to(device)
    if state is None:
        init_weights(rev)
    else:
        state = torch.load(state)
        rev.load_state_dict(state, strict=False)
    rev.train()
    return rev

def build_disc(disc_state, depth):
    with autocast(enabled=ac_enabled):
        disc = net.Discriminator(depth=depth,num_channels=args.disc_channels).to(device)
        #disc.init_spectral_norm()
        if not disc_state is None:
            try:
                disc.load_state_dict(torch.load(disc_state), strict=False)
            except Exception as e:
                print(e)
                print(disc_state+' not loaded')
        disc.train()
        disc.to(torch.device('cuda'))
    return disc

def drafting_train():
    num_rev = 0

    enc_ = torch.jit.trace(build_enc(vgg), (torch.rand((args.batch_size, 3, 256, 256))), strict=False)
    dec_ = net.ThumbAdaConv(style_contrastive_loss=args.style_contrastive_loss == 1,
                            content_contrastive_loss=args.content_contrastive_loss == 1, batch_size=args.batch_size,
                            s_d=args.s_d).to(device)

    # dec_ = torch.jit.script(net.ThumbAdaConv(batch_size=args.batch_size,s_d=args.s_d).to(device))

    if args.load_model == 'none':
        init_weights(dec_)
    else:
        path = args.load_model.split('/')
        path_tokens = args.load_model.split('_')
        new_path_func = lambda x: '/'.join(path[:-1]) + '/' + x + "_".join(path_tokens[-2:])

        dec_.load_state_dict(torch.load(args.load_model), strict=False)
        if args.load_optimizer == 1:
            try:
                dec_optimizer.load_state_dict(
                    torch.load('/'.join(args.load_model.split('/')[:-1]) + '/dec_optimizer.pth.tar'))
            except:
                print('optimizer not loaded ')
        dec_optimizer.lr = args.lr
    dec_.train()
    enc_.to(device)
    remd_loss = True if args.remd_loss == 1 else False

    for n in tqdm(range(args.max_iter), position=0):
        warmup_lr_adjust(dec_optimizer, n, warmup_start=1e-7, warmup_iters=args.warmup_iters, max_lr=args.lr,
                         decay=args.lr_decay)

        ci = content_normalize(next(content_iter))
        si = style_normalize(next(style_iter))

        if args.style_contrastive_loss == 1:
            ci_ = ci[1:]
            ci_ = torch.cat([ci_, ci[0:1]], 0)
            ci = torch.cat([ci, ci_], 0)
            si = torch.cat([si, si], 0)

        ci = [F.interpolate(ci, size=256, mode='bicubic').to(device)]
        si = [F.interpolate(si, size=256, mode='bicubic').to(device)]
        cF = enc_(ci[0])
        sF = enc_(si[0])

        for param in dec_.parameters():
            param.grad = None

        stylized, style_emb = dec_(cF, sF['r4_1'])
        disc_.eval()

        losses = calc_losses(stylized, ci[0], si[0], cF, enc_, dec_, None, None,
                             calc_identity=args.identity_loss == 1, disc_loss=False,
                             mdog_losses=args.mdog_loss, style_contrastive_loss=args.style_contrastive_loss == 1,
                             content_contrastive_loss=args.content_contrastive_loss == 1,
                             remd_loss=remd_loss, patch_loss=False, patch_stylized=None, top_level_patch=None,
                             sF=sF)
        loss_c, loss_s, content_relt, style_remd, l_identity1, l_identity2, l_identity3, l_identity4, \
        mdog, loss_Gp_GAN, patch_loss, style_contrastive_loss, content_contrastive_loss, pixel_loss = losses

        loss = loss_s * args.style_weight + content_relt * args.content_relt + \
               style_remd * args.style_remd + patch_loss * args.patch_loss + \
               loss_Gp_GAN * args.gan_loss + mdog * args.mdog_weight + l_identity1 * 50 \
               + l_identity2 + l_identity3 * 50 + l_identity4 + \
               style_contrastive_loss * 0.6 + content_contrastive_loss * 0.6 + pixel_loss / args.content_relt

        loss.backward()
        if n > 0:
            dec_optimizer.step()
        if (n + 1) % args.log_every_ == 0:

            loss_dict = {}
            for l, s in zip(
                    [dec_optimizer.param_groups[0]['lr'], loss, loss_c, loss_s, style_remd, content_relt, patch_loss,
                     mdog, style_contrastive_loss, content_contrastive_loss,
                     l_identity1, l_identity2, l_identity3, l_identity4, pixel_loss],
                    ['LR', 'Loss', 'Content Loss', 'Style Loss', 'Style REMD', 'Content RELT',
                     'Patch Loss', 'MXDOG Loss',
                     'Style Contrastive Loss', 'Content Contrastive Loss',
                     "Identity 1 Loss", "Identity 2 Loss", "Identity 3 Loss", "Identity 4 Loss",
                     'Pixel Loss']):
                if type(l) == torch.Tensor:
                    loss_dict[s] = l.item()
                elif type(l) == float or type(l) == int:
                    if l != 0:
                        loss_dict[s] = l
            if (n + 1) % 10 == 0:
                loss_dict['example'] = wandb.Image(stylized[0].transpose(2, 0).transpose(1, 0).detach().cpu().numpy())
            print(str(n) + '/' + str(args.max_iter) + ': ' + '\t'.join(
                [str(k) + ': ' + str(v) for k, v in loss_dict.items()]))

            wandb.log(loss_dict, step=n)

        with torch.no_grad():
            if (n + 1) % 50 == 0:
                draft_img_grid = make_grid(invStyleTrans(stylized), nrow=4, scale_each=True)
                style_source_grid = make_grid(si[0], nrow=4, scale_each=True)
                content_img_grid = make_grid(ci[0], nrow=4, scale_each=True)
                save_image(draft_img_grid,
                           args.save_dir + '/drafting_draft_iter' + str(n + 1) + '.jpg')
                save_image(invTrans(content_img_grid),
                           args.save_dir + '/drafting_training_iter_ci' + str(
                               n + 1) + '.jpg')
                save_image(invStyleTrans(style_source_grid),
                           args.save_dir + '/drafting_training_iter_si' + str(
                               n + 1) + '.jpg')
                del (draft_img_grid, style_source_grid, content_img_grid)

            if (n + 1) % args.save_model_interval == 0 or (n + 1) == args.max_iter:
                state_dict = dec_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'decoder_iter_{:d}.pth.tar'.format(n + 1))
                state_dict = dec_optimizer.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'dec_optimizer.pth.tar')

def revision_train():
    num_rev = {256 * 2 ** i: i for i in range(4)}[args.crop_size]

    enc_ = torch.jit.trace(build_enc(vgg), (torch.rand((args.batch_size, 3, 256, 256))),
                           strict=False)
    dec_ = net.ThumbAdaConv(style_contrastive_loss=args.style_contrastive_loss == 1,
                            content_contrastive_loss=args.content_contrastive_loss == 1,
                            batch_size=args.batch_size, s_d=args.s_d).to(device)

    rev_ = []
    disc2_ = []
    opt_D2 = []
    rev_optimizer = []
    if args.load_disc == 1:
        path = args.load_model.split('/')
        path_tokens = args.load_model.split('_')
        new_path_func = lambda x: '/'.join(path[:-1]) + '/' + x + "_".join(path_tokens[-2:])
        disc_state = new_path_func('discriminator_')
    else:
        disc_state = None
        init_weights(dec_)
    disc_ = torch.jit.trace(build_disc(
        disc_state, args.disc_depth), torch.rand(args.batch_size, 3, 256, 256, device='cuda'),
        check_trace=False)
    dec_optimizer = torch.optim.AdamW(dec_.parameters(recurse=True), lr=args.lr)
    opt_D = torch.optim.AdamW(disc_.parameters(recurse=True), lr=args.disc_lr)
    if args.load_disc == 1 and args.load_model != 'none':
        try:
            opt_D.load_state_dict(
                torch.load('/'.join(args.load_model.split('/')[:-1]) + '/disc_optimizer.pth.tar'))
        except:
            print('discriminator optimizer not loaded')
    if args.load_model == 'none':
        init_weights(dec_)
    else:
        path = args.load_model.split('/')
        path_tokens = args.load_model.split('_')
        new_path_func = lambda x: '/'.join(path[:-1]) + '/' + x + "_".join(path_tokens[-2:])

        dec_.load_state_dict(torch.load(args.load_model), strict=False)
        if args.load_optimizer == 1:
            try:
                dec_optimizer.load_state_dict(torch.load(
                    '/'.join(args.load_model.split('/')[:-1]) + '/dec_optimizer.pth.tar'))
            except:
                print('optimizer not loaded ')
        dec_optimizer.lr = args.lr
    dec_.train()
    enc_.to(device)
    remd_loss = True if args.remd_loss == 1 else False
    random_crop = [nn.Identity()] + [transforms.RandomCrop(args.crop_size / 2 ** e) for e in
                                     range(1, num_rev + 1)]
    steps_per_revision = 20000
    current_revision = 0
    n=0
    while current_revision <= num_rev-1:
        rev_.append(
            torch.jit.trace(build_rev(),
                            (torch.rand(args.batch_size, 3, 256, 256, device='cuda'),) * 2))
        disc2_.append(
            torch.jit.trace(build_disc(None, args.disc2_depth),
                            torch.rand(args.batch_size, 3, 256, 256, device='cuda'),
                            check_trace=False)
        )
        opt_D2.append(
            torch.optim.AdamW(disc2_[-1].parameters(recurse=True), lr=args.disc_lr)
        )
        rev_optimizer.append(
            torch.optim.AdamW(rev_[-1].parameters(recurse=True), lr=args.lr)
        )
        for n in tqdm(range(steps_per_revision), position=0):
            warmup_lr_adjust(dec_optimizer, n, warmup_start=1e-7, warmup_iters=args.warmup_iters,
                             max_lr=args.lr, decay=args.lr_decay)
            for rev_opt in rev_optimizer:
                warmup_lr_adjust(rev_opt, n, warmup_start=1e-7, warmup_iters=args.warmup_iters,
                                 max_lr=args.lr,
                                 decay=args.lr_decay)
            warmup_lr_adjust(opt_D, n, warmup_start=1e-7, warmup_iters=args.warmup_iters,
                             max_lr=args.lr,
                             decay=args.disc_lr)
            for d_opt in opt_D2:
                warmup_lr_adjust(d_opt, n, warmup_start=1e-7, warmup_iters=args.warmup_iters,
                                 max_lr=args.disc_lr,
                                 decay=args.disc_lr)

            ci = content_normalize(next(content_iter))
            si = style_normalize(next(style_iter))

            if args.style_contrastive_loss == 1:
                ci_ = ci[1:]
                ci_ = torch.cat([ci_, ci[0:1]], 0)
                ci = torch.cat([ci, ci_], 0)
                si = torch.cat([si, si], 0)

            crop_marks = torch.randint(0, 127, (num_rev, 2))
            ci = [F.interpolate(ci, size=256, mode='bicubic').to(device)] + [
                F.interpolate(crop_mark_extract(num_rev, crop_marks, ci, e), size=256,
                              mode='bicubic').to(device) for e in range(num_rev)]
            si = [F.interpolate(e(si), size=256, mode='bicubic').to(device) for e in random_crop]
            cF = enc_(ci[0])
            sF = enc_(si[0])
            if n > 2 and n % args.disc_update_steps == 0:
                dec_.eval()
                for rev in rev_: rev.eval()
                stylized, style_emb = dec_(cF, sF['r4_1'])
                stylized_patches = []
                for i in range(current_revision+1):
                    orig = stylized if i == 0 else patch_stylized
                    res_in = F.interpolate(orig[:, :, crop_marks[i][0]:crop_marks[i][0] + 128,
                                           crop_marks[i][1]:crop_marks[i][1] + 128], 256,
                                           mode='nearest')
                    patch_stylized, etf = rev_[i](res_in.clone().detach().requires_grad_(True),
                                                  ci[1 + i])
                    patch_stylized = patch_stylized + res_in
                    stylized_patches.append(patch_stylized)

                for param in disc_.parameters():
                    param.grad = None
                for disc in disc2_:
                    for param in disc.parameters():
                        param.grad = None

                set_requires_grad(disc_, True)
                for disc in disc2_: set_requires_grad(disc, True)
                set_requires_grad(dec_, False)
                for rev in rev_: set_requires_grad(rev, False)
                loss_D = calc_GAN_loss(si[0], stylized.clone().detach().requires_grad_(True), disc_)
                loss_D2 = 0
                for i, patch_stylized in enumerate(stylized_patches):
                    loss_D2 += calc_GAN_loss(si[1 + i],
                                             patch_stylized.clone().detach().requires_grad_(True),
                                             disc2_[i])

                loss_D.backward()
                loss_D2.backward()

                if n > 0:
                    # _clip_gradient(disc2_)
                    # _clip_gradient(disc_)
                    for d_opt in opt_D2: d_opt.step()
                    opt_D.step()

                set_requires_grad(disc_, False)
                for disc in disc2_: set_requires_grad(disc, False)
                set_requires_grad(dec_, True)
                for rev in rev_: set_requires_grad(rev, True)

            dec_.train()
            for rev in rev_: rev.train()

            for param in dec_.parameters():
                param.grad = None
            for rev in rev_:
                for param in rev.parameters():
                    param.grad = None

            stylized, style_emb = dec_(cF, sF['r4_1'])
            thumbs = []
            stylized_patches = []
            for i in range(current_revision+1):
                orig = stylized if i == 0 else patch_stylized
                res_in = F.interpolate(
                    orig[:, :, crop_marks[i][0]:crop_marks[i][0] + 128,
                    crop_marks[i][1]:crop_marks[i][1] + 128], 256, mode='nearest')
                thumbs.append(res_in)
                patch_stylized, etf = rev_[i](res_in.clone().detach().requires_grad_(True), ci[1 + i])
                patch_stylized = patch_stylized + res_in
                stylized_patches.append(patch_stylized)

            disc_.eval()
            for disc in disc2_: disc.eval()

            losses = calc_losses(stylized, ci[0], si[0], cF, enc_, dec_, None, disc_,
                                 calc_identity=args.identity_loss == 1, disc_loss=True,
                                 mdog_losses=args.mdog_loss,
                                 style_contrastive_loss=args.style_contrastive_loss == 1,
                                 content_contrastive_loss=args.content_contrastive_loss == 1,
                                 remd_loss=remd_loss, patch_loss=False, patch_stylized=None,
                                 top_level_patch=None,
                                 sF=sF)
            loss_c, loss_s, content_relt, style_remd, l_identity1, l_identity2, l_identity3, l_identity4, \
            mdog, loss_Gp_GAN, patch_loss, style_contrastive_loss, content_contrastive_loss, pixel_loss = losses

            loss = loss_s * args.style_weight + content_relt * args.content_relt + \
                   style_remd * args.style_remd + patch_loss * args.patch_loss + \
                   loss_Gp_GAN * args.gan_loss + mdog * args.mdog_weight + l_identity1 * 50 \
                   + l_identity2 + l_identity3 * 50 + l_identity4 + \
                   style_contrastive_loss * 0.6 + content_contrastive_loss * 0.6 + pixel_loss / args.content_relt

            for idx in range(current_revision+1):
                patch_cF = enc_(ci[idx + 1])
                patch_sF = enc_(si[idx + 1])
                patch_losses = calc_losses(stylized_patches[idx], ci[idx + 1], si[idx + 1], patch_cF,
                                           enc_, dec_, None, disc2_[idx],
                                           calc_identity=False, disc_loss=True,
                                           mdog_losses=False, style_contrastive_loss=False,
                                           content_contrastive_loss=False,
                                           remd_loss=False, patch_loss=True,
                                           patch_stylized=stylized_patches[idx],
                                           top_level_patch=thumbs[idx],
                                           sF=patch_sF)
                loss_c, loss_s, content_reltp, style_remdp, l_identity1, l_identity2, l_identity3, l_identity4, \
                mdog, loss_Gp_GAN, patch_loss, style_contrastive_loss, content_contrastive_loss, pixel_loss = patch_losses

                loss = loss + loss_s * args.style_weight + content_reltp * args.content_relt + \
                       style_remdp * args.style_remd + patch_loss * args.patch_loss + \
                       loss_Gp_GAN * args.gan_loss + mdog * args.mdog_weight + l_identity1 * 50 \
                       + l_identity2 + l_identity3 * 50 + l_identity4 + \
                       style_contrastive_loss * 0.6 + content_contrastive_loss * 0.6 + pixel_loss / args.content_relt

            loss.backward()
            if n > 0:
                # _clip_gradient(rev_)
                # _clip_gradient(dec_)
                for rev_opt in rev_optimizer: rev_opt.step()
                dec_optimizer.step()
            for disc in disc2_: disc.train()
            disc_.train()
            if (n + 1) % args.log_every_ == 0:

                loss_dict = {}
                for l, s in zip(
                        [dec_optimizer.param_groups[0]['lr'], loss, loss_c, loss_s, style_remd,
                         content_relt, patch_loss,
                         mdog, loss_Gp_GAN, loss_D, style_contrastive_loss, content_contrastive_loss,
                         l_identity1, l_identity2, l_identity3, l_identity4, loss_D2, pixel_loss],
                        ['LR', 'Loss', 'Content Loss', 'Style Loss', 'Style REMD', 'Content RELT',
                         'Patch Loss', 'MXDOG Loss', 'Decoder Disc. Loss', 'Discriminator Loss',
                         'Style Contrastive Loss', 'Content Contrastive Loss',
                         "Identity 1 Loss", "Identity 2 Loss", "Identity 3 Loss", "Identity 4 Loss",
                         'Discriminator Loss (detail', 'Pixel Loss']):
                    if type(l) == torch.Tensor:
                        loss_dict[s] = l.item()
                    elif type(l) == float or type(l) == int:
                        if l != 0:
                            loss_dict[s] = l
                if (n + 1) % 10 == 0:
                    loss_dict['example'] = wandb.Image(
                        stylized[0].transpose(2, 0).transpose(1, 0).detach().cpu().numpy())
                print(str(n) + '/' + str(args.max_iter) + ': ' + '\t'.join(
                    [str(k) + ': ' + str(v) for k, v in loss_dict.items()]))

                wandb.log(loss_dict, step=n)

            with torch.no_grad():
                if (n + 1) % 50 == 0:
                    draft_img_grid = make_grid(invStyleTrans(stylized), nrow=4, scale_each=True)
                    style_source_grid = make_grid(si[0], nrow=4, scale_each=True)
                    content_img_grid = make_grid(ci[0], nrow=4, scale_each=True)
                    etf_grid = make_grid(etf,nrow=4, scale_each=True)
                    for idx, patch_stylized in enumerate(stylized_patches):
                        styled_img_grid = make_grid(invStyleTrans(patch_stylized), nrow=4,
                                                    scale_each=True)
                        version = '' if idx == 0 else str(idx) + '_'
                        save_image(styled_img_grid,
                                   args.save_dir + '/drafting_revision_' + version + 'iter' + str(
                                       n + 1) + '.jpg')
                    save_image(etf_grid,
                               args.save_dir + '/etf_iter' + str(n + 1) + '.jpg')
                    save_image(draft_img_grid,
                               args.save_dir + '/drafting_draft_iter' + str(n + 1) + '.jpg')
                    save_image(invTrans(content_img_grid),
                               args.save_dir + '/drafting_training_iter_ci' + str(
                                   n + 1) + '.jpg')
                    save_image(invStyleTrans(style_source_grid),
                               args.save_dir + '/drafting_training_iter_si' + str(
                                   n + 1) + '.jpg')
                    del (draft_img_grid, styled_img_grid, style_source_grid, content_img_grid)

                if (n + 1) % args.save_model_interval == 0 or (n + 1) == args.max_iter:
                    state_dict = dec_.state_dict()
                    torch.save(copy.deepcopy(state_dict), save_dir /
                               'decoder_iter_{:d}.pth.tar'.format(n + 1))
                    state_dict = dec_optimizer.state_dict()
                    torch.save(copy.deepcopy(state_dict), save_dir /
                               'dec_optimizer.pth.tar')
                    for idx in range(current_revision+1):
                        num = '' if idx == 0 else '_' + str(idx + 1)
                        state_dict = rev_[idx].state_dict()
                        torch.save(copy.deepcopy(state_dict), save_dir /
                                   'revisor{:s}_iter_{:d}.pth.tar'.format(num, n + 1))
                        state_dict = rev_optimizer[idx].state_dict()
                        torch.save(copy.deepcopy(state_dict), save_dir /
                                   'rev_optimizer{:s}.pth.tar'.format(num))
                        state_dict = disc2_[idx].state_dict()
                        torch.save(copy.deepcopy(state_dict), save_dir /
                                   'discriminator_{:d}_iter_{:d}.pth.tar'.format(idx + 2, n + 1))
                        state_dict = opt_D2[idx].state_dict()
                        torch.save(copy.deepcopy(state_dict), save_dir /
                                   'disc{:d}_optimizer.pth.tar'.format(idx + 2))
                    state_dict = disc_.state_dict()
                    torch.save(copy.deepcopy(state_dict), save_dir /
                               'discriminator_iter_{:d}.pth.tar'.format(n + 1))
                    state_dict = opt_D.state_dict()
                    torch.save(copy.deepcopy(state_dict), save_dir /
                               'disc_optimizer.pth.tar')
        current_revision += 1
        if current_revision < num_rev-1:
            steps_per_revision= int(steps_per_revision)/1.5



def revlap_train():
    setup_torch(0)
    rev_start = True
    random_crop = transforms.RandomCrop(256)
    if args.split_style:
        random_crop2 = transforms.RandomCrop(512)
    with autocast(enabled=ac_enabled):
        enc_ = torch.jit.trace(build_enc(vgg), (torch.rand((args.batch_size, 3, 256, 256))), strict=False)
    dec_ = torch.jit.trace(net.DecoderAdaConv(batch_size=args.batch_size).to(device),
        ({k:v for k,v in zip(['r1_1','r2_1','r3_1','r4_1'],
                            [torch.rand(args.batch_size, 64, 256, 256).to(torch.device('cuda')),
                             torch.rand(args.batch_size, 128, 128, 128).to(torch.device('cuda')),
                             torch.rand(args.batch_size, 256, 64, 64).to(torch.device('cuda')),
                             torch.rand(args.batch_size, 512, 32, 32).to(torch.device('cuda'))])},
        {k: v for k, v in zip(['r1_1', 'r2_1', 'r3_1', 'r4_1'],
                              [torch.rand(args.batch_size, 64, 256, 256).to(torch.device('cuda')),
                               torch.rand(args.batch_size, 128, 128, 128).to(torch.device('cuda')),
                               torch.rand(args.batch_size, 256, 64, 64).to(torch.device('cuda')),
                               torch.rand(args.batch_size, 512, 32, 32).to(torch.device('cuda'))])}),
                         strict=False,check_trace=False)
    disc_state = None
    rev_ = LapRev(512, 512, args.batch_size, 512, args.momentumnet_beta).to(device)
    # (torch.rand(args.batch_size, 3, 256, 256).to(torch.device('cuda')),
    # torch.rand(args.batch_size, 3, 512, 512).to(torch.device('cuda')),
    # torch.rand(args.batch_size, 3, 512, 512).to(torch.device('cuda'))),check_trace=False, strict=False)
    if args.load_rev == 1 or args.load_disc == 1:
        path = args.load_model.split('/')
        path_tokens = args.load_model.split('_')
        new_path_func = lambda x: '/'.join(path[:-1]) + '/' + x + "_".join(path_tokens[-2:])
        if args.load_disc == 1:
            disc_state = new_path_func('discriminator_')
        if args.load_rev == 1:
            rev_state = new_path_func('revisor_')
        dec_state = new_path_func('revisor_')
        dec_.load_state_dict(torch.load(dec_state), strict=False)
        rev_.load_state_dict(torch.load(rev_state), strict=False)
    else:
        rev_state = None
        init_weights(dec_)

    disc_ = build_disc(disc_state)#, torch.rand(args.batch_size, 3, 256, 256).to(torch.device('cuda')), check_trace=False, strict=False)

    dec_.train()
    enc_.to(device)
    remd_loss = True if args.remd_loss == 1 else False

    dec_optimizer = torch.optim.AdamW(dec_.parameters(recurse=True),lr=args.lr)

    optimizer = torch.optim.AdamW(rev_.parameters(recurse=True), lr=args.lr)
    opt_D = torch.optim.AdamW(disc_.parameters(recurse=True),lr=args.disc_lr)
    if args.load_rev == 1:
        disc_.load_state_dict(torch.load(new_path_func('revisor_')), strict=False)
        dec_.load_state_dict(torch.load(args.load_model), strict=False)
    if args.load_optimizer ==1:
        optimizer.load_state_dict(torch.load('/'.join(path[:-1])+'/optimizer.pth.tar'))
        opt_D.load_state_dict(torch.load('/'.join(path[:-1]) + '/disc_optimizer.pth.tar'))
    for i in range(args.max_iter):
        adjust_learning_rate(optimizer, i//args.accumulation_steps, args)
        adjust_learning_rate(dec_optimizer, i // args.accumulation_steps, args)
        adjust_learning_rate(opt_D, i//args.accumulation_steps, args, disc=True)
        with autocast(enabled=ac_enabled):
            ci = next(content_iter).to(device)
            si = next(style_iter).to(device)
            ci = [F.interpolate(ci, size=256, mode='bicubic', align_corners=True), ci]
            si = [F.interpolate(si, size=256, mode='bicubic', align_corners=True), si]
        with autocast(enabled=ac_enabled):
            cF = enc_(ci[0])
            sF = enc_(si[0])

            stylized, style, patch_stats = dec_(sF, cF)

            rev_stylized = rev_(stylized, ci[-1])
            si_cropped = random_crop(si[-1])
            stylized_crop = random_crop(rev_stylized)

        with autocast(enabled=ac_enabled):

            loss_D = calc_GAN_loss(si_cropped.detach(), stylized_crop.clone().detach(), None,disc_)

        with autocast(enabled=ac_enabled):

            losses_small = calc_losses(stylized, ci[0], si[0], cF, enc_, dec_, None, disc_,
                                        calc_identity=False, disc_loss=False,
                                        mdog_losses=args.mdog_loss, content_all_layers=False,
                                        remd_loss=remd_loss,
                                        patch_loss=False, sF=sF, split_style=False)
            loss_c, loss_s, content_relt, style_remd, l_identity1, l_identity2, l_identity3, l_identity4, mdog, loss_Gp_GAN, patch_loss = losses_small
            loss = (loss_c * args.content_weight + args.style_weight * loss_s + content_relt * args.content_relt + style_remd * args.style_remd + patch_loss * args.patch_loss + mdog)*args.thumbnail_loss

            cF2 = enc_(ci[-1])
            patch_feats = enc_(F.interpolate(stylized,size=512,mode='nearest'))
            sF2 = enc_(si[-1])
            losses = calc_losses(rev_stylized, ci[-1], si[-1], cF2, enc_, dec_, patch_feats, disc_,
                                 calc_identity=False, disc_loss=True,
                                 mdog_losses=args.mdog_loss, content_all_layers=False, remd_loss=remd_loss,
                                 patch_loss=True, sF=sF2, split_style=args.split_style)
            loss_c2, loss_s2, content_relt2, style_remd2, l_identity1, l_identity2, l_identity3, l_identity4, mdog, loss_Gp_GAN, patch_loss = losses
            loss = loss + loss_c2 * args.content_weight + args.style_weight * loss_s2 + content_relt2 * args.content_relt + style_remd2 * args.style_remd + loss_Gp_GAN * args.gan_loss + patch_loss * args.patch_loss + mdog

        loss.backward()
        loss_D.backward()
        for mod in [dec_,disc_,rev_]:
            _clip_gradient(mod)

        opt_D.step()
        dec_optimizer.step()
        optimizer.step()
        opt_D.zero_grad()
        dec_optimizer.zero_grad()
        optimizer.zero_grad
        if (i + 1) % 1 == 0:

            loss_dict = {}
            for l, s in zip(
                    [loss, loss_c, loss_s, style_remd, content_relt, loss_Gp_GAN, loss_D, patch_loss,
                     mdog],
                    ['Loss', 'Content Loss', 'Style Loss', 'Style REMD', 'Content RELT',
                     'Revision Disc. Loss', 'Discriminator Loss', 'Patch Loss', 'MXDOG Loss']):
                if type(l) == torch.Tensor:
                    loss_dict[s] = l.item()
            if(i +1) % 10 ==0:
                loss_dict['example'] = wandb.Image(rev_stylized[0].transpose(2, 0).transpose(1, 0).detach().cpu().numpy())
            print('\n')
            print(str(i)+'/'+str(args.max_iter)+': '+'\t'.join([str(k) + ': ' + str(v) for k, v in loss_dict.items()]))

            wandb.log(loss_dict, step=i)

        with torch.no_grad():
            if ((i + 1) % 50 == 0 and rev_start) or ((i+1)%250==0):

                stylized = stylized.float().to('cpu')
                rev_stylized = rev_stylized.float().to('cpu')
                draft_img_grid = make_grid(stylized, nrow=4, scale_each=True)
                if rev_start:
                    styled_img_grid = make_grid(rev_stylized, nrow=4, scale_each=True)
                si[-1] = F.interpolate(si[-1], size=256, mode='bicubic')
                ci[-1] = F.interpolate(ci[-1], size=256, mode='bicubic')
                style_source_grid = make_grid(si[-1], nrow=4, scale_each=True)
                content_img_grid = make_grid(ci[-1], nrow=4, scale_each=True)
                if rev_start:
                    save_image(styled_img_grid.detach(), args.save_dir + '/drafting_revision_iter' + str(i + 1) + '.jpg')
                save_image(draft_img_grid.detach(),
                           args.save_dir + '/drafting_draft_iter' + str(i + 1) + '.jpg')
                save_image(content_img_grid.detach(),
                           args.save_dir + '/drafting_training_iter_ci' + str(
                               i + 1) + '.jpg')
                save_image(style_source_grid.detach(),
                           args.save_dir + '/drafting_training_iter_si' + str(
                               i + 1) + '.jpg')

            if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
                state_dict = rev_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'revisor_iter_{:d}.pth.tar'.format(i + 1))
                state_dict = dec_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'decoder_iter_{:d}.pth.tar'.format(i + 1))
                state_dict = disc_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'discriminator_iter_{:d}.pth.tar'.format(i + 1))
                state_dict = optimizer.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'optimizer.pth.tar')
                state_dict = opt_D.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'disc_optimizer.pth.tar')
                state_dict = dec_optimizer.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'dec_optimizer.pth.tar')

def adaconv_thumb_train():
    num_rev = {256 * 2 ** i: i for i in range(4)}[args.crop_size]

    enc_ = torch.jit.trace(build_enc(vgg), (torch.rand((args.batch_size, 3, 256, 256))), strict=False)
    dec_ = net.ThumbAdaConv(style_contrastive_loss=args.style_contrastive_loss==1,content_contrastive_loss=args.content_contrastive_loss==1,batch_size=args.batch_size,s_d=args.s_d).to(device)

    #dec_ = torch.jit.script(net.ThumbAdaConv(batch_size=args.batch_size,s_d=args.s_d).to(device))

    rev_ = [torch.jit.trace(build_rev(),(torch.rand(args.batch_size,3,256,256,device='cuda'),)*2) for i in range(num_rev)]
    if args.load_disc == 1:
        path = args.load_model.split('/')
        path_tokens = args.load_model.split('_')
        new_path_func = lambda x: '/'.join(path[:-1]) + '/' + x + "_".join(path_tokens[-2:])
        disc_state = new_path_func('discriminator_')
        disc2_state = [new_path_func('discriminator_'+str(i+2)+'_') for i in range(num_rev)]
    else:
        disc_state = None
        disc2_state = [None for i in range(num_rev)]
        init_weights(dec_)
    disc_ = torch.jit.trace(build_disc(
        disc_state, args.disc_depth), torch.rand(args.batch_size, 3, 256, 256, device='cuda'), check_trace=False)
    disc2_ = [
        torch.jit.trace(build_disc(state, args.disc2_depth), torch.rand(args.batch_size, 3, 256, 256, device='cuda'),
                        check_trace=False) for state in disc2_state]
    dec_optimizer = torch.optim.AdamW(dec_.parameters(recurse=True), lr=args.lr)
    rev_optimizer = [torch.optim.AdamW(rev.parameters(recurse=True), lr=args.lr) for rev in rev_]
    opt_D = torch.optim.AdamW(disc_.parameters(recurse=True), lr=args.disc_lr)
    opt_D2 = [torch.optim.AdamW(disc.parameters(recurse=True), lr=args.disc_lr) for disc in disc2_]
    if args.load_disc ==1 and args.load_model != 'none':
        try:
            opt_D.load_state_dict(torch.load('/'.join(args.load_model.split('/')[:-1])+'/disc_optimizer.pth.tar'))
        except:
            print('discriminator optimizer not loaded')
        for i in range(num_rev):
            try:
                [opt_D2[i].load_state_dict(torch.load('/'.join(args.load_model.split('/')[:-1])+'/disc'+str(i+2)+'_optimizer.pth.tar')) for i in range(num_rev)]
            except:
                print(f'discriminator optimizer{i+2} not loaded')
    if args.load_model == 'none':
        init_weights(dec_)
    else:
        path = args.load_model.split('/')
        path_tokens = args.load_model.split('_')
        new_path_func = lambda x: '/'.join(path[:-1]) + '/' + x + "_".join(path_tokens[-2:])

        dec_.load_state_dict(torch.load(args.load_model), strict=False)
        if args.load_rev==1:
            for idx, rev in enumerate(rev_):
                num = '' if idx==0 else str(idx+1)+'_'
                try:
                    rev.load_state_dict(torch.load(new_path_func('revisor_'+num)),strict=False)
                except:
                    print(f'revision {num} not loaded')
        if args.load_optimizer==1:
            try:
                dec_optimizer.load_state_dict(torch.load('/'.join(args.load_model.split('/')[:-1])+'/dec_optimizer.pth.tar'))
            except:
                print('optimizer not loaded ')
            for idx, rev_opt in enumerate(rev_optimizer):
                num = '' if idx==0 else '_'+str(idx+1)
                try:
                    rev_optimizer[idx].load_state_dict(torch.load('/'.join(args.load_model.split('/')[:-1])+'/rev_optimizer'+num+'.pth.tar'))
                except:
                    print(f'rev_optimizer{num} not loaded')
        dec_optimizer.lr = args.lr
    dec_.train()
    enc_.to(device)
    remd_loss = True if args.remd_loss == 1 else False
    random_crop = [nn.Identity()] + [transforms.RandomCrop(args.crop_size/2**e) for e in range(1,num_rev+1)]
    #wandb.watch((dec_,disc2_), log_freq=args.log_every_)

    for n in tqdm(range(args.max_iter), position=0):
        warmup_lr_adjust(dec_optimizer, n, warmup_start=1e-7, warmup_iters=args.warmup_iters, max_lr=args.lr, decay=args.lr_decay)
        for rev_opt in rev_optimizer:
            warmup_lr_adjust(rev_opt, n, warmup_start=1e-7, warmup_iters=args.warmup_iters, max_lr=args.lr,
                             decay=args.lr_decay)
        warmup_lr_adjust(opt_D, n, warmup_start=1e-7, warmup_iters=args.warmup_iters, max_lr=args.lr,
                         decay=args.disc_lr)
        for d_opt in opt_D2:
            warmup_lr_adjust(d_opt, n, warmup_start=1e-7, warmup_iters=args.warmup_iters, max_lr=args.disc_lr,
                             decay=args.disc_lr)

        ci = content_normalize(next(content_iter))
        si = style_normalize(next(style_iter))

        if args.style_contrastive_loss == 1:
            ci_ = ci[1:]
            ci_ = torch.cat([ci_, ci[0:1]], 0)
            ci = torch.cat([ci, ci_], 0)
            si = torch.cat([si, si], 0)

        crop_marks = torch.randint(0, 127, (num_rev, 2))
        ci = [F.interpolate(ci, size=256, mode='bicubic').to(device)] + [F.interpolate(crop_mark_extract(num_rev,crop_marks,ci,e), size=256, mode='bicubic').to(device) for e in range(num_rev)]
        si = [F.interpolate(e(si), size=256, mode='bicubic').to(device) for e in random_crop]
        cF = enc_(ci[0])
        sF = enc_(si[0])
        if n>2 and n % args.disc_update_steps == 0:
            dec_.eval()
            for rev in rev_: rev.eval()
            stylized, style_emb = dec_(cF, sF['r4_1'])
            stylized_patches = []
            for i in range(num_rev):
                orig = stylized if i==0 else patch_stylized
                res_in = F.interpolate(orig[:, :, crop_marks[i][0]:crop_marks[i][0]+128, crop_marks[i][1]:crop_marks[i][1]+128], 256, mode='nearest')
                patch_stylized, etf = rev_[i](res_in.clone().detach().requires_grad_(True), ci[1+i])
                patch_stylized = patch_stylized + res_in
                stylized_patches.append(patch_stylized)

            for param in disc_.parameters():
                param.grad = None
            for disc in disc2_:
                for param in disc.parameters():
                    param.grad = None

            set_requires_grad(disc_, True)
            for disc in disc2_: set_requires_grad(disc, True)
            set_requires_grad(dec_, False)
            for rev in rev_: set_requires_grad(rev, False)
            loss_D = calc_GAN_loss(si[0], stylized.clone().detach().requires_grad_(True), disc_)
            loss_D2 = 0
            for i, patch_stylized in enumerate(stylized_patches):
                loss_D2 += calc_GAN_loss(si[1+i], patch_stylized.clone().detach().requires_grad_(True), disc2_[i])

            loss_D.backward()
            loss_D2.backward()

            if n>0:
                #_clip_gradient(disc2_)
                #_clip_gradient(disc_)
                for d_opt in opt_D2: d_opt.step()
                opt_D.step()

            set_requires_grad(disc_, False)
            for disc in disc2_: set_requires_grad(disc, False)
            set_requires_grad(dec_, True)
            for rev in rev_: set_requires_grad(rev, True)

        dec_.train()
        for rev in rev_: rev.train()

        for param in dec_.parameters():
            param.grad = None
        for rev in rev_:
            for param in rev.parameters():
                param.grad = None

        stylized, style_emb = dec_(cF,sF['r4_1'])
        thumbs = []
        stylized_patches = []
        for i in range(num_rev):
            orig = stylized if i == 0 else patch_stylized
            res_in = F.interpolate(
                orig[:, :, crop_marks[i][0]:crop_marks[i][0] + 128, crop_marks[i][1]:crop_marks[i][1] + 128], 256, mode='nearest')
            thumbs.append(res_in)
            patch_stylized, etf = rev_[i](res_in.clone().detach().requires_grad_(True), ci[1 + i])
            patch_stylized = patch_stylized + res_in
            stylized_patches.append(patch_stylized)

        disc_.eval()
        for disc in disc2_: disc.eval()

        losses = calc_losses(stylized, ci[0], si[0], cF, enc_, dec_, None, disc_,
                             calc_identity=args.identity_loss == 1, disc_loss=True,
                             mdog_losses=args.mdog_loss, style_contrastive_loss=args.style_contrastive_loss == 1,
                             content_contrastive_loss=args.content_contrastive_loss == 1,
                             remd_loss=remd_loss, patch_loss=False, patch_stylized=None, top_level_patch=None,
                             sF=sF)
        loss_c, loss_s, content_relt, style_remd, l_identity1, l_identity2, l_identity3, l_identity4, \
        mdog, loss_Gp_GAN, patch_loss, style_contrastive_loss, content_contrastive_loss, pixel_loss = losses

        loss = loss_s* args.style_weight + content_relt * args.content_relt + \
               style_remd * args.style_remd + patch_loss * args.patch_loss + \
               loss_Gp_GAN * args.gan_loss + mdog * args.mdog_weight + l_identity1 * 50 \
               + l_identity2 + l_identity3 * 50 + l_identity4 + \
               style_contrastive_loss * 0.6 + content_contrastive_loss * 0.6 + pixel_loss/args.content_relt

        for idx in range(num_rev):
            patch_cF = enc_(ci[idx+1])
            patch_sF = enc_(si[idx+1])
            patch_losses = calc_losses(stylized_patches[idx], ci[idx+1], si[idx+1], patch_cF, enc_, dec_, None, disc2_[idx],
                                 calc_identity=False, disc_loss=True,
                                 mdog_losses=False, style_contrastive_loss=False,
                                 content_contrastive_loss=False,
                                 remd_loss=False, patch_loss=True, patch_stylized=stylized_patches[idx], top_level_patch=thumbs[idx],
                                 sF=patch_sF)
            loss_c, loss_s, content_reltp, style_remdp, l_identity1, l_identity2, l_identity3, l_identity4, \
            mdog, loss_Gp_GAN, patch_loss, style_contrastive_loss, content_contrastive_loss, pixel_loss = patch_losses

            loss = loss + loss_s * args.style_weight + content_reltp * args.content_relt + \
                   style_remdp * args.style_remd + patch_loss * args.patch_loss + \
                   loss_Gp_GAN * args.gan_loss + mdog * args.mdog_weight + l_identity1 * 50 \
                   + l_identity2 + l_identity3 * 50 + l_identity4 + \
                   style_contrastive_loss * 0.6 + content_contrastive_loss * 0.6 + pixel_loss / args.content_relt

        loss.backward()
        if n > 0:
            #_clip_gradient(rev_)
            #_clip_gradient(dec_)
            for rev_opt in rev_optimizer: rev_opt.step()
            dec_optimizer.step()
        for disc in disc2_: disc.train()
        disc_.train()
        if (n + 1) % args.log_every_ == 0:

            loss_dict = {}
            for l, s in zip(
                    [dec_optimizer.param_groups[0]['lr'], loss, loss_c, loss_s, style_remd, content_relt, patch_loss,
                     mdog, loss_Gp_GAN, loss_D,style_contrastive_loss, content_contrastive_loss,
                     l_identity1,l_identity2,l_identity3,l_identity4, loss_D2, pixel_loss],
                    ['LR','Loss', 'Content Loss', 'Style Loss', 'Style REMD', 'Content RELT',
                     'Patch Loss', 'MXDOG Loss', 'Decoder Disc. Loss','Discriminator Loss',
                     'Style Contrastive Loss','Content Contrastive Loss',
                     "Identity 1 Loss","Identity 2 Loss","Identity 3 Loss","Identity 4 Loss",
                     'Discriminator Loss (detail','Pixel Loss']):
                if type(l) == torch.Tensor:
                    loss_dict[s] = l.item()
                elif type(l) == float or type(l)==int:
                    if l != 0:
                        loss_dict[s] = l
            if(n +1) % 10 ==0:
                loss_dict['example'] = wandb.Image(stylized[0].transpose(2, 0).transpose(1, 0).detach().cpu().numpy())
            print(str(n)+'/'+str(args.max_iter)+': '+'\t'.join([str(k) + ': ' + str(v) for k, v in loss_dict.items()]))

            wandb.log(loss_dict, step=n)

        with torch.no_grad():
            if (n + 1) % 50 == 0:
                draft_img_grid = make_grid(invStyleTrans(stylized), nrow=4, scale_each=True)
                style_source_grid = make_grid(si[0], nrow=4, scale_each=True)
                content_img_grid = make_grid(ci[0], nrow=4, scale_each=True)
                etf_grid = make_grid(etf, nrow=4, scale_each=True)
                ci_closeup_grid = make_grid(ci[-1], nrow=4, scale_each=True)
                for idx, patch_stylized in enumerate(stylized_patches):
                    styled_img_grid = make_grid(invStyleTrans(patch_stylized), nrow=4, scale_each=True)
                    version = '' if idx ==0 else str(idx)+'_'
                    save_image(styled_img_grid, args.save_dir + '/drafting_revision_'+version+'iter' + str(n + 1) + '.jpg')
                save_image(draft_img_grid,
                           args.save_dir + '/drafting_draft_iter' + str(n + 1) + '.jpg')
                save_image(ci_closeup_grid,
                           args.save_dir + '/ci_patch_iter' + str(n + 1) + '.jpg')
                save_image(etf_grid,
                           args.save_dir + '/etf_iter' + str(n + 1) + '.jpg')
                save_image(invTrans(content_img_grid),
                           args.save_dir + '/drafting_training_iter_ci' + str(
                               n + 1) + '.jpg')
                save_image(invStyleTrans(style_source_grid),
                           args.save_dir + '/drafting_training_iter_si' + str(
                               n + 1) + '.jpg')
                del(draft_img_grid, styled_img_grid, style_source_grid, content_img_grid)

            if (n + 1) % args.save_model_interval == 0 or (n + 1) == args.max_iter:
                state_dict = dec_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'decoder_iter_{:d}.pth.tar'.format(n + 1))
                state_dict = dec_optimizer.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'dec_optimizer.pth.tar')
                for idx in range(num_rev):
                    num = '' if idx == 0 else '_'+str(idx+1)
                    state_dict = rev_[idx].state_dict()
                    torch.save(copy.deepcopy(state_dict), save_dir /
                               'revisor{:s}_iter_{:d}.pth.tar'.format(num,n + 1))
                    state_dict = rev_optimizer[idx].state_dict()
                    torch.save(copy.deepcopy(state_dict), save_dir /
                               'rev_optimizer{:s}.pth.tar'.format(num))
                    state_dict = disc2_[idx].state_dict()
                    torch.save(copy.deepcopy(state_dict), save_dir /
                               'discriminator_{:d}_iter_{:d}.pth.tar'.format(idx+2,n + 1))
                    state_dict = opt_D2[idx].state_dict()
                    torch.save(copy.deepcopy(state_dict), save_dir /
                               'disc{:d}_optimizer.pth.tar'.format(idx+2))
                state_dict = disc_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'discriminator_iter_{:d}.pth.tar'.format(n + 1))
                state_dict = opt_D.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'disc_optimizer.pth.tar')
        #del(ci,si,stylized,patch_stylized,rc_si,loss,loss_D,loss_D2, losses,loss_c, loss_s, content_relt, style_remd, l_identity1, l_identity2, l_identity3, l_identity4, mdog, loss_Gp_GANp, patch_loss, style_contrastive_loss, content_contrastive_loss,loss_cp, loss_sp, content_reltp, style_remdp, l_identity1p, l_identity2p, l_identity3p, l_identity4p, mdogp, loss_Gp_GAN, patch_lossp, style_contrastive_lossp, content_contrastive_lossp, cF, sF, patch_cF, patch_sF)

def adaconv_urst():
    enc_ = torch.jit.trace(build_enc(vgg), (torch.rand((args.batch_size, 3, 256, 256))), strict=False)
    dec_ = net.ThumbAdaConv(batch_size=args.batch_size,s_d=args.s_d).to(device)
    #rev_ = build_rev(args.revision_depth, None)
    random_crop = transforms.RandomCrop(256)
    if args.load_disc == 1:
        path = args.load_model.split('/')
        path_tokens = args.load_model.split('_')
        new_path_func = lambda x: '/'.join(path[:-1]) + '/' + x + "_".join(path_tokens[-2:])
        disc_state = new_path_func('discriminator_')
        #disc2_state = new_path_func('discriminator_2_')
    else:
        disc_state = None
        #disc2_state = None
        init_weights(dec_)
    disc_ = build_disc(
        disc_state, args.disc_depth) #, torch.rand(args.batch_size, 3, 256, 256).to(torch.device('cuda')), check_trace=False, strict=False)
    #disc2_ = build_disc(disc2_state, args.disc2_depth)
    dec_optimizer = torch.optim.AdamW(dec_.parameters(recurse=True), lr=args.lr)
    #rev_optimizer = torch.optim.AdamW(rev_.parameters(recurse=True), lr=args.lr)
    opt_D = torch.optim.AdamW(disc_.parameters(recurse=True), lr=args.disc_lr)
    #opt_D2 = torch.optim.AdamW(disc2_.parameters(recurse=True), lr=args.disc_lr)
    #grid = 2 * torch.arange(512).view(1,512).float() / max(float(512) - 1., 1.) - 1.
    #grid = (grid * grid.T).to(device)[:256,:256]
    #grid.requires_grad = False
    if args.load_model == 'none':
        init_weights(dec_)
    else:
        dec_.load_state_dict(torch.load(args.load_model), strict=False)
        if args.load_rev==1:
            rev_.load_state_dict(torch.load(new_path_func('revisor_')),strict=False)
        if args.load_optimizer==1:
            try:
                dec_optimizer.load_state_dict(torch.load('/'.join(args.load_model.split('/')[:-1])+'/rev_opt.pth.tar'))
            except:
                'optimizer not loaded '
            #try:
            #    rev_optimizer.load_state_dict(torch.load('/'.join(args.load_model.split('/')[:-1])+'/rev_opt.pth.tar'))
            #except:
            #    'rev_optimizer not loaded'
            try:
                opt_D.load_state_dict(torch.load('/'.join(args.load_model.split('/')[:-1])+'/disc_optimizer.pth.tar'))
            except:
                'discriminator optimizer not loaded'
            #try:
            #    opt_D2.load_state_dict(torch.load('/'.join(args.load_model.split('/')[:-1])+'/disc2_optimizer.pth.tar'))
            #except:
            #    'discriminator optimizer not loaded'
        dec_optimizer.lr = args.lr
    dec_.train()
    enc_.to(device)
    remd_loss = True if args.remd_loss == 1 else False
    for n in tqdm(range(args.max_iter), position=0):
        adjust_learning_rate(dec_optimizer, n // args.accumulation_steps, args)
        #adjust_learning_rate(rev_optimizer, n // args.accumulation_steps, args)
        adjust_learning_rate(opt_D, n // args.accumulation_steps, args, disc=True)
        #adjust_learning_rate(opt_D2, n // args.accumulation_steps, args, disc=True)
        with torch.no_grad():
            ci = next(content_iter)
            si = next(style_iter)

            ######
            ci_ = ci[1:]
            ci_ = torch.cat([ci_, ci[0:1]], 0)
            ci = torch.cat([ci, ci_], 0)
            rc_si = random_crop(si)
            si = torch.cat([si, si], 0)
            rc_si = torch.cat([rc_si, rc_si], 0)
            ######

            ci = [F.interpolate(ci, size=256, mode='bicubic').to(device), ci[:,:,:256,:256].to(device)]
            si = [F.interpolate(si, size=256, mode='bicubic').to(device), rc_si.to(device)]
            ci = [normalize(ci[0]),normalize(ci[1])]
            si = [normalize(si[0]), normalize(si[1])]
            cF = enc_(ci[0])
            sF = enc_(si[0])

            stylized, style_embedding, patch_stats = dec_(cF, sF['r4_1'], None)
            patch_cF = enc_(ci[-1])
            #patch_stylized = rev_(res_in, style_embedding)

            #patch_stylized, _, _, _ = dec_(patch_cF, style_embedding,None,saved_stats = patch_stats,precalced_emb=True)

        for param in disc_.parameters():
            param.grad = None
        #for param in disc2_.parameters():
         #   param.grad = None

        set_requires_grad(disc_, True)
        #set_requires_grad(disc2_, True)
        set_requires_grad(dec_, False)
        #set_requires_grad(rev_, False)
        si[0].requires_grad=True
        #si[-1].requires_grad = True
        #loss_D2 = torch.utils.checkpoint.checkpoint(disc2_.losses,si[-1], patch_stylized)
        loss_D = torch.utils.checkpoint.checkpoint(disc_.losses, si[0], stylized)

        loss_D.backward()
        #loss_D2.backward()
        opt_D.step()
        #opt_D2.step()

        set_requires_grad(disc_, False)
        #set_requires_grad(disc2_, False)
        set_requires_grad(dec_, True)
        #set_requires_grad(rev_, True)

        for param in dec_.parameters():
            param.grad = None
        dummy = torch.ones(1).requires_grad_(True)
        stylized, style_embedding, patch_stats = dec_(cF,sF['r4_1'], dummy)

        patches = []
        original = []
        with torch.no_grad():
            res_in = F.interpolate(stylized[:,:,:128,:128], 256,mode='bicubic')
            original.append(res_in)
        #for param in rev_.parameters():
        #    param.grad = None
        #patch_stylized = rev_(res_in.clone().detach().requires_grad_(True), style_embedding.clone().detach().requires_grad_(True))
        dummy2 = torch.ones(1).requires_grad_(True)
        patch_stylized, _, _ = dec_(patch_cF, style_embedding, dummy2, saved_stats=patch_stats, precalced_emb=True)
        patches.append(patch_stylized)

        losses = calc_losses(stylized, ci[0], si[0], cF, enc_, dec_, None, disc_,
                             calc_identity=args.identity_loss == 1, disc_loss=True,
                             mdog_losses=args.mdog_loss, content_all_layers=args.content_all_layers,
                             remd_loss=remd_loss, contrastive_loss=args.style_contrastive_loss == 1,
                             patch_loss=True, sF=sF, patch_stylized=patches, top_level_patch=original,
                             split_style=False,style_embedding=style_embedding)
        loss_c, loss_s, content_relt, style_remd, l_identity1, l_identity2, l_identity3, l_identity4, mdog, loss_Gp_GAN, patch_loss, style_contrastive_loss, content_contrastive_loss = losses
        loss = loss_c * args.content_weight + args.style_weight * loss_s + content_relt * args.content_relt + style_remd * args.style_remd + patch_loss * args.patch_loss + \
               loss_Gp_GAN * args.gan_loss + mdog + l_identity1 * 50 + l_identity2 + l_identity3 * 50 + l_identity4 + \
               style_contrastive_loss * 0.6 + content_contrastive_loss * 0.5
        '''
        with torch.no_grad():
            patch_sF = enc_(si[-1])
        p_losses = calc_losses(patch_stylized, ci[-1], si[-1], patch_cF, enc_, dec_, None, disc2_,
                               calc_identity=False, disc_loss=True,
                               mdog_losses=args.mdog_loss,
                               content_all_layers=args.content_all_layers,
                               remd_loss=remd_loss, contrastive_loss=False,
                               patch_loss=True, patch_stylized=patches, top_level_patch=original,
                               sF=patch_sF, split_style=False,style_embedding=style_embedding)
        loss_cp, loss_sp, content_reltp, style_remdp, l_identity1p, l_identity2p, l_identity3p, l_identity4p, mdogp, loss_Gp_GANp, patch_lossp, style_contrastive_lossp, content_contrastive_lossp = p_losses
        loss = loss + (
                    loss_cp * args.content_weight + args.style_weight * loss_sp + content_reltp * args.content_relt + style_remdp * args.style_remd + patch_lossp * args.patch_loss + \
                    loss_Gp_GANp * args.gan_loss2 +\
                    style_contrastive_lossp * 0.8 + content_contrastive_lossp * 0.3)
        '''
        loss.backward()
        #rev_optimizer.step()
        dec_optimizer.step()

        if (n + 1) % 10 == 0:

            loss_dict = {}
            for l, s in zip(
                    [dec_optimizer.param_groups[0]['lr'], loss, loss_c, loss_s, style_remd, content_relt, patch_loss,
                     mdog, loss_Gp_GAN, loss_D,style_contrastive_loss, content_contrastive_loss,
                     l_identity1,l_identity2,l_identity3,l_identity4],
                    ['LR','Loss', 'Content Loss', 'Style Loss', 'Style REMD', 'Content RELT',
                     'Patch Loss', 'MXDOG Loss', 'Decoder Disc. Loss','Discriminator Loss',
                     'Style Contrastive Loss','Content Contrastive Loss',
                     "Identity 1 Loss","Identity 2 Loss","Identity 3 Loss","Identity 4 Loss"]):
                if type(l) == torch.Tensor:
                    loss_dict[s] = l.item()
                elif type(l) == float or type(l)==int:
                    if l != 0:
                        loss_dict[s] = l
            if(n +1) % 10 ==0:
                loss_dict['example'] = wandb.Image(stylized[0].transpose(2, 0).transpose(1, 0).detach().cpu().numpy())
            print(str(n)+'/'+str(args.max_iter)+': '+'\t'.join([str(k) + ': ' + str(v) for k, v in loss_dict.items()]))

            wandb.log(loss_dict, step=n)

        with torch.no_grad():
            if (n + 1) % 50 == 0:

                stylized = stylized.float().to('cpu')
                draft_img_grid = make_grid(stylized, nrow=4, scale_each=True)
                styled_img_grid = make_grid(patch_stylized, nrow=4, scale_each=True)
                style_source_grid = make_grid(si[0], nrow=4, scale_each=True)
                content_img_grid = make_grid(ci[0], nrow=4, scale_each=True)
                save_image(invTrans(styled_img_grid), args.save_dir + '/drafting_revision_iter' + str(n + 1) + '.jpg')
                save_image(invTrans(draft_img_grid),
                           args.save_dir + '/drafting_draft_iter' + str(n + 1) + '.jpg')
                save_image(invTrans(content_img_grid),
                           args.save_dir + '/drafting_training_iter_ci' + str(
                               n + 1) + '.jpg')
                save_image(invTrans(style_source_grid),
                           args.save_dir + '/drafting_training_iter_si' + str(
                               n + 1) + '.jpg')
                del(draft_img_grid, styled_img_grid, style_source_grid, content_img_grid)

            if (n + 1) % args.save_model_interval == 0 or (n + 1) == args.max_iter:
                state_dict = dec_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'decoder_iter_{:d}.pth.tar'.format(n + 1))
                state_dict = dec_optimizer.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'dec_optimizer.pth.tar')
                '''
                state_dict = rev_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'revisor_iter_{:d}.pth.tar'.format(n + 1))
                state_dict = rev_optimizer.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'rev_optimizer.pth.tar')
                '''
                state_dict = disc_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'discriminator_iter_{:d}.pth.tar'.format(n + 1))
                state_dict = opt_D.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'disc_optimizer.pth.tar')
                '''
                state_dict = disc2_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'discriminator_2_iter_{:d}.pth.tar'.format(n + 1))
                state_dict = opt_D2.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'disc2_optimizer.pth.tar')
                '''
        #del(ci,si,stylized,patch_stylized,rc_si,loss,loss_D,loss_D2, p_losses,losses,loss_c, loss_s, content_relt, style_remd, l_identity1, l_identity2, l_identity3, l_identity4, mdog, loss_Gp_GANp, patch_loss, style_contrastive_loss, content_contrastive_loss,loss_cp, loss_sp, content_reltp, style_remdp, l_identity1p, l_identity2p, l_identity3p, l_identity4p, mdogp, loss_Gp_GAN, patch_lossp, style_contrastive_lossp, content_contrastive_lossp, cF, sF, patch_cF, patch_sF)


def vq_train():
    dec_ = net.VQGANTrain(args.vgg)
    init_weights(dec_)
    dec_.train()
    dec_.to(device)
    optimizer = torch.optim.Adam(dec_.parameters(), lr=args.lr)
    for i in tqdm(range(args.max_iter)):
        lr = warmup_lr_adjust(optimizer, i)
        ci = next(content_iter).to(device)
        si = next(style_iter).to(device)
        stylized, l = dec_(ci, si)

        set_requires_grad(disc_, True)
        loss_D = disc_.losses(si.detach(),stylized.detach())
        loss_D.backward()
        d_optimizer.step()
        d_optimizer.zero_grad()

        set_requires_grad(disc_,False)
        losses = calc_losses(stylized, ci, si, cF, sF, enc_, dec_, calc_identity=True)
        loss_c, loss_s, loss_r, loss_ss, l_identity1, l_identity2, l_identity3, l_identity4, mdog, codebook_loss, debug_cX = losses
        loss = loss_c * args.content_weight + loss_s * args.style_weight +\
                    l_identity1 * 50 + l_identity2 * 1 +\
                    loss_r * 14 + 26*loss_ss + mdog * .65 + l + codebook_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i + 1) % 10 == 0:
            print(f'lr: {lr:.7f} loss: {l.item():.3f} content_codebook: {l1.item()} style_codebook: {l2.item()}')
        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            print(l)
            state_dict = dec_.state_dict()
            torch.save(state_dict, save_dir /
                       'vqgan{:d}.pth.tar'.format(i + 1))
    writer.close()

if args.train_model == 'drafting':
    drafting_train()
elif args.train_model == 'revision':
    revision_train()
elif args.train_model == 'revlap':
    revlap_train()
elif args.train_model == 'adaconv_thumb':
    adaconv_thumb_train()
elif args.train_model == 'vqvae_pretrain':
    vq_train()
elif args.train_model == 'adaconv_urst':
    adaconv_urst()
import argparse
from pathlib import Path
from revlap import LapRev

import copy
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFile
import wandb
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import re, os
import math
import vgg
import net
import random
from function import _clip_gradient, setup_torch, init_weights, PositionalEncoding2D, get_embeddings
from losses import GANLoss
from modules import RiemannNoise
from net import calc_losses, calc_patch_loss, calc_GAN_loss, calc_GAN_loss_from_pred
from sampler import InfiniteSamplerWrapper, SequentialSamplerWrapper, SimilarityRankedSampler
from torch.cuda.amp import autocast, GradScaler
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

ac_enabled = False

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
parser.add_argument('--revision_depth', type=int, default=1)
parser.add_argument('--disc_depth', type=int, default=5)
parser.add_argument('--disc_channels', type=int, default=64)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--revision_full_size_depth', type=int, default=1)
parser.add_argument('--content_relt', type=float, default=18.5)
parser.add_argument('--style_remd', type=float, default=22.0)
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
parser.add_argument('--momentumnet_beta', type=float, default=.9)
parser.add_argument('--fp16', type=int, default=0)
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

def build_enc(vgg,device):
    enc = net.Encoder(vgg).to(device)
    set_requires_grad(enc, False)
    enc.train(False)
    return enc

content_tf = train_transform(args.load_size, args.crop_size)
style_tf = train_transform(args.style_load_size, args.crop_size)

def get_vgg(args):
    vgg = vgg.vgg

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children()))
    return vgg

def get_datasets(args, content_tf,style_tf):
    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)
    return content_dataset, style_dataset

def make_dataloader(args, content_dataset, style_dataset):
    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))
    return content_iter, style_iter

remd_loss = True if args.remd_loss==1 else 0
mdog_loss = True if args.mdog_loss==1 else 0


def build_rev(depth, state):
    rev = net.Revisors(levels=args.revision_depth, batch_size=args.batch_size).to(device)
    #if not state is None:
    #    state = torch.load(state)
    #    rev.load_state_dict(state, strict=False)
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

def build_disc(disc_state,device):
    with autocast(enabled=ac_enabled):
        disc = net.SpectralDiscriminator(depth=args.revision_depth, num_channels=args.disc_channels).to(device)
        disc.train()
        if not disc_state is None:
            disc.load_state_dict(torch.load(disc_state), strict=False)
        else:
            init_weights(disc)
        disc.init_spectral_norm()
    return disc

def drafting_train():
    enc_ = torch.jit.trace(build_enc(vgg),(torch.rand((args.batch_size,3,128,128))), strict=False)
    enc_.train(False)
    dec_ = net.DecoderAdaConv(args.batch_size)
    if args.load_model == 'none':
        init_weights(dec_)
    else:
        dec_.load_state_dict(torch.load(args.load_model), strict=False)
    dec_.train()
    #disc_ = net.Style_Guided_Discriminator(depth=9, num_channels=64)
    #disc_.train()
    enc_.to(device)
    dec_.to(device)
    #disc_.to(device)

    crop128 = transforms.RandomCrop(128)
    wandb.watch(dec_, log='all', log_freq=10)
    scaler = GradScaler()
    optimizer = torch.optim.Adam(dec_.parameters(), lr=args.lr)
    if args.draft_disc:
        disc_ = build_disc(None)
        opt_D = torch.optim.Adam(disc_.parameters(),lr=args.lr, weight_decay = .1)

    for i in tqdm(range(args.max_iter)):
        adjust_learning_rate(optimizer, i, args)
        if args.draft_disc:
            adjust_learning_rate(opt_D, i, args, disc=True)
        with autocast(enabled=ac_enabled):
            ci = next(content_iter).to(device)
            si = next(style_iter).to(device)
            cF = enc_(ci)
            sF = enc_(si)
            #dec_.apply(lambda x: x.set_random() if hasattr(x,'set_random') else 0)
            optimizer.zero_grad(set_to_none=True)
            stylized, style = dec_(sF, cF)

            if args.draft_disc:
                set_requires_grad(disc_, True)
                with autocast(enabled=ac_enabled):
                    loss_D = calc_GAN_loss(crop128(si.detach()), crop128(stylized.clone().detach()), disc_)
                if ac_enabled:
                    d_scaler.scale(loss_D).backward()
                    if i % args.accumulation_steps == 0:
                        d_scaler.step(opt_D)
                        d_scaler.update()
                else:
                    loss_D.backward()
                    opt_D.step()
                    opt_D.zero_grad()
                set_requires_grad(disc_, False)
            else:
                loss_D = 0

            losses = calc_losses(stylized, ci, si, cF, enc_, dec_, None, disc_ if args.draft_disc else None,
                                        calc_identity=False, disc_loss=args.draft_disc, patch_disc=True,
                                        mdog_losses=args.mdog_loss, content_all_layers=args.content_all_layers,
                                        remd_loss=remd_loss,
                                        patch_loss=False, sF=sF, split_style=args.split_style)
            loss_c, loss_s, content_relt, style_remd, l_identity1, l_identity2, l_identity3, l_identity4, mdog, loss_Gp_GAN, patch_loss = losses
            loss = loss_c * args.content_weight + args.style_weight * loss_s + l_identity1*50+ l_identity2+ l_identity3*50+ l_identity4+content_relt * args.content_relt + style_remd * args.style_remd + loss_Gp_GAN * args.gan_loss

        if ac_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if (i + 1) % 10 == 0:
            loss_dict = {}
            for l, s in zip([loss, loss_c,loss_s,style_remd,content_relt, mdog_loss, l_identity1, l_identity2, l_identity3, l_identity4, stylized, loss_Gp_GAN, loss_D],
                ['Loss', 'Content Loss', 'Style Loss','Style REMD','Content RELT', 'MDOG Loss', 'Identity Loss 1', 'Identity Loss 2', 'Identity Loss 3', 'Identity Loss 4','example', 'Decoder Disc. Loss','Discriminator Loss']):
                if s == 'example':
                   loss_dict[s] = wandb.Image(l[0].transpose(2,0).transpose(1,0).detach().cpu().numpy())
                elif type(l)==torch.Tensor:
                   loss_dict[s] = l.item()
            print('\n')
            print('\t'.join([str(k)+': '+str(v) for k,v in loss_dict.items()]))

            wandb.log(loss_dict, step=i)

        with torch.no_grad():
            if (i + 1) % 50 == 0:
                stylized = stylized.float().to('cpu')
                styled_img_grid = make_grid(stylized, nrow=4, scale_each=True)
                style_source_grid = make_grid(si.float().to('cpu'), nrow=4, scale_each=True)
                content_img_grid = make_grid(ci.float().to('cpu'), nrow=4, scale_each=True)
                out_images = make_grid([content_img_grid,style_source_grid,styled_img_grid], nrow=1)
                save_image(out_images.detach(), args.save_dir+'/drafting_training_iter'+str(i+1)+'.jpg')

            if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
                state_dict = dec_.state_dict()
                torch.save(state_dict, save_dir /
                           'decoder_iter_{:d}.pth.tar'.format(i + 1))
def revision_train():
    random_crop = transforms.RandomCrop(512 if args.split_style else 256)
    with autocast(enabled=ac_enabled):

        enc_ = torch.jit.trace(build_enc(vgg),(torch.rand((args.batch_size,3,256,256))), strict=False)
        dtype = torch.half if args.fp16 else torch.float
        dec_ = torch.jit.trace(net.DecoderAdaConv(batch_size=args.batch_size).to(device),({'r4_1': torch.rand(args.batch_size,512,32,32,dtype=dtype,device='cuda:0'),
                                                                                'r3_1': torch.rand(args.batch_size,256,64,64,dtype=dtype,device='cuda:0'),
                                                                                'r2_1': torch.rand(args.batch_size,128,128,128,dtype=dtype,device='cuda:0'),
                                                                                'r1_1': torch.rand(args.batch_size,64,256,256,dtype=dtype,device='cuda:0'),},
                                                                               {'r4_1': torch.rand(args.batch_size, 512,
                                                                                                   32, 32,dtype=dtype,device='cuda:0'),
                                                                                'r3_1': torch.rand(args.batch_size, 256,
                                                                                                   64, 64,dtype=dtype,device='cuda:0'),
                                                                                'r2_1': torch.rand(args.batch_size, 128,
                                                                                                   128, 128,dtype=dtype,device='cuda:0'),
                                                                                'r1_1': torch.rand(args.batch_size, 64,
                                                                                                   256, 256,dtype=dtype,device='cuda:0'), }
                                                                               ), check_trace=False, strict=False)
        init_weights(dec_)
        #dec_.load_state_dict(torch.load(args.load_model))
        disc_quant = True if args.disc_quantization == 1 else False
        #set_requires_grad(dec_, True)
        disc_state = None
        if args.load_rev == 1 or args.load_disc == 1:
            path = args.load_model.split('/')
            path_tokens = args.load_model.split('_')
            new_path_func = lambda x: '/'.join(path[:-1])+'/'+x+"_".join(path_tokens[-2:])
            if args.load_disc == 1:
                disc_state = new_path_func('discriminator_')
            if args.load_rev == 1:
                rev_state = new_path_func('revisor_')
        elif args.revision_depth>1:
            path = args.load_model.split('/')
            path_tokens = args.load_model.split('_')
            new_path_func = lambda x: '/'.join(path[:-1]) + '/' + x + "_".join(path_tokens[-2:])
            rev_state = new_path_func('revisor_')
        else:
            rev_state = None

        rev_ = torch.jit.trace(build_rev(args.revision_depth, rev_state),(torch.rand(args.batch_size,3,256,256,dtype=dtype,device='cuda:0'),torch.rand(args.batch_size,3,2048,2048,dtype=dtype,device='cuda:0'),torch.rand(args.batch_size,512,4,4,dtype=dtype,device='cuda:0'),torch.randint(256, (args.revision_depth, 2),device='cuda:0',dtype=torch.int32)), check_trace=False, strict=False)
        #disc_inputs = {'forward': (
        #torch.rand(args.batch_size, 3, 256, 256).to(device), torch.rand(args.batch_size, 320, 4, 4).to(device)),
        #'losses': (torch.rand(args.batch_size, 3, 512, 512).to(device), torch.rand(args.batch_size, 3, 256, 256).to(device), torch.rand(args.batch_size,320,4,4).to(device)),
        #'get_ganloss': (torch.rand(args.batch_size,1,256,256).to(device),torch.Tensor([True]).to(device))}
        disc_ = build_disc(disc_state)#, torch.rand(args.batch_size, 3, 256, 256).to(device).detach(), strict=False)
        disc_.train()

        #if not disc_state is None:
        #    disc_.load_state_dict(torch.load(new_path_func('discriminator_')), strict=False)
        #else:
        init_weights(disc_)
        rev_.train()
        dec_.train()
        enc_.to(device)
        dec_.to(device)
        disc_.to(device)
        rev_.to(device)
    remd_loss = True if args.remd_loss==1 else False
    scaler = GradScaler(init_scale=128)
    d_scaler = GradScaler(init_scale=128)
    optimizers = [torch.optim.AdamW(list(dec_.parameters()), lr=args.lr)]
    #for i in rev_.layers:
    #    optimizers.append(torch.optim.AdamW(list(i.parameters()), lr=args.lr))
    optimizers.append(torch.optim.AdamW(list(rev_.parameters())+list(dec_.parameters()), lr=args.lr))
    opt_D = torch.optim.SGD(disc_.parameters(), lr=args.disc_lr, momentum = .5)
    for i in tqdm(range(args.max_iter)):
        choice = random.randrange(args.revision_depth+1)
        for optimizer in optimizers:
            adjust_learning_rate(optimizer, i//args.accumulation_steps, args)
        adjust_learning_rate(opt_D, i//args.accumulation_steps, args, disc=True)
        with autocast(enabled=ac_enabled):
            ci = next(content_iter).to(device)
            si = next(style_iter).to(device)
            ci = [F.interpolate(ci, size=256, mode='bicubic', align_corners=False), ci]
            si = [F.interpolate(si, size=256, mode='bicubic', align_corners=False), si]
            cF = enc_(ci[0])
            sF = enc_(si[0])
            stylized, style = dec_(sF, cF)

            crop_marks = torch.randint(256, (args.revision_depth, 2)).int().to(device)
            crop_marks.requires_grad = False
            #embeddings = get_embeddings(pos_embeddings, crop_marks)

            rev_outputs, ci_patches, patches = rev_(stylized, ci[-1].detach(), style, crop_marks)
            N, C, h, w = ci[0].shape
            ci_patches = torch.cat([ci[0].view(1,N,C,h,w), ci_patches],dim=0)
            cropped_si = [si[0]]
            p = torch.zeros_like(patches)[0]
            N,C,h,w = p.shape
            patches=torch.cat([p.view(1,N,C,h,w),patches],dim=0)

            patch_feats = [torch.zeros(1,device='cuda:0')]

            size = 256
            for idx in range(args.revision_depth):
                size *= 2
                scaled_si = F.interpolate(si[-1], size=size, mode='bicubic',
                                          align_corners=False).detach()
                for j in range(idx + 1):
                    tl = (crop_marks[j][0] * 2 ** (idx - j)).int()
                    tr = (tl + (512 * 2 ** (idx - 1 - j))).int()
                    bl = (crop_marks[j][1] * 2 ** (idx - j)).int()
                    br = (bl + (512 * 2 ** (idx - 1 - j))).int()
                    scaled_si = scaled_si[:, :, tl:tr, bl:br]
                cropped_si.append(scaled_si.detach())
            for stylized_patch in patches[1:]:
                patch_feats.append(enc_(stylized_patch))
        cropped_si = torch.stack(cropped_si)



        set_requires_grad(disc_, True)
        loss_D = calc_GAN_loss(cropped_si.clone().detach().float(), rev_outputs.clone().detach().float(), crop_marks, disc_)
        loss_D.backward()
        if i % args.accumulation_steps == 0:
            opt_D.step()
            opt_D.zero_grad(set_to_none=True)

        set_requires_grad(disc_, False)

        if args.split_style:
            del(cropped_si[1:])
            size = 512
            for i in range(args.revision_depth):
                scaled_si = F.interpolate(si[-1], size=size, mode='bicubic',
                                          align_corners=False).detach()
                if size != 512:
                    scaled_si = random_crop(scaled_si)
                cropped_si.append(scaled_si)

        for idx in [random.randrange(args.revision_depth+1)]:
            ploss = False if idx==0 else True
            if idx != 0:
                cF = enc_(ci_patches[idx])
                sF = enc_(cropped_si[idx])

            with autocast(enabled=ac_enabled):
                losses = calc_losses(rev_outputs,
                                     ci_patches[idx],
                                     cropped_si[idx],
                                     cF, enc_, dec_,
                                     patch_feats[idx],
                                     None,
                                     calc_identity=False, disc_loss=False,
                                     mdog_losses=args.mdog_loss,
                                     content_all_layers=args.content_all_layers,
                                     remd_loss=remd_loss if idx != 0 else False, patch_loss=ploss,
                                     sF=sF, split_style = args.split_style if idx != 0 else False,
                                     rev_depth=idx)
                loss_c, loss_s, content_relt, style_remd, l_identity1, l_identity2, l_identity3, l_identity4, mdog, loss_Gp_GAN, patch_loss = losses
                loss = loss_c * args.content_weight + args.style_weight * loss_s + content_relt * 16 + style_remd * 10 + loss_Gp_GAN * args.gan_loss + patch_loss * args.patch_loss + mdog
        pred_fake = disc_(rev_outputs.float(), crop_marks)
        loss_D_fake = calc_GAN_loss_from_pred(pred_fake, True)

        loss = loss + loss_D_fake * args.gan_loss

        if ac_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if i % args.accumulation_steps == 0:
            if ac_enabled:
                for idx, optimizer in enumerate(optimizers):
                    scaler.step(optimizer)
                scaler.update()
            else:
                for optimizer in optimizers:
                    optimizer.step()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)


        if (i + 1) % 10 == 0:
            loss_dict = {}
            for l, s in zip([loss, loss_c, loss_s, style_remd, content_relt, loss_D_fake,loss_D, stylized,patch_loss, mdog],
                            ['Loss', 'Content Loss', 'Style Loss', 'Style REMD', 'Content RELT',
                            'Revision Disc. Loss','Discriminator Loss','example','Patch Loss', 'MXDOG Loss']):
                if s == 'example':
                   loss_dict[s] = wandb.Image(l[0].transpose(2,0).transpose(1,0).detach().cpu().numpy())
                elif type(l) == torch.Tensor:
                    loss_dict[s] = l.item()
            print('\t'.join([str(k) + ': ' + str(v) for k, v in loss_dict.items()]))

            wandb.log(loss_dict, step=i)
            print(f'{loss.item():.2f}')


        with torch.no_grad():
            if (i + 1) % 50 == 0:
                rev_outputs = torch.vstack([i.float().to('cpu') for i in rev_outputs])

                draft_img_grid = make_grid(rev_outputs, nrow=args.batch_size, scale_each=True)
                si[-1] = F.interpolate(si[-1], size=256, mode='bicubic')
                ci[-1] = F.interpolate(ci[-1], size=256, mode='bicubic')
                style_source_grid = make_grid(si[-1], nrow=4, scale_each=True)
                content_img_grid = make_grid(ci[-1], nrow=4, scale_each=True)
                save_image(draft_img_grid.detach(),
                           args.save_dir + '/drafting_revision_iter' + str(i + 1) + '.jpg')
                save_image(content_img_grid.detach(),
                           args.save_dir + '/drafting_training_iter_ci' + str(
                               i + 1) + '.jpg')
                save_image(style_source_grid.detach(),
                           args.save_dir + '/drafting_training_iter_si' + str(
                               i + 1) + '.jpg')

            if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
                print(loss)
                state_dict = rev_.state_dict()
                torch.save(state_dict, save_dir /
                           'revisor_iter_{:d}.pth.tar'.format(i + 1))
                state_dict = dec_.state_dict()
                torch.save(state_dict, save_dir /
                           'decoder_iter_{:d}.pth.tar'.format(i + 1))
                state_dict = disc_.state_dict()
                torch.save(state_dict, save_dir /
                           'discriminator_iter_{:d}.pth.tar'.format(i + 1))
                state_dict = optimizers[0].state_dict()
                torch.save(state_dict, save_dir /
                           'optimizer.pth.tar')
                state_dict = opt_D.state_dict()
                torch.save(state_dict, save_dir /
                           'disc_optimizer.pth.tar')

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

            stylized, style = dec_(sF, cF)

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

def adaconv_thumb_train(index, args):
    torch.manual_seed(1)
    device = xm.xla_device()

    if not xm.is_master_ordinal():
        xm.rendezvous('load_only_once')

    content_dataset, style_dataset = get_datasets(args, content_tf, style_tf)
    vgg = get_vgg(args)

    if xm.is_master_ordinal():
        xm.rendezvous('load_only_once')

    content_sampler = torch.utils.data.distributed.DistributedSampler(
        content_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    style_sampler = torch.utils.data.distributed.DistributedSampler(
        style_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler= content_sampler,
        shuffle=False,
        num_workers=args.n_threads,
        drop_last=True))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler= style_sampler,
        shuffle=False,
        num_workers=args.n_threads,
        drop_last=True))

    enc_ = torch.jit.trace(build_enc(vgg,device), (torch.rand((args.batch_size, 3, 256, 256).to(device))), strict=False)
    dec_ = net.ThumbAdaConv(batch_size=args.batch_size).to(device)
    if args.load_disc == 1:
        path = args.load_model.split('/')
        path_tokens = args.load_model.split('_')
        new_path_func = lambda x: '/'.join(path[:-1]) + '/' + x + "_".join(path_tokens[-2:])
        disc_state = new_path_func('discriminator_')

    else:
        disc_state = None
        init_weights(dec_)
    disc_ = build_disc(
        disc_state, device)  # , torch.rand(args.batch_size, 3, 256, 256).to(torch.device('cuda')), check_trace=False, strict=False)

    dec_optimizer = torch.optim.Adam(dec_.parameters(recurse=True), lr=args.lr)
    opt_D = torch.optim.AdamW(disc_.parameters(recurse=True), lr=args.disc_lr)
    if args.load_model == 'none':
        init_weights(dec_)
    else:
        dec_.load_state_dict(torch.load(args.load_model), strict=False)
        try:
            dec_optimizer.load_state_dict(torch.load('/'.join(args.load_model.split('/')[:-1])+'/dec_optimizer.pth.tar'))
        except:
            'optimizer not loaded'
        try:
            opt_D.load_state_dict(torch.load('/'.join(args.load_model.split('/')[:-1])+'/disc_optimizer.pth.tar'))
        except:
            'discriminator optimizer not loaded'
        dec_optimizer.lr = args.lr
        dec_.train()
        enc_.to(device)
        remd_loss = True if args.remd_loss == 1 else False

    for n in range(args.max_iter):
        #adjust_learning_rate(dec_optimizer, i // args.accumulation_steps, args)
        ci = next(content_iter).to(device)
        si = next(style_iter).to(device)
        ci = [F.interpolate(ci, size=256, mode='bicubic', align_corners=True), ci]
        si = [F.interpolate(si, size=256, mode='bicubic', align_corners=True), si]
        cF = enc_(ci[0])
        sF = enc_(si[0])

        stylized, style = dec_(sF, cF,patch_num=0)


        patches = []
        thumbnails = []
        original = []
        patch_stylized = stylized
        size = 128
        ci_size = 1024
        for i in range(3):

            original.append(F.interpolate(stylized[:,:,0:size,0:size],256))
            ci_to_crop = ci[-1][:, :, 0:ci_size, 0:ci_size]
            scale = F.interpolate(ci_to_crop,256)
            cF_patch = enc_(scale)

            patch_stylized, _ = dec_(None, cF_patch, style,patch_num=i+1)
            patches.append(patch_stylized)
            size = int(size/2)
            ci_size = int(ci_size/2)


        loss_D = calc_GAN_loss(si[0].detach(), stylized.clone().detach(), None, disc_)


        losses = calc_losses(stylized, ci[0], si[0], cF, enc_, dec_, None, disc_,
                                   calc_identity=args.identity_loss==1, disc_loss=True,
                                   mdog_losses=args.mdog_loss, content_all_layers=False,
                                   remd_loss=remd_loss,
                                   patch_loss=True, patch_stylized = patches, top_level_patch = original, sF=sF, split_style=False)
        loss_c, loss_s, content_relt, style_remd, l_identity1, l_identity2, l_identity3, l_identity4, mdog, loss_Gp_GAN, patch_loss, patch_disc_loss = losses
        loss = loss_c * args.content_weight + args.style_weight * loss_s + content_relt * args.content_relt + style_remd * args.style_remd + patch_loss * args.patch_loss +loss_Gp_GAN*args.gan_loss + patch_disc_loss*args.gan_loss +mdog + l_identity1*50 + l_identity2 + l_identity3*50 + l_identity4

        loss.backward()
        loss_D.backward()
        dec_optimizer.step()
        dec_optimizer.zero_grad()
        opt_D.step()
        opt_D.zero_grad()
        xm.optimizer_step(dec_optimizer)
        xm.optimizer_step(opt_D)

        if xm.is_master_ordinal():
            xm.rendezvous('logging')
        if (n + 1) % 1 == 0:
            loss_dict = {}
            for l, s in zip(
                    [loss, loss_c, loss_s, style_remd, content_relt, patch_loss,
                     mdog, loss_Gp_GAN, loss_D, patch_disc_loss],
                    ['Loss', 'Content Loss', 'Style Loss', 'Style REMD', 'Content RELT',
                     'Patch Loss', 'MXDOG Loss', 'Decoder Disc. Loss','Discriminator Loss',
                     'Patch Disc. Loss']):
                if type(l) == torch.Tensor:
                    loss_dict[s] = l.item()
            if(n +1) % 10 ==0:
                loss_dict['example'] = wandb.Image(stylized[0].transpose(2, 0).transpose(1, 0).detach().cpu().numpy())
            print('\n')
            print(str(n)+'/'+str(args.max_iter)+': '+'\t'.join([str(k) + ': ' + str(v) for k, v in loss_dict.items()]))
            wandb.log(loss_dict, step=n)

        with torch.no_grad():
            if (n + 1) % 50 == 0:

                stylized = stylized.float().to('cpu')
                patch_stylized = torch.vstack(patches).float().to('cpu')
                draft_img_grid = make_grid(stylized, nrow=4, scale_each=True)
                styled_img_grid = make_grid(patch_stylized, nrow=4, scale_each=True)
                style_source_grid = make_grid(si[0], nrow=4, scale_each=True)
                content_img_grid = make_grid(ci[0], nrow=4, scale_each=True)
                save_image(styled_img_grid.detach(), args.save_dir + '/drafting_revision_iter' + str(n + 1) + '.jpg')
                save_image(draft_img_grid.detach(),
                           args.save_dir + '/drafting_draft_iter' + str(n + 1) + '.jpg')
                save_image(content_img_grid.detach(),
                           args.save_dir + '/drafting_training_iter_ci' + str(
                               n + 1) + '.jpg')
                save_image(style_source_grid.detach(),
                           args.save_dir + '/drafting_training_iter_si' + str(
                               n + 1) + '.jpg')

            if (n + 1) % args.save_model_interval == 0 or (n + 1) == args.max_iter:
                state_dict = dec_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'decoder_iter_{:d}.pth.tar'.format(n + 1))

                state_dict = dec_optimizer.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'dec_optimizer.pth.tar')
                state_dict = disc_.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'discriminator_iter_{:d}.pth.tar'.format(n + 1))
                state_dict = opt_D.state_dict()
                torch.save(copy.deepcopy(state_dict), save_dir /
                           'disc_optimizer.pth.tar')
        if xm.is_master_ordinal():
            xm.rendezvous('logging')

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
    xmp.spawn(map_fn, args=(args,), nprocs=8, start_method='fork')
elif args.train_model == 'vqvae_pretrain':
    vq_train()
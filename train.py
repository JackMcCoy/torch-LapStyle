import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import re, os
import math
import vgg
import net
from function import init_weights
from net import calc_losses
from sampler import InfiniteSamplerWrapper, SequentialSamplerWrapper, SimilarityRankedSampler
from torch.cuda.amp import autocast, GradScaler

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

disc_scaler = GradScaler(init_scale=1024,growth_interval=1000)
scaler = GradScaler(init_scale=1024,growth_interval=1000)
ac_enabled = True

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

def warmup_lr_adjust(optimizer, iteration_count, warmup_start=1e-7, warmup_iters=500, max_lr = 1e-4, decay=5e-5):
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

with autocast(enabled=ac_enabled):
    vgg = vgg.vgg

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children()))

    content_tf = train_transform(args.load_size, args.crop_size)
    style_tf_small = train_transform(128, args.crop_size)
    style_tf = train_transform(args.style_load_size, args.crop_size)

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
'''
tmp_dataset_2 = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=SequentialSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
'''
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))


if args.train_model=='drafting':

    with autocast(enabled=ac_enabled):
        enc_ = net.Encoder(vgg)
        set_requires_grad(enc_, False)
        enc_.train(False)
        dec_ = net.DecoderAdaConv()
        disc_ = net.Style_Guided_Discriminator(depth=9, num_channels=64)
        init_weights(dec_)
        init_weights(disc_)
        dec_.train()
        disc_.train()
        enc_.to(device)
        dec_.to(device)
        disc_.to(device)

    optimizer = torch.optim.Adam(dec_.parameters(), lr=args.lr)
    opt_D = torch.optim.Adam(disc_.parameters(),lr=args.lr, weight_decay = .1)
    '''
    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=SimilarityRankedSampler(content_dataset, next(style_iter).to(device), tmp_dataset, tmp_dataset_2, enc_),
        num_workers=1))

    del(tmp_dataset)
    del(tmp_dataset_2)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))
    '''
    for i in tqdm(range(args.max_iter)):
        warmup_lr_adjust(optimizer, i)
        warmup_lr_adjust(opt_D, i)
        with autocast():
            ci = next(content_iter).to(device)
            si = next(style_iter).to(device)
            cF = enc_(ci)
            sF = enc_(si)
            stylized = dec_(sF, cF)

            opt_D.zero_grad()
            set_requires_grad(disc_, True)
            loss_D, style = disc_.losses(si.detach(),stylized.detach(), sF['r1_1'])

        disc_scaler.scale(loss_D).backward()
        disc_scaler.step(opt_D)
        disc_scaler.update()
        set_requires_grad(disc_,False)

        with autocast(enabled=ac_enabled):

            optimizer.zero_grad()
            losses = calc_losses(stylized, ci.detach(), si.detach(), cF, sF, enc_, dec_, style, disc_, calc_identity=False, disc_loss=True, mdog_losses=False)
            loss_c, loss_s, l_identity1, l_identity2, l_identity3, l_identity4, mdog, loss_Gp_GAN = losses
            loss = loss_c * args.content_weight + args.style_weight * loss_s + loss_Gp_GAN * 2.5
            #            content_relt * 25 + l_identity1*50 + l_identity2 * 1 +\
            #            l_identity3* 25 + l_identity4 * .5 + mdog * .33 + loss_Gp_GAN * 5 + cb_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (i + 1) % 10 == 0:
            print(f'{loss.item():.2f}')
            print(f'c: {loss_c.item():.3f} s: {loss_s.item():.3f} gan:{loss_D.item():.3f} gp_gan:{loss_Gp_GAN.item():.3f}')

            writer.add_scalar('loss_content', loss_c.item(), i + 1)
            writer.add_scalar('loss_style', loss_s.item(), i + 1)

        with torch.no_grad():
            if (i + 1) % 100 == 0:
                stylized = stylized.float().to('cpu')
                styled_img_grid = make_grid(stylized, nrow=4, scale_each=True)
                style_source_grid = make_grid(si, nrow=4, scale_each=True)
                content_img_grid = make_grid(ci, nrow=4, scale_each=True)
                save_image(styled_img_grid.detach(), args.save_dir+'/drafting_training_iter'+str(i+1)+'.jpg')
                save_image(content_img_grid.detach(),
                           args.save_dir + '/drafting_training_iter_ci' + str(
                               i + 1) + '.jpg')
                save_image(style_source_grid.detach(),
                           args.save_dir + '/drafting_training_iter_si' + str(
                               i + 1) + '.jpg')

            if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
                print(loss)
                state_dict = dec_.state_dict()
                torch.save(state_dict, save_dir /
                           'decoder_iter_{:d}.pth.tar'.format(i + 1))
    writer.close()
elif args.train_model=='vqgan_pretrain':
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
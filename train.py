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

def warmup_lr_adjust(optimizer, iteration_count, warmup_start=6.5e-8, warmup_iters=1500, max_lr = 1e-4, decay=5e-5):
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

if args.train_model=='drafting':

    enc_ = net.Encoder(vgg)
    set_requires_grad(enc_, False)
    enc_.train(False)
    dec_ = net.DecoderVQGAN()
    disc_ = net.Discriminator(depth=5)
    init_weights(dec_)
    init_weights(disc_)
    dec_.train()
    disc_.train()
    enc_.to(device)
    dec_.to(device)
    disc_.to(device)

    optimizer = torch.optim.Adam(dec_.parameters(), lr=args.lr, weight_decay = .1)
    opt_D = torch.optim.Adam(disc_.parameters(),lr=args.lr, weight_decay = .1)
    for i in tqdm(range(args.max_iter)):
        warmup_lr_adjust(optimizer, i)

        warmup_lr_adjust(opt_D, i)
        ci = next(content_iter).to(device)
        si = next(style_iter).to(device)
        cF = enc_(ci)
        sF = enc_(si)
        stylized, l = dec_(sF, cF)

        opt_D.zero_grad()
        set_requires_grad(disc_, True)
        loss_D = disc_.losses(si.detach(),stylized.detach())

        loss_D.backward()
        opt_D.step()
        set_requires_grad(disc_,False)

        dec_.zero_grad()
        optimizer.zero_grad()
        losses = calc_losses(stylized, ci, si, cF, sF, enc_, dec_, disc_, calc_identity=True, disc_loss=True, mdog_losses=True)
        loss_c, loss_s, loss_r, loss_ss, l_identity1, l_identity2, l_identity3, l_identity4, mdog, codebook_loss, loss_Gp_GAN = losses
        loss = loss_c * args.content_weight + loss_s * args.style_weight +\
                    l_identity1 * 50 + l_identity2 * 1 +l_identity3 * 50 + l_identity4 * 1 +\
                    loss_r * 9 + 16*loss_ss + mdog * .5 + loss_Gp_GAN * 2 + l + codebook_loss
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'{loss.item():.2f}')
            print(f'disc: {loss_D.item():.4f} gan_loss: {loss_Gp_GAN.item():.3f}, c: {loss_c.item():.3f} s: {loss_s.item():.3f} \
            r: {loss_r.item():.3f} ss: {loss_ss.item():.3f} \
            id1: {l_identity1.item():.3f} id2: {l_identity2.item():.3f} + {l_identity3.item():.3f} id2: {l_identity4.item():.3f} \
            mdog_loss: {mdog.item():.3f}, codebook_loss: {l.item():.3f} ident_cb_loss: {codebook_loss.item():.3f}')

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)

        with torch.no_grad():
            if (i + 1) % 100 == 0:
                stylized = stylized.float().to('cpu')
                for j in range(1):
                    save_image(stylized[j].detach(), args.save_dir+'/drafting_training_'+str(j)+'_iter'+str(i+1)+'.jpg')
                    save_image(ci[j].float().detach(),
                               args.save_dir + '/drafting_training_' + str(j) + '_iter_ci' + str(
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
        stylized, l, l1, l2 = dec_(ci, si)

        set_requires_grad(disc_, True)
        loss_D = disc_.losses(si.detach(),stylized.detach())
        loss_D.backward()
        d_optimizer.step()
        d_optimizer.zero_grad()

        set_requires_grad(disc_,False)
        losses = calc_losses(stylized, ci, si, cF, sF, enc_, dec_, calc_identity=True)
        loss_c, loss_s, loss_r, loss_ss, l_identity1, l_identity2, l_identity3, l_identity4, mdog, codebook_loss, debug_cX = losses
        loss = loss_c * args.content_weight + loss_s * args.style_weight +\
                    l_identity1 * 50 + l_identity2 * 1 + l_identity3 * 25 + l_identity4 * 1 +\
                    loss_r * 16 + 10*loss_ss + mdog + l + l1 + l2
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
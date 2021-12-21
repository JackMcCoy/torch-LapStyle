import copy
import revlib
from net import style_encoder_block, RiemannNoise, ResBlock
import torch.nn.functional as F

upsample = nn.Upsample(scale_factor=2, mode='bicubic')
downsample = nn.Upsample(scale_factor=.5, mode='bicubic')

def additive_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return upsample(other_stream),  + fn_out


def additive_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return output - downsample(fn_out)


class RevisorLap(nn.Module):
    def __init__(self, encoder, levels= 1):
        super(RevisorLap, self).__init__()
        self.layers = nn.ModuleList([])
        self.levels = levels
        self.encoder = encoder
        self.ci = None
        self.style_embedding = None
        self.revision_net = RevisionNet(self)
        self.stem = revlib.ReversibleSequential(*[self.revision_net.copy(i) for i in range(levels)],
                                                coupling_forward=[momentum_coupling_forward],
                                                coupling_inverse=[momentum_coupling_inverse],
                                                target_device=ctx.model.device)
        for i in range(levels):
            self.layers.append(RevisionNet(self))

    def forward(self, x, ci, enc_, crop_marks):
        self.ci = ci
        self.style_embedding = style
        x = self.stem(x)
        return x

class RevisionNet(nn.Module):
    def __init__(self, mod):
        super(RevisionNet, self).__init__()

        self.parent = mod
        self.lap_weight = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
        self.lap_weight = torch.Tensor(self.lap_weight).to(torch.device('cuda')).requires_grad(False)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.style_encoding = nn.Sequential(
            *style_encoder_block(512),
            *style_encoder_block(512),
            *style_encoder_block(512)
        )
        self.s_d = 128
        self.style_projection = nn.Sequential(
            nn.Linear(8192, self.s_d * 16)
        )
        self.style_riemann_noise = RiemannNoise(4)

        self.resblock = ResBlock(64)
        self.adaconvs = nn.ModuleList([
            AdaConv(64, 1, s_d=s_d),
            AdaConv(64, 1, s_d=s_d),
            AdaConv(128, 2, s_d=s_d),
            AdaConv(128, 2, s_d=s_d)])

        self.relu = nn.ReLU()
        self.layer_num = 0

        self.riemann_style_noise = RiemannNoise(4)
        self.DownBlock = nn.Sequential(RiemannNoise(256),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(6, 128, kernel_size=3),
            nn.ReLU(),
            RiemannNoise(256),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            RiemannNoise(256),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            RiemannNoise(256),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            RiemannNoise(128),)
        self.UpBlock = nn.ModuleList([nn.Sequential(RiemannNoise(128, first_layer),
                                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(64, 256, kernel_size=3),
                                                    nn.ReLU(),
                                                    RiemannNoise(128, first_layer),
                                                    nn.PixelShuffle(2),
                                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(64, 64, kernel_size=3),
                                                    nn.ReLU(),
                                                    RiemannNoise(256, first_layer)),
                                      nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(64, 128, kernel_size=3),
                                                    nn.ReLU(),
                                                    RiemannNoise(256, first_layer)),
                                      nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(128, 128, kernel_size=3),
                                                    nn.ReLU(),
                                                    RiemannNoise(256, first_layer)),
                                      nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                                    nn.Conv2d(128, 3, kernel_size=3)
                                                    )])

    def copy(self, index):
        out = copy.deepcopy(self)
        out.layer_num = index
        return out

    def thumbnail_style_calc(self, style):
        style = self.parent.encoder(style)
        style = self.style_encoding(style['r4_1'])
        style = style.flatten(1)
        style = self.style_projection(style)
        style = style.reshape(b, self.s_d, 4, 4)
        style = self.riemann_style_noise(style)
        return style

    def recursive_controller(self, x, ci, thumbnail):
        holder = []
        base_case = False
        if x.shape[-1] == 512:
            base_case = True
            thumbnail_style = self.thumbnail_style_calc(thumbnail)

        for i, c in zip(x.chunk(2,dim=2), ci.chunk(2,dim=2)):
            for j, c2 in zip(i.chunk(2,dim=3), ci.chunk(2,dim=3)):
                if base_case:
                    holder.append(self.recursive_controller(j, c2, x))
                else:
                    holder.append(self.generator(j, c2, thumbnail_style))
        holder = torch.cat((torch.cat([holder[0],holder[2]],dim=2),
                            torch.cat([holder[1],holder[3]],dim=2)),dim=3)
        return holder

    def generator(self, x, ci, style):
        lap_pyr = F.conv2d(F.pad(ci.detach(), (1, 1, 1, 1), mode='reflect'), weight=self.lap_weight,
                           groups=3).to(device)
        out = torch.cat([x, lap_pyr], dim=1)

        out = self.DownBlock(out.clone().detach())
        out = self.resblock(out)
        for adaconv, learnable in zip(self.adaconvs, self.UpBlock):
            out = out + adaconv(style, out, norm=True)
            out = learnable(out)
        return out

    def forward(self, input):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """

        input = self.upsample(input)
        size *= 2
        scaled_ci = F.interpolate(self.parent.ci, size=256*2**self.layer_num+1, mode='bicubic', align_corners=False)
        out = recursive_controller(input, scaled_ci, input)
        return out
import torch.nn as nn

from function import adaptive_instance_normalization as adain
from function import calc_mean_std
from modules import ResBlock, ConvBlock
from losses import CalcContentLoss, CalcContentReltLoss, CalcStyleEmdLoss, CalcStyleLoss, GramErrors

decoder_1 = nn.Sequential(
    ResBlock(512),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, kernel_size=3),
    nn.ReLU())

decoder_2 = nn.Sequential(
    ResBlock(256),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, kernel_size=3),
    nn.ReLU()
    )
decoder_3 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, kernel_size=3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, kernel_size=3),
    nn.ReLU()
    )
decoder_4 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, kernel_size=3),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, kernel_size=3),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=1),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, kernel_size=3),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, kernel_size=3),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, kernel_size=3),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, kernel_size=3),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, kernel_size=3),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, kernel_size=3),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, kernel_size=3),
    nn.ReLU()  # relu5-4
)

class Discriminator(nn.Module):
    def __init__(self, depth, num_channels):
        super(Discriminator, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3,num_channels,3,stride=1,padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2)
            )
        self.body = nn.Sequential()
        for i in range(depth - 2):
            self.body.add_sublayer(
                nn.Conv2D(num_channel,
                          num_channel,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            self.body.add_sublayer(nn.BatchNorm2D(num_channel))
            self.body.add_sublayer(nn.LeakyReLU(0.2))
        self.tail = nn.Conv2D(num_channel,
                              1,
                              kernel_size=3,
                              stride=1,
                              padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class Net(nn.Module):
    def __init__(self, encoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:])
        self.decoder_1 = decoder_1
        self.decoder_2 = decoder_2
        self.decoder_3 = decoder_3
        self.decoder_4 = decoder_4

        self.mse_loss = nn.MSELoss()
        self.style_remd_loss = CalcStyleEmdLoss()
        self.content_emd_loss = CalcContentReltLoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def fix_decoder(self):
        for name in ['decoder_1', 'decoder_2', 'decoder_3', 'decoder_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_losses(self,g_t,content_image=None,style_image=None,calc_identity=True):
        g_t_feats = self.encode_with_intermediate(g_t)
        if calc_identity==True:
            Icc = self.decode(self.content_feat,self.content_feat)
            l_identity1 = CalcContentLoss(Icc, content_image)
            Fcc = self.encode_with_intermediate(Icc)
            l_identity2 = CalcContentLoss(Fcc[0], self.content_feat[0])
        else:
            l_identity1 = None
            l_identity2 = None
        loss_c = CalcContentLoss(g_t_feats[0], self.content_feat[0],norm=True)
        loss_s = CalcStyleLoss(g_t_feats[0], self.style_feats[0])
        loss_ss = CalcContentReltLoss(g_t_feats[2], self.content_feat[2])
        loss_ss += CalcContentReltLoss(g_t_feats[3], self.content_feat[3])
        remd_loss = CalcStyleEmdLoss(g_t_feats[2],self.style_feats[2])
        remd_loss += CalcStyleEmdLoss(g_t_feats[3],self.style_feats[3])
        for i in range(1, 6):
            loss_s += CalcStyleLoss(g_t_feats[i], self.style_feats[i])
            loss_c += CalcContentLoss(g_t_feats[i], self.content_feat[i],norm=True)
            if calc_identity==True:
                l_identity2 += CalcContentLoss(Fcc[i], self.content_feat[i])
        return loss_c, loss_s, remd_loss, loss_ss, l_identity1, l_identity2

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def decode(self, content_feat,style_feats,alpha=1):
        t = adain(content_feat[-2], style_feats[-2])
        t = alpha * t + (1 - alpha) * content_feat[-2]
        g_t = self.decoder_1(t)
        m = nn.Upsample(scale_factor=2, mode='nearest')
        g_t = m(g_t)
        #t_2 = UPSCALE CONTENT FEAT!
        t = adain(content_feat[-3], style_feats[-3])
        t = alpha * t + (1 - alpha) * content_feat[-3]

        t = torch.add(g_t,t)
        g_t = self.decoder_2(t)
        t = adain(content_feat[-4], style_feats[-4])
        t = alpha * t + (1 - alpha) * content_feat[-4]
        g_t = m(g_t)
        t = torch.add(g_t,t)
        g_t = self.decoder_3(t)
        g_t = self.decoder_4(g_t)
        return g_t

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        sizes=[64,128,256,512]
        if hasattr(self,'style_feats_fixed'):
            self.style_feats=self.style_feats_fixed
        else:
            self.style_feats = self.encode_with_intermediate(style)
        self.content_feat = self.encode_with_intermediate(content)
        g_t = self.decode(self.style_feats,self.content_feat)
        return g_t

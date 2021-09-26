import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(dim, dim, 3),
                                        nn.ReLU(),
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(dim, dim, 3))

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ConvBlock(nn.Module):

    def __init__(self, dim1, dim2,noise=0):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(dim1, dim2, 3),
                                        nn.ReLU())

    def forward(self, x):
        out = self.conv_block(x)
        return out
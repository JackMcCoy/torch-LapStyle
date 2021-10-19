import torch
from torch import nn

class KernelPredictor(nn.Module):
    def __init__(self, c_in, c_out,, p):
        super(KernelPredictor, self).__init__()
        self.depthwise_kernel_conv = nn.Conv2d()

    def forward(self, style_encoding):
        pass
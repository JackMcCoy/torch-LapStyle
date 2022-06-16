import torch.nn as nn
import torch


vgg = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),  # relu1-1
    # 4
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    # 8
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),  # relu2-1
    # 11
    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    # 15
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(),  # relu3-1
    # 18
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(),  # relu3-2
    # 21
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(),  # relu3-3
    # 24
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    # 28
    nn.Conv2d(256, 512, kernel_size=3, padding=1),
    nn.ReLU(),  # relu4-1, this is the last layer used
    # 31
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(),  # relu4-2
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(),  # relu4-3
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(),  # relu5-1
)

'''
To break VGG into each layer:

self.encoder_module = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(*enc_layers[:4]),
                nn.Sequential(*enc_layers[4:8]),
            ]),
            nn.ModuleList([
                nn.Sequential(*enc_layers[8:11]),
                nn.Sequential(*enc_layers[11:15]),
            ]),
            nn.ModuleList([
                nn.Sequential(*enc_layers[15:18]),
                nn.Sequential(*enc_layers[18:21]),
                nn.Sequential(*enc_layers[21:24]),
                nn.Sequential(*enc_layers[24:27]),
            ]),
            nn.ModuleList([
                nn.Sequential(*enc_layers[27:31]),
            ])
        ])
'''

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x


def make_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg19(pretrained: bool = False, progress: bool = True, **kwargs):

    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)
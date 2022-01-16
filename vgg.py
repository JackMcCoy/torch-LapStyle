import torch.nn as nn

vgg = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=1),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, kernel_size=3),
    nn.ReLU(),  # relu1-1
    # 4
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, kernel_size=3),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    # 8
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, kernel_size=3),
    nn.ReLU(),  # relu2-1
    # 11
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, kernel_size=3),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    # 15
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, kernel_size=3),
    nn.ReLU(),  # relu3-1
    # 18
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(),  # relu3-2
    # 21
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(),  # relu3-3
    # 24
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    # 28
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, kernel_size=3),
    nn.ReLU(),  # relu4-1, this is the last layer used
    # 31
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
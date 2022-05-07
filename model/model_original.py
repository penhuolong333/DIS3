import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
from utils import load_param_from_t7 as load_param


# Original VGG19
# Encoder1/Decoder1
    
# Encoder5/Decoder5
# Encoder = nn.Sequential(
#     nn.Conv2d(3, 3, (1, 1)),                          #Conv0
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(3, 64, (3, 3)),                         #Conv 11
#     nn.ReLU(),  # relu1-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),                        #Conv 12
#     nn.ReLU(),  # relu1-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 128, (3, 3)),                       #Conv 21
#     nn.ReLU(),  # relu2-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),                       #Conv 22
#     nn.ReLU(),  # relu2-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 256, (3, 3)),                        #Conv31
#     nn.ReLU(),  # relu3-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),                        #Conv 32
#     nn.ReLU(),  # relu3-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),                        #Conv 33
#     nn.ReLU(),  # relu3-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),                         #Conv 34
#     nn.ReLU(),  # relu3-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 512, (3, 3)),                           #Conv41
#     nn.ReLU(),  # relu4-1, this is the last layer used
#     # nn.ReflectionPad2d((1, 1, 1, 1)),
#     # nn.Conv2d(512, 512, (3, 3)),                          #Conv42
#     # nn.ReLU(),  # relu4-2
#     # nn.ReflectionPad2d((1, 1, 1, 1)),
#     # nn.Conv2d(512, 512, (3, 3)),                           #Conv 43
#     # nn.ReLU(),  # relu4-3
#     # nn.ReflectionPad2d((1, 1, 1, 1)),
#     # nn.Conv2d(512, 512, (3, 3)),                             #Conv 44
#     # nn.ReLU(),  # relu4-4
#     # nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     # nn.ReflectionPad2d((1, 1, 1, 1)),
#     # nn.Conv2d(512, 512, (3, 3)),                                 #Conv 51
#     # nn.ReLU(),  # relu5-1
#     # nn.ReflectionPad2d((1, 1, 1, 1)),
#     # nn.Conv2d(512, 512, (3, 3)),
#     # nn.ReLU(),  # relu5-2
#     # nn.ReflectionPad2d((1, 1, 1, 1)),
#     # nn.Conv2d(512, 512, (3, 3)),
#     # nn.ReLU(),  # relu5-3
#     # nn.ReflectionPad2d((1, 1, 1, 1)),
#     # nn.Conv2d(512, 512, (3, 3)),
#     # nn.ReLU()  # relu5-4
# )
#
#
#
# decoder = nn.Sequential(
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 256, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 128, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 64, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 3, (3, 3)),
# )




class Encoder(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(Encoder, self).__init__()
        self.vgg =  nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),                          #Conv0
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),                         #Conv 11
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),                        #Conv 12
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),                       #Conv 21
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),                       #Conv 22
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),                        #Conv31
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),                        #Conv 32
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),                        #Conv 33
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),                         #Conv 34
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),                           #Conv41
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),                          #Conv42
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),                           #Conv 43
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),                             #Conv 44
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),                                 #Conv 51
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
            )
        if model:
            self.vgg.load_state_dict(torch.load(model))

        self.enc_layers = list(self.vgg.children())
        print(self.enc_layers)
        self.enc_0 = nn.Sequential(*self.enc_layers[:31])
        print(self.enc_0)
        self.enc_1 = nn.Sequential(*self.enc_layers[:4])  # input -> relu1_1
        print(self.enc_1)
        self.enc_2 = nn.Sequential(*self.enc_layers[4:11])  # relu1_1 -> relu2_1
        print(self.enc_2)
        self.enc_3 = nn.Sequential(*self.enc_layers[11:18])  # relu2_1 -> relu3_1
        print(self.enc_3)
        self.enc_4 = nn.Sequential(*self.enc_layers[18:31])  # relu3_1 -> relu4_1
        print(self.enc_4)
        self.enc_5 = nn.Sequential(*self.enc_layers[31:44])
        print(self.enc_5)

        if fixed:
            for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
                for param in getattr(self, name).parameters():
                    param.requires_grad = False

    def forward(self,input):
        out = self.enc_0(input)
        return out

    def forward_branch(self, input):
        out11 = self.enc_1(input)
        out21 =  self.enc_2(out11)
        out31 = self.enc_3(out21)
        out41 = self.enc_4(out31)
        out51 = self.enc_5(out41)
        return out11,out21,out31,out41,out51


class Decoder(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512, 256, (3, 3)),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 128, (3, 3)),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128, 128, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128, 64, (3, 3)),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64, 64, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64, 3, (3, 3)),
        )

        self.enc_layers = list(self.decoder.children())
        print(self.enc_layers)
        self.enc_0 = nn.Sequential(*self.enc_layers)
        print(self.enc_0)
        self.enc_1 = nn.Sequential(*self.enc_layers[:4])
        print(self.enc_1)
        self.enc_2 = nn.Sequential(*self.enc_layers[4:17])
        print(self.enc_2)
        self.enc_3 = nn.Sequential(*self.enc_layers[17:24])  # relu2_1 -> relu3_1
        print(self.enc_3)
        self.enc_4 = nn.Sequential(*self.enc_layers[24:])  # relu3_1 -> relu4_1
        print(self.enc_4)

        if model:
            self.decoder.load_state_dict(torch.load(model))

        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        out = self.enc_0(input)
        return out

    def forward_branch(self, input):
        out41 = self.enc_1(input)
        out31 = self.enc_2(out41)
        out21 = self.enc_3(out31)
        out11 = self.enc_4(out21)
        return out41, out31, out21, out11
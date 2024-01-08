import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as transforms




class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        #for conv in self.blocks:
          #  fea = fea + conv(fea)
        fea = self.out_conv(fea)



        #光照强度
        illu = fea +input
        illu_k = illu
        illu = torch.clamp(illu, 0.0001, 1)

        return illu, illu_k





class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta



class DNetwork(nn.Module):
    def __init__(self, stage=3):
          super(DNetwork, self).__init__()
          self.stage = stage
          self.enhance = EnhanceNetwork(layers=1, channels=1)
          self.calibrate = CalibrateNetwork(layers=3, channels=16)

    def weights_init(self, m):
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1., 0.02)

    def forward(self, input):

            ilist, rlist, inlist, attlist = [], [], [], []
            input_op = input
            for i in range(self.stage):
                inlist.append(input_op)
                i, i_k = self.enhance(input_op)
                r = input / i
                r = torch.clamp(r, 0, 1)
                att = self.calibrate(r)
                input_op = input + att
                ilist.append(i)
                rlist.append(r)
                attlist.append(torch.abs(att))
            image = input_op.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

            return input_op, image
# Copyright 2021 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from kp3d.utils.image import image_grid


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock2(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2, self).__init__()

        self.conv = nn.Sequential( nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)

        return out

class KeypointEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained):
        super(KeypointEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.encoder = models.resnet18(pretrained)
        self.dropout = nn.Dropout2d(0.2)
        self.normalize_input = True
        self.use_dropout = False


    def forward(self, input_image):
        self.features = []
        
        if self.normalize_input:
            x = (input_image - 0.45) / 0.225
        else:
            x = input_image
        
        x  = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        l1 = self.encoder.layer1(self.encoder.maxpool(x)) if not self.use_dropout else self.dropout(self.encoder.layer1(self.encoder.maxpool(x)))
        l2 = self.encoder.layer2(l1) if not self.use_dropout else self.dropout(self.encoder.layer2(l1))
        l3 = self.encoder.layer3(l2) if not self.use_dropout else self.dropout(self.encoder.layer3(l2))
        l4 = self.encoder.layer4(l3) if not self.use_dropout else self.dropout(self.encoder.layer4(l3))

        return [x, l1, l2, l3, l4]

class KeypointDecoder(nn.Module):
    def __init__(self, num_ch_enc):
        super(KeypointDecoder, self).__init__()

        self.detect_scales = [3]
        self.feat_scales = [1]
        self.depth_scales = [0]

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_ch_dec = np.array([32, 64, 128, 256, 256])

        # Layer4
        self.upconv4_0 = ConvBlock2(self.num_ch_enc[4], self.num_ch_dec[4])
        self.upconv4_1 = ConvBlock2(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])

        # Layer3
        self.upconv3_0 = ConvBlock2(self.num_ch_dec[4], self.num_ch_dec[3])
        self.upconv3_1 = ConvBlock2(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])

        # Layer2
        self.upconv2_0 = ConvBlock2(self.num_ch_dec[3], self.num_ch_dec[2])
        self.upconv2_1 = ConvBlock2(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])

        # Layer1
        self.upconv1_0 = ConvBlock2(self.num_ch_dec[2], self.num_ch_dec[1])
        self.upconv1_1 = ConvBlock2(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])

        # Detector and score
        self.scoreconv = nn.Sequential( nn.Conv2d(self.num_ch_dec[3], self.num_ch_dec[3], kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(self.num_ch_dec[3]), nn.LeakyReLU(inplace=True), Conv3x3(self.num_ch_dec[3], 1) )
        self.locconv   = nn.Sequential( nn.Conv2d(self.num_ch_dec[3], self.num_ch_dec[3], kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(self.num_ch_dec[3]), nn.LeakyReLU(inplace=True), Conv3x3(self.num_ch_dec[3], 2) )        
        self.featconv  = nn.Sequential( nn.Conv2d(self.num_ch_dec[1], self.num_ch_dec[1], kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(self.num_ch_dec[1]), nn.LeakyReLU(inplace=True), Conv3x3(self.num_ch_dec[1], self.num_ch_dec[1]) )

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[4]
        # Layer4
        x = self.upconv4_0(x)
        x = [upsample(x)]
        x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.upconv4_1(x)
        # Layer3
        x = self.upconv3_0(x)
        x = [upsample(x)]
        x += [input_features[2]]
        x = torch.cat(x, 1)
        x = self.upconv3_1(x)
        # Detector and score
        self.outputs[("location")]  = self.tanh(self.locconv(x))
        self.outputs[("score")]     = self.sigmoid(self.scoreconv(x))
        # Layer2
        x = self.upconv2_0(x)
        x = [upsample(x)]
        x += [input_features[1]]
        x = torch.cat(x, 1)
        x = self.upconv2_1(x)
        # Layer1
        x = self.upconv1_0(x)
        x = [upsample(x)]
        x += [input_features[0]]
        x = torch.cat(x, 1)
        x = self.upconv1_1(x)
        # Descriptor features
        self.outputs[("feature")]   = self.featconv(x)

        return self.outputs


class KeypointResnet(nn.Module):
    def __init__(self):
        super().__init__()

        num_layers = 18 
        pretrained = True

        self.encoder = KeypointEncoder(num_layers=num_layers, pretrained=pretrained)
        self.decoderK = KeypointDecoder(num_ch_enc=self.encoder.num_ch_enc)

        self.cross_ratio = 2.0
        self.cell = 8

        self.encoder.normalize_input = False
        self.encoder.use_dropout = True
        
    def forward(self, x):

        B, _, H, W = x.shape

        x = self.encoder(x)
        xK = self.decoderK(x)

        score = xK[('score')]
        center_shift = xK[('location')]
        feat = xK[('feature')]

        _, _, Hc, Wc = score.shape
    
        ############ Remove border for score ##############
        
        border_mask = torch.ones(B,Hc,Wc)
        border_mask[:,0] = 0
        border_mask[:,Hc-1] = 0
        border_mask[:,:,0] = 0 
        border_mask[:,:,Wc-1] = 0
        border_mask = border_mask.unsqueeze(1)
        score = score * border_mask.to(score.device)

        ############ Remap coordinate ##############

        step = (self.cell-1) / 2.
        center_base = image_grid(B, Hc, Wc,
                                dtype=center_shift.dtype,
                                device=center_shift.device,
                                ones=False, normalized=False).mul(self.cell) + step

        coord_un = center_base.add(center_shift.mul(self.cross_ratio * step))
        coord = coord_un.clone()
        coord[:,0] = torch.clamp(coord_un[:,0], min=0, max=W-1)
        coord[:,1] = torch.clamp(coord_un[:,1], min=0, max=H-1)

        ############ Sampling feature ##############
        if self.training is False:
            coord_norm = coord[:,:2].clone()
            coord_norm[:,0] = (coord_norm[:,0] / (float(W-1)/2.)) - 1.
            coord_norm[:,1] = (coord_norm[:,1] / (float(H-1)/2.)) - 1.
            coord_norm = coord_norm.permute(0, 2, 3, 1)

            feat = torch.nn.functional.grid_sample(feat, coord_norm)

            dn = torch.norm(feat, p=2, dim=1) # Compute the norm.
            feat = feat.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        return score, coord, feat

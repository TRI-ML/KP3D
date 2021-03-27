# Copyright 2021 Toyota Research Institute.  All rights reserved.

from functools import partial

import torch
import torch.nn as nn
from kp3d.externals.monodepth2.models.depth_decoder import DepthDecoder
from kp3d.externals.monodepth2.models.layers import disp_to_depth
from kp3d.externals.monodepth2.models.resnet_encoder import ResnetEncoder


class DispResnet(nn.Module):
    """
    Inverse depth network based on the monodepth2 repository
    Receives a version string, in the form of XY:
        X (int): Number of residual layers [18, 34, 50]
        Y (str): If Y == pt, use a pretrained model
    """
    def __init__(self, version=None, **kwargs):
        super().__init__()
        assert version is not None, "DispResnet needs a version"

        num_layers = int(version[:2])       # First two characters are the number of layers
        pretrained = version[2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]


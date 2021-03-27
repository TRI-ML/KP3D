# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.dropout = nn.Dropout2d(0.2)
        
        self.normalize_input = True

        self.use_dropout = False

    def forward(self, input_image):
        self.features = []
        
        if self.normalize_input:
            x = (input_image - 0.45) / 0.225
        else:
            x = input_image

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))

        if self.use_dropout is False:
            self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
            self.features.append(self.encoder.layer2(self.features[-1]))
            self.features.append(self.encoder.layer3(self.features[-1]))
            self.features.append(self.encoder.layer4(self.features[-1]))
        else:
            self.features.append(self.dropout(self.encoder.layer1(self.encoder.maxpool(self.features[-1]))))
            self.features.append(self.dropout(self.encoder.layer2(self.features[-1])))
            self.features.append(self.dropout(self.encoder.layer3(self.features[-1])))
            self.features.append(self.dropout(self.encoder.layer4(self.features[-1])))

        return self.features

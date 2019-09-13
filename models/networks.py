"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

from .i3dpt import I3D
from external.maskrcnn_benchmark.roi_layers import ROIAlign, ROIPool


class ROINet(nn.Module):
    """
    Module for ROI operations: ROI pool | align
    """

    def __init__(self, pool_mode, pool_size=7):
        super(ROINet, self).__init__()

        self.pool_mode = pool_mode
        self.pool_size = pool_size

        if self.pool_mode == 'pool':
            self.pool_layer = ROIPool((self.pool_size, self.pool_size), 1./16.)
        elif self.pool_mode == 'align':
            self.pool_layer = ROIAlign((self.pool_size, self.pool_size), 1./16., 0)
        else:
            raise NotImplementedError

    def forward(self, conv_feat, tubes):
        """
        Args:
            conv_feat: input conv_feat sequences. Shape: [batch_size, T, C, W, H]
            tubes: flatten proposal tubes. Shape: [num_tubes, T, 5]   (dim 0 is batch idx)
        """

        _,_, C, W, H = conv_feat.size()

        pooled_feat = self.pool_layer(conv_feat.view(-1, C,W,H),
                                      tubes.view(-1, 5).detach())
        
        return pooled_feat


class BaseNet(nn.Module):
    """
    Backbone network of the model
    """

    def __init__(self, cfg):
        super(BaseNet, self).__init__()
        
        self.base_name = cfg.base_net
        self.kinetics_pretrain = cfg.kinetics_pretrain
        self.freeze_stats = cfg.freeze_stats
        self.freeze_affine = cfg.freeze_affine
        self.fp16 = cfg.fp16

        if self.base_name == "i3d":
            self.base_model = build_base_i3d(self.kinetics_pretrain, self.freeze_affine)
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Applies network layers on input images

        Args:
            x: input image sequences. Shape: [batch_size, T, C, W, H]
        """

        x = x.permute(0, 2, 1, 3, 4)    # [N,T,C,W,H] --> [N,C,T,W,H]
        conv_feat = self.base_model(x)

        # reshape to original size
        conv_feat = conv_feat.permute(0, 2,1,3,4)    # [N,C,T,W,H] --> [N,T,C,W,H]

        return conv_feat

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        
        # not update the running statistics if specified
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                if self.fp16:
                    m.eval().half()
                else:
                    m.eval()

        if mode and self.freeze_stats:
            self.base_model.apply(set_bn_eval)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def build_base_i3d(kinetics_pretrain=None, freeze_affine=True):

    print ("Building I3D model...")
    i3d = I3D(num_classes=400)

    if kinetics_pretrain is not None:
        if os.path.isfile(kinetics_pretrain):
            print ("Loading I3D pretrained on Kinetics dataset from {}...".format(kinetics_pretrain))
            i3d.load_state_dict(torch.load(kinetics_pretrain))
        else:
            raise ValueError ("Kinetics_pretrain doesn't exist: {}".format(kinetics_pretrain))
            

    base_model = nn.Sequential(i3d.conv3d_1a_7x7,
                               i3d.maxPool3d_2a_3x3,
                               i3d.conv3d_2b_1x1,
                               i3d.conv3d_2c_3x3,
                               i3d.maxPool3d_3a_3x3,
                               i3d.mixed_3b,
                               i3d.mixed_3c,
                               i3d.maxPool3d_4a_3x3,
                               i3d.mixed_4b,
                               i3d.mixed_4c,
                               i3d.mixed_4d,
                               i3d.mixed_4e,
                               i3d.mixed_4f)

    def set_bn_fix(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False

    if freeze_affine:
        base_model.apply(set_bn_fix)

    return base_model


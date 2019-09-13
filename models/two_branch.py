"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .networks import weights_init
from .i3dpt import I3D_head
import sys
sys.path.append('../')
from utils.tube_utils import encode_coef

__all__ = ['ContextNet', 'TwoBranchNet']


def build_conv(base_name='i3d', kinetics_pretrain=None, mode='global', freeze_affine=True):

    if base_name == "i3d":
        print ("Building I3D head for {} branch...".format(mode))
        i3d = I3D_head()

        model_dict = i3d.state_dict()
        if kinetics_pretrain is not None:
            if os.path.isfile(kinetics_pretrain):
                print ("Loading I3D head pretrained on Kinetics dataset...")
                pretrained_dict = torch.load(kinetics_pretrain)
                pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                i3d.load_state_dict(model_dict)
            else:
                raise ValueError ("Kinetics_pretrain doesn't exist: {}".format(kinetics_pretrain))

        if mode == 'context':
            # for context net
            model = nn.Sequential(i3d.maxPool3d,
                                  i3d.mixed_5b,
                                  i3d.mixed_5c)
        else:
            # for global branch
            model = nn.Sequential(i3d.mixed_5b,
                                  i3d.mixed_5c)

    else:
        raise NotImplementedError

    def set_bn_fix(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False

    if freeze_affine:
        model.apply(set_bn_fix)
    return model


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        
        out += residual
        out = self.relu(out)

        return out

class Bottleneck_resample(nn.Module):

    def __init__(self, inplanes, outplanes, planes, stride=1):
        super(Bottleneck_resample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv4 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = self.conv1(x)

        out = self.conv2(x)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        
        out += residual
        out = self.relu(out)

        return out

class ContextNet(nn.Module):
    """
    Context branch
    """

    def __init__(self, cfg):
        super(ContextNet, self).__init__()

        self.T = cfg.T
        self.freeze_stats = cfg.freeze_stats
        self.freeze_affine = cfg.freeze_affine
        self.fp16 = cfg.fp16

        self.i3d_conv_context = build_conv(cfg.base_net, cfg.kinetics_pretrain, 'context', self.freeze_affine)
        self.avg_pool = nn.AvgPool3d((1,13,13), (1,1,1))

        self._init_net()


    def forward(self, conv_feat):

        conv_feat = conv_feat.permute(0,2,1,3,4)
        context_feat = self.i3d_conv_context(conv_feat)
        context_feat = self.avg_pool(context_feat)

        return context_feat
    
    def _init_net(self):
        pass

    def set_device(self, device):
        self.device = device

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)

        if mode:
            # not update the running statistics
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    if self.fp16:
                        m.eval().half()
                    else:
                        m.eval()

            if self.freeze_stats:
                self.i3d_conv_context.apply(set_bn_eval)


class TwoBranchNet(nn.Module):

    def __init__(self, cfg, cls_only=False):
        super(TwoBranchNet, self).__init__()

        self.num_classes = cfg.num_classes
        self.T = cfg.T
        self.base_net = cfg.base_net
        self.freeze_stats = cfg.freeze_stats
        self.freeze_affine = cfg.freeze_affine
        self.fc_dim = cfg.fc_dim
        self.dropout_prob = cfg.dropout
        self.pool_size = cfg.pool_size
        self.no_context = cfg.no_context
        self.fp16 = cfg.fp16
        self.cls_only = cls_only

        self.i3d_conv = build_conv(cfg.base_net, cfg.kinetics_pretrain, 'global', self.freeze_affine)
        self.downsample = nn.Conv3d(1024, self.fc_dim,
                    kernel_size=1, stride=1, bias=True)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.global_cls = nn.Conv3d(
                self.fc_dim * self.pool_size**2 + (1024 if not self.no_context else 0),
                self.num_classes,
                (1,1,1),
                bias=True)

        if not self.cls_only:
            self.local_conv = nn.Sequential(
                    Bottleneck_resample(832+self.fc_dim, 1024, 256),
                    Bottleneck(1024, 256),
                    Bottleneck(1024, 256))
            self.downsample2 = nn.Conv2d(1024, self.fc_dim,
                        kernel_size=1, stride=1, bias=True)
            self.local_reg = nn.Linear(self.fc_dim * self.pool_size**2, 4)
            self.neighbor_reg1 = nn.Linear(self.fc_dim * self.pool_size**2, 4)    # for tube t-1
            self.neighbor_reg2 = nn.Linear(self.fc_dim * self.pool_size**2, 4)    # for tube t+1

        self._init_net()


    def forward(self, global_feat, context_feat=None, tubes=None, targets=None):
        """
        Args:
            global_feat: ROI pooled feat for global branch. Shape: [num_tubes, T, C, W, H]
            tubes: flatten proposal tubes. Shape: [num_tubes, T, 5]   (dim 0 is batch idx)
                   should have absolute value of the position
            targets: flatten target tubes. 
                    for training: Shape: [num_tubes, 3, 67]   
                            (:4 is regression labels
                             k is indicator for classification,
                             5 is indicator for regression, 
                             6: is label for action classfication)
                   should have absolute value of the position
                   [:, 0] is for tube t-1
                   [:, 1] is for center tube
                   [:, -1] is for tube t+1
        """

        global_feat = global_feat.to(self.device)
        if context_feat is not None:
            context_feat = context_feat.to(self.device)


        N, T, C, W, H = global_feat.size()
        chunks = int(T / self.T)
        chunk_idx = [j*self.T + int(self.T/2) for j in range(chunks)]    # used to index the middel frame of each chunk
        half_T = int(self.T/2)

        #### global branch ####

        global_feat_conv = self.i3d_conv(global_feat.permute(0,2,1,3,4))
        global_feat_conv = self.downsample(global_feat_conv)

        # flatten 7x7 feature map
        global_feat_conv_flat = global_feat_conv.permute(0,2,1,3,4).contiguous().view(N,T,-1,1,1)
        global_feat_conv_flat = global_feat_conv_flat.permute(0,2,1,3,4).contiguous()
        # concatenate context feature along channel dimension
        if context_feat is not None:
            global_feat_conv_flat = torch.cat([global_feat_conv_flat, context_feat], dim=1)
        global_feat_conv_flat = self.dropout(global_feat_conv_flat)

        global_class = self.global_cls(global_feat_conv_flat)
        global_class = global_class.squeeze(3)
        global_class = global_class.squeeze(3)
        global_class = global_class.mean(2)

        #### local branch ####

        local_loc, first_loc, last_loc = torch.tensor([0.]).to(global_class), torch.tensor([0.]).to(global_class), torch.tensor([0.]).to(global_class)
        if not self.cls_only:
            local_feat = global_feat.permute(0,2,1,3,4)
            local_feat = torch.cat([local_feat, global_feat_conv], dim=1)
            # (N,C,T,W,H) --> (N,T,C,W,H) --> (N*T,C,W,H)
            local_feat = local_feat.permute(0,2,1,3,4).contiguous().view(N*T,-1,W,H)
            local_feat = self.local_conv(local_feat)
            local_feat = self.downsample2(local_feat)
            local_feat = self.dropout(local_feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_loc = self.local_reg(local_feat).view(N, T, -1)
    
            first_loc = local_loc[:, chunk_idx[0]-half_T:chunk_idx[0]+half_T+1].contiguous().clone()
            last_loc = local_loc[:, chunk_idx[-1]-half_T:chunk_idx[-1]+half_T+1].contiguous().clone()
    
            # neighbor prediction
            first_loc += self.neighbor_reg1(local_feat.view(N,T,-1)[:, chunk_idx[0]-half_T:chunk_idx[0]+half_T+1].contiguous().view(-1, self.fc_dim*self.pool_size**2)).view(N,self.T,-1)
            last_loc += self.neighbor_reg2(local_feat.view(N,T,-1)[:, chunk_idx[-1]-half_T:chunk_idx[-1]+half_T+1].contiguous().view(-1, self.fc_dim*self.pool_size**2)).view(N,self.T,-1)
    
            center_pred = local_loc[:, chunk_idx[int(chunks/2)]].contiguous().view(N, -1)
            first_pred = first_loc[:, half_T].contiguous().view(N, -1)
            last_pred = last_loc[:, half_T].contiguous().view(N, -1)

        #### compute losses ####

        loss_global_cls = torch.tensor(0.).to(global_class)
        loss_local_loc = torch.tensor(0.).to(local_loc)
        loss_neighbor_loc = torch.tensor(0.).to(local_loc)
        if targets is not None:
            tubes = tubes.to(self.device)
            targets = targets.to(self.device)

            center_targets = targets[:, 1].contiguous()
            first_targets = targets[:, 0].contiguous()
            last_targets = targets[:, -1].contiguous()
            center_tubes = tubes[:, chunk_idx[int(chunks/2)]].contiguous()
            first_tubes = tubes[:, chunk_idx[0]].contiguous()
            last_tubes = tubes[:, chunk_idx[-1]].contiguous()

            ######### classification loss for center clip #########

            with torch.no_grad():
                mask = center_targets[:, 4].view(-1, 1)
            if mask.sum():
                target = center_targets[:, 6:] * mask    # mask out background samples
                loss_global_cls = F.binary_cross_entropy_with_logits(global_class, 
                        target, reduction='none')

            if not self.cls_only:
                ######### regression loss for center clip #########
    
                # transform target to regression parameterization
                center_targets_loc = center_targets[:, :4].clone()
                center_targets_loc = encode_coef(center_targets_loc, center_tubes.view(-1,5)[:, 1:])
    
                with torch.no_grad():
                    mask = center_targets[:, 5].view(-1, 1).repeat(1,4)
                if mask.sum():
                    loss_local_loc = F.smooth_l1_loss(center_pred, center_targets_loc, reduction='none')
                    loss_local_loc = torch.sum(loss_local_loc * mask.detach()) / torch.sum(mask.detach())    # masked average
    
                ######### regression loss for neighbor clips #########
    
                # transform target to regression parameterization
                first_targets_loc = first_targets[:, :4].clone()
                last_targets_loc = last_targets[:, :4].clone()
                neighbor_targets_loc = torch.cat([first_targets_loc, last_targets_loc], dim=0)
                neighbor_targets_loc = encode_coef(neighbor_targets_loc, 
                                        torch.cat([first_tubes.view(-1,5)[:, 1:],
                                                   last_tubes.view(-1,5)[:, 1:]], dim=0))
    
                with torch.no_grad():
                    first_mask = first_targets[:, 5].view(-1, 1).repeat(1,4)
                    last_mask = last_targets[:, 5].view(-1, 1).repeat(1,4)
                    neighbor_mask = torch.cat([first_mask, last_mask], dim=0)
    
                if neighbor_mask.sum():
                    neighbor_loc = torch.cat([first_pred, last_pred], dim=0)
    
                    loss_neighbor_loc = F.smooth_l1_loss(neighbor_loc, neighbor_targets_loc, reduction='none')
                    loss_neighbor_loc = torch.sum(loss_neighbor_loc * neighbor_mask.detach()) / torch.sum(neighbor_mask.detach())

        #### Output ####

        global_prob = torch.sigmoid(global_class)
        loss_global_cls = loss_global_cls.view(-1)
        loss_local_loc = loss_local_loc.view(-1)
        loss_neighbor_loc = loss_neighbor_loc.view(-1)

        return global_prob, local_loc, first_loc, last_loc, loss_global_cls, loss_local_loc, loss_neighbor_loc
    
    def _init_net(self):
    
        self.global_cls.apply(weights_init)
        self.downsample.apply(weights_init)
        if not self.cls_only:
            self.local_conv.apply(weights_init)
            self.local_reg.apply(weights_init)
            self.downsample2.apply(weights_init)
            self.neighbor_reg1.apply(weights_init)
            self.neighbor_reg2.apply(weights_init)

    def set_device(self, device):
        self.device = device

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)

        if mode:
            # not update the running statistics
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    if self.fp16:
                        m.eval().half()
                    else:
                        m.eval()

            if self.freeze_stats:
                self.i3d_conv.apply(set_bn_eval)


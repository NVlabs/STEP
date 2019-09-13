"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import math
from bisect import bisect_right

__all__ = ['get_params', 'WarmupCosineLR', 'WarmupStepLR']

def get_params(nets, args):

    params = []
    parameter_dict = dict(nets['base_net'].named_parameters())
    #Set different learning rate to bias layers and set their weight_decay to 0
    for name in sorted(parameter_dict.keys()):
        param = parameter_dict[name]
        if param.requires_grad:
            # i3d.conv3d_1a_7x7
            if name.find('base_model.0')>-1:
                layer_lr = args.base_lr/8 if args.input_type == "rgb" else args.base_lr/4
                if name.find('bias') > -1:
                    params += [{'params': [param], 'lr': layer_lr*2, 'weight_decay': 0}]
                else:
                    params += [{'params':[param], 'lr': layer_lr, 'weight_decay':args.weight_decay}]

            # i3d.conv3d_2b_1x1, i3d.conv3d_2c_3x3
            elif name.find('base_model.2')>-1 or name.find('base_model.3')>-1:
                layer_lr = args.base_lr/4
                if name.find('bias') > -1:
                    params += [{'params': [param], 'lr': layer_lr*2, 'weight_decay': 0}]
                else:
                    params += [{'params':[param], 'lr': layer_lr, 'weight_decay':args.weight_decay}]

            # i3d.mixed_3b, i3d.mixed_3c
            elif name.find('base_model.5')>-1 or name.find('base_model.6')>-1:
                layer_lr = args.base_lr/2
                if name.find('bias') > -1:
                    params += [{'params': [param], 'lr': layer_lr*2, 'weight_decay': 0}]
                else:
                    params += [{'params':[param], 'lr': layer_lr, 'weight_decay':args.weight_decay}]

            # i3d.mixed_4b, i3d.mixed_4c, i3d.mixed_4d, i3d.mixed_4e, i3d.mixed_4f
            else:
                layer_lr = args.base_lr
                if name.find('bias') > -1:
                    params += [{'params': [param], 'lr': layer_lr*2, 'weight_decay': 0}]
                else:
                    params += [{'params':[param], 'lr': layer_lr, 'weight_decay':args.weight_decay}]

    if 'context_net' in nets:
        parameter_dict = dict(nets['context_net'].named_parameters())
        #Set different learning rate to bias layers and set their weight_decay to 0
        for name in sorted(parameter_dict.keys()):
            param = parameter_dict[name]
            if param.requires_grad:
                if True:
                    layer_lr = args.det_lr0
                    if name.find('bias') > -1:
                        params += [{'params': [param], 'lr': layer_lr*2, 'weight_decay': 0}]
                    else:
                        params += [{'params': [param], 'lr': layer_lr, 'weight_decay': args.weight_decay}]


    for i in range(args.max_iter):
        parameter_dict = dict(nets['det_net%d' % i].named_parameters())
        #Set different learning rate to bias layers and set their weight_decay to 0
        for name in sorted(parameter_dict.keys()):
            param = parameter_dict[name]
            if param.requires_grad:
                if name.find('i3d') > -1:
                    layer_lr = args.det_lr0
                    if name.find('bias') > -1:
                        params += [{'params': [param], 'lr': layer_lr*2, 'weight_decay': 0}]
                    else:
                        params += [{'params': [param], 'lr': layer_lr, 'weight_decay': args.weight_decay}]

    # separate this group to ensure param_group[-1] is det_lr
    for i in range(args.max_iter):
        parameter_dict = dict(nets['det_net%d' % i].named_parameters())
        #Set different learning rate to bias layers and set their weight_decay to 0
        for name in sorted(parameter_dict.keys()):
            param = parameter_dict[name]
            if param.requires_grad:
                if not name.find('i3d') > -1:
                    layer_lr = args.det_lr
                    if name.find('bias') > -1:
                        params += [{'params': [param], 'lr': layer_lr*2, 'weight_decay': 0}]
                    else:
                        params += [{'params': [param], 'lr': layer_lr, 'weight_decay': args.weight_decay}]
    
    return params


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing learning rate scheduler with periodic restarts.
    Reference: https://www.jeremyjordan.me/nn-learning-rate/
    Author: Xitong Yang

    Arguments:
        milestones: a list of steps for each cycle (e.g., [1000, 2000, 3000])
        min_ratio (float): Minimum ratio decay for base_lr, i.e. eta_min = min_ratio * base_lr
        cycle_decay (float): Reduce the base_lr after the completion of each cycle, default: 1.0 (no decay)
        last_epoch: The index of last step (Note that we schedule lr per mini-batch step, not "epoch" as default lr shedulers)
        warmup_iters: Iterations for linear warmup
        warmup_factor
    """

    def __init__(self, optimizer, milestones, min_ratio=0., cycle_decay=1., warmup_iters=1000, warmup_factor=1./10, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}".format(milestones)
            )
        self.milestones = [warmup_iters]+milestones
        self.min_ratio = min_ratio
        self.cycle_decay = cycle_decay
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha

            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        else:
            # which cyle is it
            cycle = min(bisect_right(self.milestones, self.last_epoch), len(self.milestones)-1)
            # calculate the fraction in the cycle
            fraction = min((self.last_epoch - self.milestones[cycle-1]) / (self.milestones[cycle]-self.milestones[cycle-1]), 1.)

            return [base_lr*self.min_ratio + (base_lr * self.cycle_decay**(cycle-1) - base_lr*self.min_ratio) *
                    (1 + math.cos(math.pi * fraction)) / 2
                    for base_lr in self.base_lrs]

class WarmupStepLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Reference: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/solver/lr_scheduler.py
    Author: Xitong Yang

    Arguments:
        milestones: a list of steps to decay (e.g., [1000, 2000, 3000])
        gamma: decay rate
        last_epoch: The index of last step (Note that we schedule lr per mini-batch step, not "epoch" as default lr shedulers)
        warmup_iters: Iterations for linear warmup
        warmup_factor
    """

    def __init__(self, optimizer, milestones, gamma=0.1, warmup_iters=1000, warmup_factor=1./10, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}".format(milestones)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        super(WarmupStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha

            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]


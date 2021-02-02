# -*- coding:utf-8 -*-
###
# File: mixup_wrapper.py
# Created Date: Wednesday, May 27th 2020, 4:59:08 pm
# Author: yusnows
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 yusnows
#
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import numpy as np
import torch
import torch.nn as nn

#from ..nn import SyncBatchNorm
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ['MixUpWrapper']


class MixUpWrapper(object):
    def __init__(self, alpha, num_classes, dataloader, device):
        self.alpha = alpha
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = device

    def mixup_loader(self, loader):
        def mixup(alpha, num_classes, data, target):
            with torch.no_grad():
                bs = data.size(0)
                c = np.random.beta(alpha, alpha)
                perm = torch.randperm(bs).cuda()

                md = c * data + (1-c) * data[perm, :]
                mt = c * target + (1-c) * target[perm, :]
                return md, mt

        for input, target in loader:
            input, target = input.cuda(self.device), target.cuda(self.device)
            target = torch.nn.functional.one_hot(target, self.num_classes)
            i, t = mixup(self.alpha, self.num_classes, input, target)
            yield i, t

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self.mixup_loader(self.dataloader)

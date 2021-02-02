# -*- coding:utf-8 -*-
###
# File: network.py
# Created Date: Thursday, June 4th 2020, 4:12:46 pm
# Author: yusnows
# -----
# Last Modified: Sat Jun 13 2020
# Modified By: yusnows
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_backbone

__all__ = ['ClsNetwork']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class ClsNetwork(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(ClsNetwork, self).__init__()
        self.num_classes = cfg.NETWORK.NUM_CLASSES
        self.fc_dropout = cfg.NETWORK.DROPOUT
        self.ifbnneck = cfg.NETWORK.BNNECK
        self.backbone = get_backbone(cfg.NETWORK.BACKBONE_ARCH, cfg.NETWORK.PRETRAINED, **kwargs)
        if self.ifbnneck:
            self.bnneck = nn.BatchNorm1d(self.backbone.feat_dim)
            self.bnneck.bias.requires_grad_(False)
            self.fc = nn.Linear(self.backbone.feat_dim, self.num_classes, bias=False)
            self.bnneck.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)
        else:
            self.fc = nn.Linear(self.backbone.feat_dim, self.num_classes)
            self.fc.apply(weights_init_classifier)

    def forward(self, x):
        feat = self.backbone(x)
        if self.ifbnneck:
            feat = self.bnneck(feat)
        if self.fc_dropout > 0 and self.fc_dropout < 1:
            feat = F.dropout(feat, p=self.fc_dropout, training=self.training)
        la = self.fc(feat)
        if self.training:
            return la, feat
        return la

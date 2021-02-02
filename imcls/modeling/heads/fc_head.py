# -*- coding:utf-8 -*-
###
# File: fc_head.py
# Created Date: Wednesday, October 7th 2020, 3:03:40 pm
# Author: yusnows
# -----
# Last Modified: Wed Oct 07 2020
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
import torch.nn as nn
import torch.nn.functional as F
from ...nn_module import CrossEntropyLabelSmooth

__all__ = ["FcHead"]


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


class FcHead(nn.Module):
    def __init__(self, cfg, in_feats) -> None:
        super(FcHead, self).__init__()
        self.in_feats = in_feats
        self.num_cls = cfg.NETWORK.NUM_CLASSES
        self.ifbnneck = cfg.NETWORK.BNNECK
        self.fc_dropout = cfg.NETWORK.FC_DROPOUT
        if self.ifbnneck:
            self.bnneck = nn.BatchNorm1d(self.backbone.feat_dim)
            self.bnneck.bias.requires_grad_(False)
            self.fc = nn.Linear(self.backbone.feat_dim, self.num_classes, bias=False)
            self.bnneck.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)
        else:
            self.fc = nn.Linear(self.backbone.feat_dim, self.num_classes)
            self.fc.apply(weights_init_classifier)

        self.cls_critera = CrossEntropyLabelSmooth(self.num_cls, cfg.NETWORK.LABEL_SMOOTHING)

    def forward(self, feat, gt=None):
        if self.ifbnneck:
            feat = self.bnneck(feat)
        if self.fc_dropout > 0 and self.fc_dropout < 1:
            feat = F.dropout(feat, p=self.fc_dropout, training=self.training)
        la = self.fc(feat)
        if not self.training:
            return la
        assert gt is not None, "when in train state, input gt should not be None"
        loss_dict = {}
        loss_dict.update(self.cls_loss(la, gt))
        return loss_dict

    def cls_loss(self, pred, gt):
        return {"cls_loss": self.cls_critera(pred, gt)}

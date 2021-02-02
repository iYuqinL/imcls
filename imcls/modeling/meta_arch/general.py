# -*- coding:utf-8 -*-
###
# File: general.py
# Created Date: Sunday, October 4th 2020, 5:10:09 pm
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
import torch
import torch.nn as nn
from ..backbone import get_backbone
from ..heads.fc_head import FcHead


class GeneralClsModel(nn.Module):
    def __init__(self, cfg, **kwargs) -> None:
        super(GeneralClsModel, self).__init__()
        self.device = torch.device(cfg.DIVICE)
        self.backbone = get_backbone(
            cfg.NETWORK.BACKBONE_ARCH, cfg.NETWORK.PRETRAINED, **kwargs)
        self.fc_head = FcHead(cfg, self.backbone.feat_dim)

    def forward(self, data):
        if not self.training:
            im = data
            im = im.to(self.device)
            feat = self.backbone(im)
            la = self.fc_head(feat)
            return la
        im, gt = data
        im, gt = im.to(self.device), gt.to(self.device)
        feat = self.backbone(im)
        loss_dict = self.fc_head(feat, gt)
        return loss_dict

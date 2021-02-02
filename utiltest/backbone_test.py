# -*- coding:utf-8 -*-
###
# File: backbone_test.py
# Created Date: Thursday, June 4th 2020, 9:06:23 pm
# Author: yusnows
# -----
# Last Modified: Sun Jun 07 2020
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
import os
import sys
sys.path.append(os.path.abspath('.'))
import torch
import imcls.networkarch.backbone as backbone

if __name__ == "__main__":
    net = backbone.get_backbone("resnet18", pretrained=True)
    print(net)
    im = torch.randn(1, 3, 224, 224)
    feat = net(im)
    print(feat.shape)
    print(net.feat_dim)
    
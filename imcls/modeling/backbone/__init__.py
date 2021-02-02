# -*- coding:utf-8 -*-
###
# File: __init__.py
# Created Date: Thursday, June 4th 2020, 4:13:13 pm
# Author: yusnows
# -----
# Last Modified: Tue Oct 13 2020
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
from . import efficientnet
from . import resnet_cbam
from . import torchvis
from . import resnest

__all__ = ['get_backbone']

backbone_arch = {
    # shufflenet
    "shufflenet_v2_0_5": torchvis.shufflenet_v2_x0_5,
    "shufflenet_v2_1_0": torchvis.shufflenet_v2_x1_0,
    "shufflenet_v2_1_5": torchvis.shufflenet_v2_x1_5,
    "shufflenet_v2_2_0": torchvis.shufflenet_v2_x2_0,
    # mobilenetv2
    "mobilenet_v2": torchvis.mobilenet_v2,
    # densenet
    "densenet121": torchvis.densenet121,
    "densenet161": torchvis.densenet161,
    "densenet169": torchvis.densenet169,
    "densenet201": torchvis.densenet201,
    # squeezenet
    "squeezenet1_0": torchvis.squeezenet1_0,
    "squeezenet1_1": torchvis.squeezenet1_1,
    # mnasnet
    "mnasnet0_5": torchvis.mnasnet0_5,
    "mnasnet0_75": torchvis.mnasnet0_75,
    "mnasnet1_0": torchvis.mnasnet1_0,
    "mnasnet1_3": torchvis.mnasnet1_3,
    # resnet
    "resnet18": torchvis.resnet18,
    "resnet34": torchvis.resnet34,
    "resnet50": torchvis.resnet50,
    "resnet101": torchvis.resnet101,
    "resnet152": torchvis.resnet152,
    "resnext50_32x4d": torchvis.resnext50_32x4d,
    "resnext101_32x8d": torchvis.resnext101_32x8d,
    # resnet wsl
    "resnext101_32x8d_wsl": torchvis.resnext101_32x8d_wsl,
    "resnext101_32x16d_wsl": torchvis.resnext101_32x16d_wsl,
    "resnext101_32x32d_wsl": torchvis.resnext101_32x32d_wsl,
    "resnext101_32x48d_wsl": torchvis.resnext101_32x48d_wsl,
    # resnet cbam
    "resnet18_cbam": resnet_cbam.resnet18,
    "resnet34_cbam": resnet_cbam.resnet34,
    "resnet50_cbam": resnet_cbam.resnet50,
    "resnet101_cbam": resnet_cbam.resnet101,
    "resnet152_cbam": resnet_cbam.resnet152,
    "resnext50_32x4d": resnet_cbam.resnext50_32x4d,
    "resnext101_32x8d": resnet_cbam.resnext101_32x8d,
    # resnet cbam wsl
    "resnext101_32x8d_wsl_cbam": resnet_cbam.resnext101_32x8d_wsl,
    "resnext101_32x16d_wsl_cbam": resnet_cbam.resnext101_32x16d_wsl,
    "resnext101_32x32d_wsl_cbam": resnet_cbam.resnext101_32x32d_wsl,
    "resnext101_32x48d_wsl_cbam": resnet_cbam.resnext101_32x48d_wsl,
    # efficientnet
    "efficientnet-b0": efficientnet.efficientnet_b0,
    "efficientnet-b1": efficientnet.efficientnet_b1,
    "efficientnet-b2": efficientnet.efficientnet_b2,
    "efficientnet-b3": efficientnet.efficientnet_b3,
    "efficientnet-b4": efficientnet.efficientnet_b4,
    "efficientnet-b5": efficientnet.efficientnet_b5,
    "efficientnet-b6": efficientnet.efficientnet_b6,
    "efficientnet-b7": efficientnet.efficientnet_b7,
    # resnest
    "resnest50": resnest.resnest50,
    "resnest101": resnest.resnest101,
    "resnest200": resnest.resnest200,
    "resnest269": resnest.resnest269,
}


def get_backbone(name, pretrained=False, **kwargs) -> nn.Module:
    return backbone_arch[name](pretrained, **kwargs)

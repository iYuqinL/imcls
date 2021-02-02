# -*- coding:utf-8 -*-
###
# File: build.py
# Created Date: Friday, October 9th 2020, 9:54:28 pm
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
from .general import GeneralClsModel

meta_arch_dict = {
    "GeneralClsModel": GeneralClsModel,
}


def build_model(cfg):
    meta_arch = cfg.NETWORK.META_ARCH
    model = meta_arch_dict[meta_arch]
    model.to(torch.device)
    return model

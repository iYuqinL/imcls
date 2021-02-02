# -*- coding:utf-8 -*-
###
# File: backbonebase.py
# Created Date: Wednesday, October 7th 2020, 3:27:21 pm
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


class BackBone(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(BackBone, self).__init__()

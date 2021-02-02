# -*- coding:utf-8 -*-
###
# File: build.py
# Created Date: Saturday, October 10th 2020, 10:43:25 am
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
import torchvision.transforms as tvtrans
from .datasets import CallSmoke


def build_cls_train_loader(cfg):
    train_dir = cfg.DATASETS.TRAIN_ROOT
    batch_size = cfg.DATALOADER.BATCH_SIZE
    workers = cfg.DATALOADER.NUM_WORKERS

    train_trans = tvtrans.Compose(
        [tvtrans.RandomHorizontalFlip(),
         tvtrans.Resize(size=(224, 224)),
         tvtrans.ToTensor(),
         tvtrans.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])]
    )

    train_set = CallSmoke(train_dir, transforms=train_trans)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True, pin_memory=False)
    return train_loader


def build_cls_valid_loader(cfg):
    train_dir = cfg.DATASETS.VALID_ROOT
    batch_size = cfg.DATALOADER.BATCH_SIZE
    workers = cfg.DATALOADER.NUM_WORKERS

    train_trans = tvtrans.Compose(
        [tvtrans.Resize(size=(224, 224)),
         tvtrans.ToTensor(),
         tvtrans.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])

    valid_set = CallSmoke(train_dir, transforms=train_trans)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, num_workers=workers, shuffle=False, drop_last=True, pin_memory=False)
    return valid_loader

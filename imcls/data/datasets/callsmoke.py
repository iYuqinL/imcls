# -*- coding:utf-8 -*-
###
# File: callsmoke.py
# Created Date: Wednesday, September 30th 2020, 2:17:06 pm
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
import os
import glob
import numpy as np
import PIL.Image as Image
import torch.utils.data as torchdata

__all__ = ["CallSmoke", "CallSmokeTest"]

cls_names = ["normal", "calling", "smoking", "smoking_calling"]


class CallSmoke(torchdata.Dataset):
    def __init__(self, dataroot, transforms=None, impostfix=["jpg"]) -> None:
        """
        Parameters
        ----------
        dataroot: dataset root path, there are cls_names subdirs in the path.
        """
        super(CallSmoke, self).__init__()
        self.dataroot = dataroot
        self.trans = transforms
        self.impostfix = impostfix
        cdir = os.getcwd()
        os.chdir(self.dataroot)
        imlabel_pairs = []
        for idx in range(len(cls_names)):
            subdir = cls_names[idx]
            cls_ims = []
            for postfix in self.impostfix:
                cls_ims += glob.glob(os.path.join(subdir, "*."+postfix))
                # print(cls_ims)
            cls_label = [idx] * len(cls_ims)
            imlabel_pairs += [(cls_ims[i], cls_label[i]) for i in range(len(cls_ims))]
        np.random.shuffle(imlabel_pairs)
        self.imlabel_pairs = imlabel_pairs
        os.chdir(cdir)

    def __getitem__(self, index: int):
        im_file = os.path.join(self.dataroot, self.imlabel_pairs[index][0])
        image = Image.open(im_file)
        label = self.imlabel_pairs[index][1]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.trans is not None:
            image = self.trans(image)
        return image, label

    def __len__(self):
        return len(self.imlabel_pairs)


class CallSmokeTest(torchdata.Dataset):
    def __init__(self, datadir, transforms=None, impostfix=["jpg"]) -> None:
        """
        Parameters
        ----------
        dataroot: dataset root path, there are cls_names subdirs in the path.
        """
        super(CallSmokeTest, self).__init__()
        self.datadir = datadir
        self.trans = transforms
        self.impostfix = impostfix
        cdir = os.getcwd()
        os.chdir(self.datadir)
        im_names = glob.glob("*.jpg")
        os.chdir(cdir)
        im_names.sort(key=lambda x: int(x[:-4]))
        self.im_names = im_names

    def __getitem__(self, index: int):
        im_name = self.im_names[index]
        im_file = os.path.join(self.datadir, im_name)
        image = Image.open(im_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.trans is not None:
            image = self.trans(image)
        return image, im_name

    def __len__(self):
        return len(self.im_names)

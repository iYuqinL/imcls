# -*- coding:utf-8 -*-
###
# File: data_process.py
# Created Date: Friday, September 27th 2019, 11:00:00 pm
# Author: yusnows
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2019 yusnows
#
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np


class ResizeFill(object):
    """
    resize the images
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        super(ResizeFill, self).__init__()
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w/h < ratio:
            t = int(h*ratio)
            w_padding = (t-w)//2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w/ratio)
            h_padding = (t-h)//2
            img = img.crop((0, -h_padding, w, h+h_padding))
        img = img.resize(self.size, self.interpolation)
        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, scale=(0.1, 1.0)):
        self.scale = scale
        self.randmax = 1000000000

    def __call__(self, img):
        rand_low = self.scale[0] * self.randmax
        rand_high = self.scale[1] * self.randmax
        scale = float(np.random.randint(rand_low, rand_high+1))/rand_high
        w, h = img.size
        new_h, new_w = int(h*scale), int(w*scale)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img = img.crop((left, top, left + new_w, top + new_h))
        return img


class ToHSVToRGB(object):
    def __init__(self):
        super(ToHSVToRGB, self).__init__()

    def __call__(self, img):
        fraction = 0.50
        img_hsv = img.convert('HSV')
        img_hsv = np.array(img_hsv)
        # 类似于BGR，HSV的shape=(w,h,c)，其中三通道的c[0,1,2]含有h,s,v信息
        S = img_hsv[:, :, 1].astype(np.float32)
        V = img_hsv[:, :, 2].astype(np.float32)
        a = (np.random.random() * 2 - 1) * fraction + 1
        S *= a
        if a > 1:
            np.clip(S, a_min=0, a_max=255, out=S)
        a = (np.random.random() * 2 - 1) * fraction + 1
        V *= a
        if a > 1:
            np.clip(V, a_min=0, a_max=255, out=V)
        img_hsv[:, :, 1] = S.astype(np.uint8)
        img_hsv[:, :, 2] = V.astype(np.uint8)
        img_hsv = Image.fromarray(img_hsv, 'HSV')
        img = img_hsv.convert("RGB")
        return img


class RandomToHSVToRGB(object):
    def __init__(self, probility=0.5):
        super(RandomToHSVToRGB, self).__init__()
        self.probility = probility

    def __call__(self, img):
        prob = np.random.random()
        if prob <= self.probility:
            fraction = 0.50
            img_hsv = img.convert('HSV')
            img_hsv = np.array(img_hsv)
            # 类似于BGR，HSV的shape=(w,h,c)，其中三通道的c[0,1,2]含有h,s,v信息
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)
            a = (np.random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)
            a = (np.random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)
            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            img_hsv = Image.fromarray(img_hsv, 'HSV')
            img = img_hsv.convert("RGB")
        return img


if __name__ == "__main__":
    im_file = "/home/yusnows/Documents/DataSets/competition/weatherRecog/original/test.jpg"
    im_save = "/home/yusnows/Documents/DataSets/competition/weatherRecog/original/test-1.jpg"
    img = Image.open(im_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    trans = RandomCrop(scale=(0.4, 1.0))
    img = trans(img)
    trans = ToHSVToRGB()
    img = trans(img)
    trans = ResizeFill(224)
    img = trans(img)
    img.save(im_save)

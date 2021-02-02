# -*- coding:utf-8 -*-
###
# File: save2onnx.py
# Created Date: Thursday, September 26th 2019, 2:15:41 pm
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
import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import PIL
from efficientnet import EfficientNet
import classifi_model as cmodel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',  default='efficientnet-b4', help='efficient net architecture')
    parser.add_argument('--csv', default='/root/notebook/river_pollute/data_csv/test.csv', help='path to dataset')
    parser.add_argument('--dataroot', default='/home/data/V1.0/test/', help='path to dataset')
    parser.add_argument('--workers', type=int,  default=16, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--model_url', type=str, default='model/model_valid_855.pth',
                        help='the pretrained model path to load')
    parser.add_argument('--imageSize', type=int, default=224,
                        help='the height and width of the input image to network')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--from_pretrained', action='store_true', default=False,
                        help='if use the official pretrained model, default is True')
    opt = parser.parse_args()
    print(opt)
    gpus = [opt.gpu]
    if len(gpus) == 0:
        device = torch.device('cpu')
    else:
        if gpus[0] >= 0:
            device = torch.device('cuda:%d' % gpus[0])
        else:
            device = torch.device('cpu')
    if 'efficientnet' in opt.arch:
        print('efficientnet image size')
        image_size = EfficientNet.get_image_size(opt.arch)
    else:
        print('not efficientnet image size')
        image_size = opt.imageSize
    dummy_input = torch.randn(64, 3, 224, 224, device=device)
    classi_model = cmodel.ClassiModel(
        opt.arch, gpus=[opt.gpu],
        num_classes=opt.num_classes, from_pretrained=opt.from_pretrained)
    if opt.model_url is not None:
        classi_model.loadmodel(opt.model_url, ifload_fc=True)
    torch.onnx.export(classi_model, dummy_input, "eff-b4-s-valid-855.onnx", verbose=True)

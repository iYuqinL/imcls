# -*- coding:utf-8 -*-
###
# File: train.py
# Created Date: Wednesday, September 11th 2019, 11:39:44 am
# Author: yusnows
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2019 nju-visg
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
from efficientnet_pytorch import EfficientNet
from csv_dataset import CsvDataset
import classifi_model as cmodel
import config
import time
import data_process

if __name__ == "__main__":
    conf = config.Config()
    opt = conf.create_opt()
    print(opt)
    start_time = time.time()
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        # data_process.RandomCrop(scale=(0.6, 1.0)),
        data_process.RandomAffine(30, p=0.85, translate=(0.15, 0.15)),
        data_process.ResizeFill(image_size),
        data_process.RandomBlur(p=0.5),
        data_process.RandomNoise(p=0.75),
        data_process.RandomErasing(p=0.85),
        data_process.RandomShear(p=0.9),
        data_process.RandomHSVShift(),
        data_process.RandomContrast(p=0.8),
        data_process.RandomFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = CsvDataset(opt.traincsv, opt.trainroot, transform=train_transforms, extension="")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers, drop_last=True,
        pin_memory=False)
    classi_model = cmodel.ClassiModel(
        arch=opt.arch, gpus=[opt.gpu], optimv=opt.optimizer, num_classes=opt.num_classes,
        lr=opt.lr_list[0], weight_decay=opt.weight_decay, from_pretrained=opt.from_pretrained, ifcbam=opt.ifcbam)
    if opt.model_url is not None:
        classi_model.loadmodel(opt.model_url, ifload_fc=opt.load_fc)
    if opt.eval is False:
        classi_model.train_fold(train_loader, None, 100, opt)
    else:
        num_correct = 0
        num_error = 0
        for i, data in enumerate(train_loader, start=0):
            images, labels = data
            labels = labels.to(device)
            outputs = classi_model.test(images)
            pred_label = torch.argmax(outputs, 1)
            num_correct += torch.sum(pred_label == labels, 0)
        print('num_correct: ', num_correct.item())
        print('accuracy of prediction on imgs: %f' % (num_correct.item()/len(train_dataset)))
    end_time = time.time()
    print("train/eval time use: %f" % (end_time - start_time))

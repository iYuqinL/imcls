# -*- coding:utf-8 -*-
###
# File: test.py
# Created Date: Thursday, September 12th 2019, 9:12:38 am
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
from torch.utils.tensorboard import SummaryWriter
import PIL
from efficientnet_pytorch import EfficientNet
from csv_dataset import CsvDataset
from csv_dataset import CsvDatasetTest
import classifi_model as wrmodel
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',  default='efficientnet-b5', help='efficient net architecture')
    parser.add_argument('--csv', required=True, help='path to dataset')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int,  default=4, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--model_url', type=str, default=None, help='the pretrained model path to load')
    parser.add_argument('--num_classes', type=int, default=9)
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
    image_size = EfficientNet.get_image_size(opt.arch)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = CsvDatasetTest(opt.csv, opt.dataroot, transform=train_transforms, extension="")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers, drop_last=False,
        pin_memory=False)
    weather_model = wrmodel.ClassiModel(arch=opt.arch, gpus=[opt.gpu], from_pretrained=False)
    if opt.model_url is not None:
        weather_model.loadmodel(opt.model_url, ifload_fc=True)
    ims_list = []
    prelabels = []
    for i, data in enumerate(test_loader, start=0):
        images, labels, ims_name = data
        labels = labels.to(device)
        outputs = weather_model.test(images)
        pred_label = torch.argmax(outputs, 1)
        for x in pred_label:
            prelabels.append(x.item() + 1)
        for x in ims_name:
            ims_list.append(x)
    # ims_list = np.array(ims_list)
    # prelabels = np.asarray(prelabels)
    dataframe = pd.DataFrame({'FileName': ims_list, 'type': prelabels})
    dataframe.to_csv(opt.dataroot + "/test.csv", index=False, sep=',')

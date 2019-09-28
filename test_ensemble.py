# -*- coding:utf-8 -*-
###
# File: test_ensemble.py
# Created Date: Tuesday, September 17th 2019, 8:07:30 pm
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
import PIL
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from efficientnet_pytorch import EfficientNet
import ensemble_model as ensemble
import csv_dataset as csvdset
import config
import time
import data_process


label_list = ['garbage', 'health', 'others', 'waterpollute']
save_file = "/root/notebook/river_result_ensemble.txt"


if __name__ == "__main__":
    conf = config.Config()
    opt = conf.create_opt()
    emodel = ensemble.EnsembleModel(opt.fold_num, opt)
    model_url = "/root/notebook/model/model_ensemble-5.pth"
    emodel.loadmodel(model_url)
    emodel._eval()
    st = time.time()
    image_size = EfficientNet.get_image_size(opt.arch)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transforms = transforms.Compose([
        # transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        data_process.ResizeFill(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    st = time.time()
    image_list, pred_labels = emodel.test_set(opt.testcsv, opt.testroot, transform=test_transforms, bs=128)
    et = time.time()
    print("test set use time: %f" % (et-st))
    # with open(save_file, mode='w') as f:
    #     for i in range(len(image_list)):
    #         f.write(image_list[i]+" "+str(label_list[pred_labels[i]]) + '\n')
    dataframe = pd.DataFrame({'FileName': image_list, 'type': pred_labels})
    dataframe.to_csv("./test-10-b2-seq-tb-bs.csv", index=False, sep=',')

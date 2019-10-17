# -*- coding:utf-8 -*-
###
# File: main.py
# Created Date: Sunday, September 15th 2019, 12:28:36 pm
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
import json
import time
import config
import classifi_model as cmodel
import csv_dataset as csvdset
from efficientnet_pytorch import EfficientNet
import torch
import torchvision.transforms as transforms
import PIL
import data_process


if __name__ == "__main__":
    conf = config.Config()
    opt = conf.create_opt()
    print(opt)
    # gen the fold csv foles
    if opt.fold_need_gen:
        print("generate fold csv files")
        train_csvs, valid_csvs = csvdset.generate_k_fold_seq(opt.traincsv, opt.fold_csv_dir, opt.fold_num)
    else:
        if os.path.exists(os.path.join(opt.fold_csv_dir, "%d/trian_%d.csv" % (0, 0))):
            print("use the existing csv files")
            train_csvs, valid_csvs = [], []
            for i in range(opt.fold_num):
                train_csvs.append(os.path.join(opt.fold_csv_dir, "%d/trian_%d.csv" % (i, i)))
                valid_csvs.append(os.path.join(opt.fold_csv_dir, "%d/valid_%d.csv" % (i, i)))
        else:
            print("generate fold csv files")
            train_csvs, valid_csvs = csvdset.generate_k_fold_seq(opt.traincsv, opt.fold_csv_dir, opt.fold_num)

    if 'efficientnet' in opt.arch:
        image_size = EfficientNet.get_image_size(opt.arch)
        print("efficientnet image size: {}".format(image_size))
    else:
        image_size = opt.imageSize
        print('not efficientnet image size: {}'.format(image_size))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        # data_process.RandomCrop(scale=(0.6, 1.0)),
        data_process.ExifTranspose(),
        data_process.RandomAffine(30, p=0.85, translate=(0.15, 0.15)),
        data_process.ResizeFill(image_size),
        data_process.RandomBlur(p=0.5),
        data_process.RandomNoise(p=0.75),
        # data_process.RandomErasing(p=0.85),
        data_process.RandomShear(p=0.9),
        # data_process.RandomHSVShift(),
        # data_process.RandomContrast(p=0.8),
        data_process.RandomFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    valid_transforms = transforms.Compose([
        data_process.ExifTranspose(),
        data_process.ResizeFill(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    score_list, acc_list = [], []
    for fold_idx in range(opt.fold_begin, opt.fold_num):
        print("training on %d fold" % fold_idx)
        opt.model_save_dir = os.path.join(opt.model_base_dir, opt.arch)
        trian_csv = train_csvs[fold_idx]
        valid_csv = valid_csvs[fold_idx]
        train_dataset = csvdset.CsvDataset(trian_csv, opt.trainroot, transform=train_transforms)
        valid_dataset = csvdset.CsvDataset(valid_csv, opt.trainroot, transform=valid_transforms)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers, drop_last=True,
            pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers, drop_last=False,
            pin_memory=False)
        classi_model = cmodel.ClassiModel(
            arch=opt.arch, gpus=[opt.gpu],
            optimv=opt.optimizer, num_classes=opt.num_classes, lr=opt.lr_list[0],
            weight_decay=opt.weight_decay, from_pretrained=opt.from_pretrained, ifcbam=opt.ifcbam,
            fix_bn_v=opt.fix_bn, criterion_v=opt.criterion_v)
        print("there are %d images in the training set, %d in the validation set" %
              (len(train_dataset), len(valid_dataset)))
        avg_valid_acc, avg_valid_score, _, _ = classi_model.train_fold(train_loader, valid_loader, fold_idx, opt)
        acc_list.append(avg_valid_acc)
        score_list.append(avg_valid_score)

    # 打印在验证集上的平均准确率
    print('total accuracy:', sum(acc_list) / opt.fold_num)
    print('total score:', sum(score_list) / opt.fold_num)

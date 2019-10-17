# -*- coding:utf-8 -*-
###
# File: train_baseline.py
# Created Date: Monday, October 14th 2019, 9:22:07 pm
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
import pandas as pd
import train_baseline_config as config
import classifi_model as cmodel
import csv_dataset as csvdset
from efficientnet_pytorch import EfficientNet
import numpy as np
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
    # reset the fold_num to a little number for save training baseline time
    opt.fold_num = 1
    # intialize baseline csv dataframe
    bl_csv_dict = {}
    if os.path.exists(opt.baseline_csv):
        bl_csv_info = pd.read_csv(opt.baseline_csv)
        bl_csv_dict['architecture'] = list(bl_csv_info['architecture'])
        for i in range(opt.fold_num):
            bl_csv_dict['f%d_train_accur' % i] = list(bl_csv_info['f%d_train_accur' % i])
        bl_csv_dict['avg_train_accur'] = list(bl_csv_info['avg_train_accur'])
        for i in range(opt.fold_num):
            bl_csv_dict['f%d_train_score' % i] = list(bl_csv_info['f%d_train_score' % i])
        bl_csv_dict['avg_train_score'] = list(bl_csv_info['avg_train_score'])
        for i in range(opt.fold_num):
            bl_csv_dict['f%d_valid_accur' % i] = list(bl_csv_info['f%d_valid_accur' % i])
        bl_csv_dict['avg_train_accur'] = list(bl_csv_info['avg_train_accur'])
        for i in range(opt.fold_num):
            bl_csv_dict['f%d_valid_score' % i] = list(bl_csv_info['f%d_valid_score' % i])
        bl_csv_dict['avg_valid_score'] = list(bl_csv_info['avg_valid_score'])
        bl_csv_info = pd.DataFrame(bl_csv_dict)
        print(bl_csv_info)
    else:
        bl_csv_dict['architecture'] = []
        for i in range(opt.fold_num):
            bl_csv_dict['f%d_train_accur' % i] = []
        bl_csv_dict['avg_train_accur'] = []
        for i in range(opt.fold_num):
            bl_csv_dict['f%d_train_score' % i] = []
        bl_csv_dict['avg_train_score'] = []
        for i in range(opt.fold_num):
            bl_csv_dict['f%d_valid_accur' % i] = []
        bl_csv_dict['avg_train_accur'] = []
        for i in range(opt.fold_num):
            bl_csv_dict['f%d_valid_score' % i] = []
        bl_csv_dict['avg_valid_score'] = []
        # bl_csv_info = pd.DataFrame(bl_csv_dict)
    arch_id_begin = len(bl_csv_dict['avg_valid_score'])

    for arch_id in range(arch_id_begin, len(opt.archs)):
        # fix the random seed
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        arch = opt.archs[arch_id]
        bl_csv_dict['architecture'].append(arch)

        if 'efficientnet' in arch:
            image_size = EfficientNet.get_image_size(arch)
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
            data_process.RandomCrop(p=0.9, scale=(0.75, 1.0)),
            data_process.ResizeFill(image_size),
            data_process.RandomBlur(p=0.5),
            data_process.RandomNoise(p=0.75),
            # data_process.RandomErasing(p=0.85),
            data_process.RandomShear(p=0.9),
            # data_process.RandomHSVShift(),
            # data_process.RandomContrast(p=0.8),
            data_process.RandomBrightness(p=0.8),
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
        valid_score_list, valid_acc_list = [], []
        train_score_list, train_acc_list = [], []
        for fold_idx in range(opt.fold_begin, opt.fold_num):
            print("%d/%dth network: %s training on %d fold" % (arch_id, len(opt.archs), arch, fold_idx))
            # set model save directory
            opt.model_save_dir = os.path.join(opt.model_base_dir, arch)
            # select the csv file and begin one fold train
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
                arch=arch, gpus=[opt.gpu],
                optimv=opt.optimizer, num_classes=opt.num_classes, lr=opt.lr_list[0],
                weight_decay=opt.weight_decay, from_pretrained=opt.from_pretrained, ifcbam=opt.ifcbam,
                fix_bn_v=opt.fix_bn, criterion_v=opt.criterion_v)
            print("there are %d images in the training set, %d in the validation set" %
                  (len(train_dataset), len(valid_dataset)))
            valid_acc, valid_score, train_acc, train_score = classi_model.train_fold(
                train_loader, valid_loader, fold_idx, opt)
            bl_csv_dict['f%d_train_accur' % fold_idx].append(train_acc)
            bl_csv_dict['f%d_train_score' % fold_idx].append(train_score)
            bl_csv_dict['f%d_valid_accur' % fold_idx].append(valid_acc)
            bl_csv_dict['f%d_valid_score' % fold_idx].append(valid_score)

            valid_acc_list.append(valid_acc)
            valid_score_list.append(valid_score)
            train_acc_list.append(train_acc)
            train_score_list.append(train_score)

        bl_csv_dict['avg_train_accur'].append(sum(train_acc_list)/len(train_acc_list))
        bl_csv_dict['avg_train_score'].append(sum(train_score_list)/len(train_score_list))
        bl_csv_dict['avg_valid_accur'].append(sum(valid_acc_list)/len(valid_acc_list))
        bl_csv_dict['avg_valid_score'].append(sum(valid_score_list)/len(valid_score_list))
        bl_csv_info = pd.DataFrame(bl_csv_dict)
        bl_csv_info.to_csv(opt.baseline_csv, sep=',', index=False, float_format='%.4f')

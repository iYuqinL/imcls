# -*- coding:utf-8 -*-
###
# File: save_filename_csv.py
# Created Date: Saturday, September 14th 2019, 2:06:35 pm
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
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


def _get_dir_list(path):
    dir_list = [
        f for f in os.listdir(path)
        if not os.path.isfile(os.path.join(path, f))
    ]
    dir_list.sort(key=str.lower)
    return dir_list


def _get_file_list(path, extension=None):
    if extension is None:
        file_list = [
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
    else:
        file_list = [
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
            and os.path.splitext(f)[1] == extension
        ]
    file_list.sort(key=str.lower)
    return file_list


def create_csv_from_dataroot(dataroot, save_name):
    dir_list = _get_dir_list(dataroot)
    file_names = []
    labels = []
    label = 0
    for dir_name in dir_list:
        path = os.path.join(dataroot, dir_name)
        files = _get_file_list(path)
        for file in files:
            file_names.append(dir_name + "/" + file)
            labels.append(label)
        label += 1
    dataframe = pd.DataFrame({'FileName': file_names, 'type': labels})
    dataframe.to_csv(save_name, index=False, sep=',')


def create_test_csv(dataroot, save_name):
    files = _get_file_list(dataroot)
    labels = [0 for x in files]
    dataframe = pd.DataFrame({'FileName': files, 'type': labels})
    dataframe.to_csv(save_name, index=False, sep=',')


if __name__ == "__main__":
    train_path = "/home/data/V1.0/train"
    train_save_name = "/root/notebook/water_pollution/data/train.csv"
    test_path = "/root/notebook/water_pollution/data/test"
    test_save_name = "/root/notebook/water_pollution/data/test.csv"
    create_csv_from_dataroot(train_path, train_save_name)
#     create_test_csv(test_path, test_save_name)

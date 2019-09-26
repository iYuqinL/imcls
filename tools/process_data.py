# -*- coding:utf-8 -*-
###
# File: process_data.py
# Created Date: Sunday, September 15th 2019, 12:39:19 pm
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
from PIL import Image
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


class MyDataset(object):
    def __init__(self, dataroot,  extension="", save_root=""):
        super(MyDataset, self).__init__()
        self.root_dir = dataroot
        self.extension = extension
        self.save_root = save_root
        self.saveim_root = os.path.join(self.save_root, "images")
        os.makedirs(self.saveim_root, exist_ok=True)
        self.dir_list = _get_dir_list(self.root_dir)
        self.image_list = []
        self.image_save_name = []
        self.labels = []
        label = 0
        for dir_name in self.dir_list:
            path = os.path.join(dataroot, dir_name)
            files = _get_file_list(path)
            cnt = 0
            for file in files:
                self.image_list.append(dir_name + "/" + file)
                # self.image_name.append(file)
                saveim_name = str(label) + "-" + str(cnt) + ".jpg"
                self.image_save_name.append(saveim_name)
                self.labels.append(label)
                cnt += 1
            label += 1

    def traverse(self):
        for i in range(len(self.image_list)):
            self.__getitem__(i)
        dataframe = pd.DataFrame({'FileName': self.image_save_name, 'type': self.labels})
        csv_name = os.path.join(self.save_root, "process_data.csv")
        dataframe.to_csv(csv_name, index=False, sep=',')

    def save_csv(self):
        dataframe = pd.DataFrame({'FileName': self.image_save_name, 'type': self.labels})
        csv_name = os.path.join(self.save_root, "process_data.csv")
        dataframe.to_csv(csv_name, index=False, sep=',')

    def __getitem__(self, index):
        im_name = os.path.join(self.root_dir, (self.image_list[index] + self.extension))
        im_save_name = os.path.join(self.saveim_root, self.image_save_name[index])
        image = Image.open(im_name)
        label = int(self.labels[index])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(im_save_name, quality=95)

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    dataroot = "/home/yusnows/Documents/DataSets/competition/weatherRecog/HM-1700/"
    saveroot = "/home/yusnows/Documents/DataSets/competition/weatherRecog/HM-1700/process"
    dataset = MyDataset(dataroot, save_root=saveroot)
    # dataset.traverse()
    dataset.save_csv()

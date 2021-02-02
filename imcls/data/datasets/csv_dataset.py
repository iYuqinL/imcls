# -*- coding:utf-8 -*-
###
# File: data_loader.py
# Created Date: Wednesday, September 11th 2019, 11:40:42 am
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
from PIL import Image
import torch
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
# import random
from ..transforms import operations as data_process

__all__ = ['get_num_classes_by_labels', 'generate_k_fold_sampling', 'generate_k_fold_seq', 'CsvDataset']

Image.MAX_IMAGE_PIXELS = 10000000000


def get_num_classes_by_labels(labels_info):
    class_base = np.Infinity
    num_classes = 0
    # labels_info = str(labels_info)
    for label in labels_info:
        label = [int(x) for x in label.split(';')]
        label = np.asarray(label)
        # print(label)
        if class_base > np.min(label):
            class_base = np.min(label)
        if num_classes < np.max(label):
            num_classes = np.max(label)
    if class_base == 0:
        num_classes += 1
    return class_base, num_classes


def generate_k_fold_sampling(csv_file, out_dir, k):
    '''
    * 用csv_file进行k次划分，用于交叉验证，每次划分，都有1/k的样本作为验证集
    * 划分后的train_*.csv和valid_*.csv保存在output_dir中
    '''
    print("the csv_file which to seperate k folds is {}".format(csv_file))
    csv_info = pd.read_csv(csv_file)
    image_name = np.asarray(csv_info.iloc[:, 0])
    labels = np.asarray(csv_info.iloc[:, 1])
    size = len(image_name) // k
    folds = []
    for i in range(k):
        indices = np.random.randint(0, len(image_name), size=size)
        folds.append(np.asarray([image_name[indices], labels[indices]]))
        os.makedirs(os.path.join(out_dir, "%d/" % i), exist_ok=True)
    train_csv_list = []
    valid_csv_list = []
    for i in range(k):  # 作为验证集
        valid = folds[i]
        train = []
        for j in range(k):
            if j == i:
                continue
            train.append(folds[j])
        train_v = np.concatenate(train, axis=1)
        train_frame = pd.DataFrame({'FileName': train_v[0], 'type': train_v[1]})
        valid_frame = pd.DataFrame({'FileName': valid[0], 'type': valid[1]})
        train_csv_list.append(os.path.join(out_dir, "%d/trian_%d.csv" % (i, i)))
        valid_csv_list.append(os.path.join(out_dir, "%d/valid_%d.csv" % (i, i)))
        train_frame.to_csv(train_csv_list[i], index=False, sep=',')
        valid_frame.to_csv(valid_csv_list[i], index=False, sep=',')
    return train_csv_list, valid_csv_list


def generate_k_fold_seq(csv_file, out_dir, k, shuffle=True):
    '''
    * 用csv_file进行k次划分，用于交叉验证，每次划分，都有1/k的样本作为验证集
    * 划分后的train_*.csv和valid_*.csv保存在output_dir中
    * 该函数自适应label的起始值，可以是从0开始的，也可以是从1开始
    * 该函数可以处理多标签数据也可以处理单标签数据
    * 该函数在处理数据量小于k的类别时，会把所有该类的数据都放到训练集
    '''
    print("the csv_file which to seperate k folds is {}".format(csv_file))
    csv_info = pd.read_csv(csv_file)
    # labels_info = str(csv_info.iloc[:, 1])
    labels_info = [str(x) for x in csv_info.iloc[:, 1]]
    class_base, num_classes = get_num_classes_by_labels(labels_info)
    print(num_classes)
    print(class_base)
    image_name = np.asarray(csv_info.iloc[:, 0])
    labels = np.asarray(csv_info.iloc[:, 1])
    file_info = np.c_[image_name, labels]
    if shuffle:
        np.random.shuffle(file_info)
    dataset = []
    fold_class_size = np.zeros(num_classes, dtype=np.int)
    indices = []
    for i in range(class_base, num_classes + class_base):
        indices.append([])
    for i in range(len(file_info)):
        label = str(file_info[i][1])
        label = list(set([int(x) for x in label.split(';')]))
        for t in label:
            indices[t-class_base].append(i)
    num_data_less_k = []
    for i in range(class_base, num_classes + class_base):
        fold_class_size[i - class_base] = len(indices[i - class_base]) // k
        if fold_class_size[i-class_base] == 0:
            num_data_less_k.append(indices[i-class_base])
        class_data = file_info[indices[i - class_base]]
        dataset.append(class_data)
    if len(num_data_less_k) != 0:
        num_data_less_k = np.concatenate(num_data_less_k, axis=0)
    folds = []
    base_indices = []
    for i in range(class_base, num_classes + class_base):
        base_index = np.arange(0, fold_class_size[i-class_base])
        base_indices.append(base_index)
    for fold_idx in range(k):
        fold = []
        for i in range(class_base, num_classes + class_base):
            fold.append(
                dataset[i - class_base][base_indices[i - class_base] +
                                        int((fold_idx) * fold_class_size[i - class_base])])
        fold = np.concatenate(fold, axis=0)
        folds.append(fold)
    train_csv_list = []
    valid_csv_list = []
    for i in range(k):  # 作为验证集
        os.makedirs(os.path.join(out_dir), exist_ok=True)
        valid = [folds[i]]
        valid.append(file_info[num_data_less_k])
        valid = np.concatenate(valid, axis=0)
        train = []
        for j in range(k):
            if j == i:
                continue
            train.append(folds[j])
        if len(num_data_less_k) != 0:
            train.append(file_info[num_data_less_k])
        train_v = np.concatenate(train, axis=0)
        if shuffle:
            np.random.shuffle(train_v)
            np.random.shuffle(valid)
        train_frame = pd.DataFrame({'FileName': train_v[:, 0], 'type': train_v[:, 1]})
        valid_frame = pd.DataFrame({'FileName': valid[:, 0], 'type': valid[:, 1]})
        train_csv_list.append(os.path.join(out_dir, "%02d_train.csv" % (i)))
        valid_csv_list.append(os.path.join(out_dir, "%02d_valid.csv" % (i)))
        train_frame.to_csv(train_csv_list[i], index=False, sep=',')
        valid_frame.to_csv(valid_csv_list[i], index=False, sep=',')
    return train_csv_list, valid_csv_list


class CsvDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, extension=""):
        super(CsvDataset, self).__init__()
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        print("csv file is: {}".format(csv_file))
        self.labels = np.asarray(self.data_info.iloc[:, 1])
        self.image_list = self.data_info.iloc[:, 0]
        self.transform = transform
        self.extension = extension

    def __getitem__(self, index):
        im_name = os.path.join(self.root_dir, (self.image_list[index] + self.extension))
        # print(im_name)
        image = Image.open(im_name)
        label = int(self.labels[index])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # print("num classes: %d, class_base: %d" % (self.num_classes, self.class_base))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_list)


class CsvDatasetTest(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, extension=""):
        super(CsvDatasetTest, self).__init__()
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.labels = np.asarray(self.data_info.iloc[:, 1])
        self.image_list = self.data_info.iloc[:, 0]
        self.transform = transform
        self.extension = extension

    def __getitem__(self, index):
        im_name = os.path.join(self.root_dir, (self.image_list[index] + self.extension))
        # print(im_name)
        image = Image.open(im_name)
        label = int(self.labels[index])

        if image.mode != 'RGB':
            image = image.convert('RGB')
        # target = int(self.target_labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label, self.image_list[index] + self.extension

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    csv_file = "/home/yusnows/Documents/cloudRecog/data/processed/train_labels.csv"
    train_csv = "/home/yusnows/Documents/cloudRecog/data/processed/train_labels.csv"
    train_root = "/home/yusnows/Documents/cloudRecog/data/processed/train/"
    # generate_k_fold(csv_file, "./folds", 10)
    # generate_k_fold_seq(csv_file, "./folds-test", 10)
    image_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0), interpolation=Image.BICUBIC),
        data_process.RandomCrop(scale=(0.4, 1.0)),
        data_process.ResizeFill(image_size),
        # data_process.RandomToHSVToRGB(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = CsvDataset(train_csv, train_root, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=16, drop_last=True,
        pin_memory=False)

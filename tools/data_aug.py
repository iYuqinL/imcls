# -*- coding:utf-8 -*-
###
# File: data_aug.py
# Created Date: Tuesday, October 1st 2019, 2:13:36 pm
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
import numpy as np
import smote_variants as sv
import csv_dataset as csvdset
import torch
import PIL.Image as Image
import os
import pandas as pd
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import data_process


im_dtype_np = np.float16
im_dtype_torch = torch.float32
im_dtype_np_torch = np.float32


def load_dataset(csv_file, dataroot, imSize=320, label=None):
    image_size = imSize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    datatransform = transforms.Compose([data_process.ResizeFill(image_size), transforms.ToTensor(), normalize])
    dataset = csvdset.CsvDataset(csv_file, dataroot, transform=datatransform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False, pin_memory=False)
    print(len(dataset))
    # exit()
    X, Y = [], []
    for i, data in enumerate(dataloader, start=0):
        ims, labels = data
        ims = ims.numpy().astype(im_dtype_np)
        labels = labels.numpy().argmax(axis=1).astype(np.int32)
        if label is None:
            X.append(ims)
            Y.append(labels)
        else:
            if isinstance(label, int):
                for j in range(labels.shape[0]):
                    if label == labels[j]:
                        X.append(ims[j][np.newaxis, :])
                        Y.append(np.asarray([labels[j]]))
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X, Y


def unnormal_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = torch.from_numpy(image.astype(im_dtype_np_torch))
    mean = torch.tensor(mean, dtype=im_dtype_torch)
    std = torch.tensor(std, dtype=im_dtype_torch)
    std = std.reshape(std.shape[0], 1, 1)
    mean = mean.reshape(mean.shape[0], 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1)
    image = F.to_pil_image(image)
    return image


def save_dataset(X, Y, rootdir, csv_name="train.csv", dataname="train/"):
    classes = np.unique(Y)
    num_class = len(classes)
    class_max = np.max(classes)
    class_cnt = np.zeros(class_max+1, dtype=np.int32)
    im_names = []
    labels = []
    impath = os.path.join(rootdir, dataname)
    os.makedirs(impath, exist_ok=True)
    for i in range(X.shape[0]):
        im, label = X[i], Y[i]
        im = unnormal_image(im)
        imname = "%02d-%05d.jpg" % (label, class_cnt[label])
        im_names.append(imname)
        labels.append(label)
        imname = os.path.join(impath, imname)
        class_cnt[label] += 1
        im.save(imname)
    csv_name = os.path.join(rootdir, csv_name)
    dataframe = pd.DataFrame({'FileName': im_names, 'type': labels})
    dataframe.to_csv(csv_name, index=False, sep=',')
    return None


if __name__ == "__main__":
    csv_file = "/home/yusnows/Documents/DataSets/competition/weatherRecog/process/Train_label.csv"
    dataroot = "/home/yusnows/Documents/DataSets/competition/weatherRecog/process/train/"
    save_root = "/home/yusnows/Documents/DataSets/competition/weatherRecog/data_merge/origin_smote/"
    imSize = 480
    # X_std, Y_std = load_dataset(csv_file, dataroot, imSize=imSize, label=1)
    # [print('Class {} has {} instances'.format(label, count))
    #  for label, count in zip(*np.unique(Y_std, return_counts=True))]
    X, Y = load_dataset(csv_file, dataroot, imSize=imSize, label=None)
    [print('Class {} has {} instances'.format(label, count))
     for label, count in zip(*np.unique(Y, return_counts=True))]
    # X = np.concatenate([X_std, X], axis=0)
    # Y = np.concatenate([Y_std, Y], axis=0)
    # smote = sv.KernelADASYN()
    smote = sv.MulticlassOversampling(sv.kmeans_SMOTE(proportion=1.0, n_neighbors=6, n_clusters=20, irt=2.0, n_jobs=16))
    print(X.shape)
    X = X.reshape(X.shape[0], -1)
    X_resampled, Y_resampled = smote.sample(X, Y)
    [print('Class {} has {} instances after oversampling'.format(label, count))
     for label, count in zip(*np.unique(Y_resampled, return_counts=True))]
    X_resampled = X_resampled.reshape(-1, 3, imSize, imSize)
    # X_resampled = X_resampled.transpose(0, 2, 3, 1)
    save_dataset(X_resampled, Y_resampled, save_root)

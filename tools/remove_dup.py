# -*- coding:utf-8 -*-
###
# File: remove_dup.py
# Created Date: Thursday, October 3rd 2019, 7:19:32 pm
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
import pandas as pd
import numpy as np

if __name__ == "__main__":
    csv_file = "train.csv"
    csv_save = "train-remove-dup.csv"
    dup_file = "duplicate_im.txt"
    dup_list = []
    with open(dup_file, 'r') as f:
        line = f.readline()
        while line:
            dup_list.append(line)
            line = f.readline()
    csv_info = pd.read_csv(csv_file)
    new_im_names, new_labels = [], []
    im_names = np.array(csv_info.iloc[:, 0])
    labels = [str(x) for x in csv_info.iloc[:, 1]]
    for i in range(len(labels)):
        ifdup = False
        for dup_name in dup_list:
            if dup_name == im_names[i]:
                print('remove {}'.format(dup_name))
                ifdup = True
                break
        if ifdup is False:
            new_im_names.append(im_names[i])
            new_labels.append(labels[i])
    dataframe = pd.DataFrame({'FileName': new_im_names, 'type': new_labels})
    dataframe.to_csv(csv_save, index=False, sep=',')

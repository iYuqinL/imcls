# -*- coding:utf-8 -*-
###
# File: make_submit_file.py
# Created Date: Thursday, September 12th 2019, 11:44:32 am
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
import torch
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


if __name__ == "__main__":
    example_csv = "/home/yusnows/Documents/DataSets/competition/weatherRecog/original/submit_example.csv"
    # result_csv = "/home/yusnows/Documents/DataSets/competition/weatherRecog/process/test-e60.csv"
    result_csv = "../test_results/test-10-b2-seq-tb-bs-1002.csv"
    submit_csv = "/home/yusnows/Documents/DataSets/competition/weatherRecog/submits/csv/submit-en-10-b2-seq-tb-bs-1002.csv"
    example_file = pd.read_csv(example_csv)
    result_file = pd.read_csv(result_csv)
    image_list = example_file.iloc[:, 0]
    labels = np.asarray(result_file.iloc[:, 1])
    labels = labels + 1
    dataframe = pd.DataFrame({'FileName': image_list, 'type': labels})
    dataframe.to_csv(submit_csv, index=False, sep=',')

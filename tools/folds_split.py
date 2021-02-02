# -*- coding:utf-8 -*-
###
# File: folds_split.py
# Created Date: Saturday, November 7th 2020, 3:13:03 pm
# Author: yusnows
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 yusnows
#
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
from imcls.data.datasets import generate_k_fold_seq


if __name__ == "__main__":
    csv_file = "DataSet/S2/mergeS1/balance/train_balance/train_balance.csv"
    out_dir = "DataSet/S2/mergeS1/balance/train_balance/train_folds"
    generate_k_fold_seq(csv_file, out_dir, 6)

# -*- coding:utf-8 -*-
###
# File: train_en.py
# Created Date: Saturday, November 7th 2020, 3:24:32 pm
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
import os
import time
import glob
import torch
from imcls.config import get_cfg
from imcls.modeling import ClsModel


def train_fold(cfg, fold_idx, train_csv, valid_csv, output_dir,
               start_epoch=0, epochs=320, premodels_dir=None, gpu=0):
    cfg.DATA.DATASETS.TRAIN_CSV = train_csv
    cfg.DATA.DATASETS.VALID_CSV = valid_csv
    cfg.OUTPUT_DIR = os.path.join(output_dir, "%02d" % fold_idx)
    model_url = None
    if premodels_dir is not None:
        model_url = os.path.join(premodels_dir, "%02d" % fold_idx, "cls_model_last.pth")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    clsmodel = ClsModel(cfg, model_url=model_url, gpu=gpu)
    cls_net = clsmodel.train_model(cfg, start_epoch, epochs)
    model_file = os.path.join(cfg.OUTPUT_DIR, "cls_model_final.pth")
    torch.save(cls_net.state_dict(), model_file)


def train_folds(cfg, folds_dir, premodels_dir, output_dir, fold_nums=None):
    cdir = os.getcwd()
    os.chdir(folds_dir)
    train_csvs = glob.glob("*_train.csv")
    valid_csvs = glob.glob("*_valid.csv")
    os.chdir(cdir)
    train_csvs.sort(key=lambda x: int(x[:-10]))
    valid_csvs.sort(key=lambda x: int(x[:-10]))
    train_csvs = [os.path.join(folds_dir, i) for i in train_csvs]
    valid_csvs = [os.path.join(folds_dir, i) for i in valid_csvs]
    print(train_csvs)
    print(valid_csvs)
    if fold_nums is None:
        fold_nums = range(len(train_csvs))
    for fold_idx in fold_nums:
        fold_st = time.time()
        train_fold(cfg, fold_idx, train_csvs[fold_idx], valid_csvs[fold_idx], output_dir, premodels_dir=premodels_dir)
        fold_et = time.time()
        print("train on fold: %02d, used time: %.2fs" % (fold_idx, fold_et-fold_st))


if __name__ == "__main__":
    cfg_file = "configs/call_smoke_cls/resnest200-en-S2.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)

    cfg.SOLVER.MAX_ITER = 60000
    cfg.SOLVER.WARMUP_ITERS = 2400
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / cfg.SOLVER.WARMUP_ITERS
    cfg.DATA.DATASETS.ROOT_DIR = "DataSet/S2/mergeS1/balance/train_balance"

    folds_dir = "DataSet/S2/mergeS1/balance/train_balance/s2_folds"
    premodels_dir = "train-outs/S2/resnest200-en/"
    output_dir = "train-outs/S2/resnest200-en-02/"
    train_folds(cfg, folds_dir, premodels_dir, output_dir, fold_nums=[0, 2, 5])

# -*- coding:utf-8 -*-
###
# File: test_en.py
# Created Date: Wednesday, November 11th 2020, 7:26:37 pm
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
import json
import torch
import torch.utils.data
import torchvision.transforms as tvtrans
from imcls.modeling import ClsNetwork
from imcls.config import get_cfg
import imcls.data.transforms as mytrans
from imcls.data.datasets import CallSmokeTest

# cls_names = ["normal", "calling", "smoking"]
cls_names = ["normal", "calling", "smoking", "smoking_calling"]


def test_fold(cfg, model_url, fold_idx, tta=False, gpu=0):
    device = torch.device("cuda:%d" % gpu if gpu >= 0 else "cpu")
    # print(cfg.dump())
    print("begin testing %02dth fold......" % fold_idx)
    test_st = time.time()
    cls_model = ClsNetwork(cfg)
    cls_model = cls_model.to(device)
    print("load trained model parameters from [%s]" % model_url)
    cls_model.load_state_dict(torch.load(model_url))
    cls_model.eval()
    trans = tvtrans.Compose(
        [
            mytrans.ECenterCrop(imgsize=cfg.DATA.TEST_SIZE[0]),
            tvtrans.ToTensor(),
            tvtrans.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    testset = CallSmokeTest(cfg.DATA.DATASETS.TEST_DIR, trans)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.DATA.BATCHSIZE, num_workers=cfg.DATA.NUM_WORKERS,
        shuffle=False, drop_last=False, pin_memory=False)

    pred_fold = []
    im_names = []

    for idx, data in enumerate(testloader):
        image, im_name = data
        image = image.to(device)
        with torch.no_grad():
            preds = cls_model(image)
        pred_fold.append(preds)
        im_names += list(im_name)
    pred_fold = torch.cat(pred_fold, dim=0)
    if tta:
        trans2 = tvtrans.Compose(
            [
                tvtrans.RandomHorizontalFlip(p=1.0),
                mytrans.ECenterCrop(imgsize=cfg.DATA.TEST_SIZE[0]),
                tvtrans.ToTensor(),
                tvtrans.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
            ]
        )
        testset2 = CallSmokeTest(cfg.DATA.DATASETS.TEST_DIR, trans2)
        testloader2 = torch.utils.data.DataLoader(
            testset2, batch_size=cfg.DATA.BATCHSIZE, num_workers=cfg.DATA.NUM_WORKERS,
            shuffle=False, drop_last=False, pin_memory=False)

        pred_fold2 = []
        for idx, data2 in enumerate(testloader2):
            image2, _ = data2
            image2 = image2.to(device)
            with torch.no_grad():
                preds2 = cls_model(image2)
            pred_fold2.append(preds2)
            # im_names += list(im_name)
        pred_fold2 = torch.cat(pred_fold2, dim=0)
        pred_fold = (pred_fold + pred_fold2)/2
    test_et = time.time()
    print("test %02dth fold end!, used time: [total: %.4f, avg: %.4f]" %
          (fold_idx, test_et-test_st, (test_et-test_st)/len(testset)))
    return pred_fold, im_names


def test_folds(base_dir, fold_nums, cfg_name, model_name, savedir, tta=False, gpu=0):
    assert max(fold_nums) < 100, "fold_nums too large"
    os.makedirs(savedir, exist_ok=True)
    btime = time.time()
    preds = []
    im_names = []
    print("begin testing folds......")
    for fold_idx in fold_nums:
        cfg_file = os.path.join(base_dir, "%02d" % fold_idx, cfg_name)
        cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
        model_url = os.path.join(base_dir, "%02d" % fold_idx, model_name)
        cfg.NETWORK.PRETRAINED = False
        cfg.DATA.BATCHSIZE = 32
        cfg.DATA.NUM_WORKERS = 8
        cfg.DATA.DATASETS.TEST_DIR = "DataSet/S2/testA"
        pred_fold, im_names = test_fold(cfg, model_url, fold_idx, tta, gpu)
        preds.append(pred_fold)

    preds = torch.stack(preds, dim=0)
    # preds = preds.softmax(dim=-1)
    preds = preds.mean(dim=0)
    arg_ind = preds.argmax(dim=1)
    print(preds.shape, arg_ind.shape)
    # for i in range(len(arg_ind)):
    #     preds[i, arg_ind[i]] *= 1.5
    preds = preds.softmax(dim=-1)
    results = []
    for i in range(len(arg_ind)):
        results.append({"image_name": im_names[i], "category": cls_names[arg_ind[i]],
                        "score": preds[i][arg_ind[i]].item()})
    etime = time.time()
    print("test total folds end!, used time: [total: %.4f, avg: %.4f]" %
          (etime-btime, (etime-btime)/len(arg_ind)))
    save_file = os.path.join(savedir, "result.json")
    with open(save_file, 'w') as f:
        json.dump(results, f)
    save_file = os.path.join(savedir, "result-1.json")
    with open(save_file, 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":

    base_dir = "train-outs/S2/resnest200-en-02/"
    cfg_name = "cfg.yaml"
    model_name = "cls_model_0100.pth"
    res_save_dir = "results/S2/testA/resnest200-en-02/epoch_0100_025-tta/"
    tta = True

    test_folds(base_dir, [0, 2, 5], cfg_name, model_name, res_save_dir, tta)

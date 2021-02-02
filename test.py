# -*- coding:utf-8 -*-
###
# File: test.py
# Created Date: Thursday, October 1st 2020, 12:30:48 pm
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


def test(cfg, savedir, model_url, gpu=0):
    os.makedirs(savedir, exist_ok=True)
    device = torch.device("cuda:%d" % gpu if gpu >= 0 else "cpu")
    print(cfg.dump())
    cls_model = ClsNetwork(cfg)
    cls_model = cls_model.to(device)
    print("load trained model parameters from [%s]" % model_url)
    cls_model.load_state_dict(torch.load(model_url))
    cls_model.eval()
    print("begin testing......")

    trans = tvtrans.Compose(
        [
            # tvtrans.Resize(size=cfg.DATA.SIZE),
            mytrans.ResizeFill(size=cfg.DATA.TEST_SIZE),
            # mytrans.ECenterCrop(imgsize=cfg.DATA.TEST_SIZE[0]),
            tvtrans.ToTensor(),
            tvtrans.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    testset = CallSmokeTest(cfg.DATA.DATASETS.TEST_DIR, trans)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.DATA.BATCHSIZE, num_workers=cfg.DATA.NUM_WORKERS,
        shuffle=False, drop_last=False, pin_memory=False)

    results = []
    test_st = time.time()
    for idx, data in enumerate(testloader):
        image, im_name = data
        image = image.to(device)
        with torch.no_grad():
            preds = cls_model(image)
        arg_ind = preds.argmax(dim=1)
        preds = torch.softmax(preds, dim=1)
        for i in range(len(arg_ind)):
            results.append({"image_name": im_name[i], "category": cls_names[arg_ind[i]],
                            "score": preds[i][arg_ind[i]].item()})
    test_et = time.time()
    print("testing end!, used time: [total: %.4f, avg: %.4f]" %
          (test_et-test_st, (test_et-test_st)/len(testset)))
    save_file = os.path.join(savedir, "result.json")
    with open(save_file, 'w') as f:
        json.dump(results, f)
    save_file = os.path.join(savedir, "result-1.json")
    with open(save_file, 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    cfg_file = "train-outs/S2/resnest200-05/cfg.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.NETWORK.PRETRAINED = False
    cfg.DATA.BATCHSIZE = 32
    cfg.DATA.NUM_WORKERS = 8
    cfg.DATA.DATASETS.TEST_DIR = "DataSet/S2/testA"

    res_save_dir = "results/S2/testA/resnest200-05/epoch_0140/"
    model_url = "train-outs/S2/resnest200-05/cls_model_0140.pth"
    test(cfg, res_save_dir, model_url)

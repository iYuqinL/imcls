# -*- coding:utf-8 -*-
###
# File: train.py
# Created Date: Wednesday, September 30th 2020, 3:20:17 pm
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
import torch
import torch.utils.data
import torchvision.transforms as tvtrans
from imcls.modeling import ClsNetwork
from imcls.data.datasets import CallSmoke
from imcls.config import get_cfg
from imcls.solver import build_optimizer, build_lr_scheduler
from imcls import loss_trick
import imcls.data.transforms as mytrans


def train(cfg, start_epoch=0, epochs=32, model_url=None, gpu=0):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:%d" % gpu if gpu >= 0 else "cpu")
    print(cfg.dump())

    cls_model = ClsNetwork(cfg)
    cls_model = cls_model.to(device)
    if model_url is not None:
        cls_model.load_state_dict(torch.load(model_url))
    cls_model.train()

    train_trans = tvtrans.Compose(
        [
            # tvtrans.RandomRotation((0, 10), Image.BICUBIC, expand=True),
            # mytrans.RandomBrightness(p=0.733, min_factor=0.666, max_factor=1.2222),
            # mytrans.RandomNoise(p=0.5),
            # mytrans.RandomBlur(p=0.5),
            #  mytrans.RandomCrop(p=0.7, scale=(0.8666, 1)),
            # tvtrans.Resize(size=cfg.DATA.TRAIN_SIZE),
            mytrans.ResizeFill(size=cfg.DATA.TRAIN_SIZE),
            # mytrans.ERandomCrop(imgsize=cfg.DATA.TRAIN_SIZE[0]),
            tvtrans.RandomHorizontalFlip(),
            tvtrans.ToTensor(),
            # mytrans.Lighting(0.1, mytrans._imagenet_pca['eigval'], mytrans._imagenet_pca['eigvec']),
            tvtrans.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
    valid_trans = tvtrans.Compose(
        [
            # tvtrans.Resize(size=cfg.DATA.VALID_SIZE),
            mytrans.ResizeFill(size=cfg.DATA.VALID_SIZE),
            # mytrans.ECenterCrop(imgsize=cfg.DATA.VALID_SIZE[0]),
            tvtrans.ToTensor(),
            tvtrans.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
    train_set = CallSmoke(cfg.DATA.DATASETS.TRAIN_DIR, transforms=train_trans)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.DATA.BATCHSIZE, num_workers=cfg.DATA.NUM_WORKERS,
        shuffle=True, drop_last=True, pin_memory=False)

    cfg.SOLVER.MAX_ITER = round(140*len(train_loader))
    cfg.SOLVER.WARMUP_ITERS = max(1000, round(4*len(train_loader)))
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / cfg.SOLVER.WARMUP_ITERS
    print("max iteration: %d" % cfg.SOLVER.MAX_ITER)
    with open(os.path.join(cfg.OUTPUT_DIR, "cfg.yaml"), 'w') as f:
        f.write(cfg.dump())
    print_interval = int(len(train_loader)/10)

    valid_set = CallSmoke(cfg.DATA.DATASETS.VALID_DIR, transforms=valid_trans)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=cfg.DATA.BATCHSIZE, num_workers=cfg.DATA.NUM_WORKERS,
        shuffle=False, drop_last=False, pin_memory=False)

    critera = loss_trick.CrossEntropyLabelSmooth(classes=cfg.NETWORK.NUM_CLASSES)
    optimizer = build_optimizer(cfg, cls_model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    # foo step to avoid foo lr_scheduler warning
    optimizer.zero_grad()
    optimizer.step()
    # foo loop to schedule lr
    for epoch in range(0, start_epoch):
        for i in range(len(train_loader)):
            lr_scheduler.step()
    lr_list = lr_scheduler.get_lr()
    print("lr: [%.4f, %.4f]" % (min(lr_list), max(lr_list)))

    max_acc = 0
    iters_cnt = 0
    for epoch in range(start_epoch, epochs):
        loss_epoch = 0
        batch_cnt = 0
        epoch_st = time.time()
        for idx, data in enumerate(train_loader, start=0):
            batch_st = time.time()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            preds, feat = cls_model(images)
            optimizer.zero_grad()
            loss = critera(preds, labels)
            loss.backward()
            optimizer.step()
            lr_list = lr_scheduler.get_lr()
            if idx % print_interval == 0:
                print(
                    "epoch: %04d, batch: %04d, lr: [%.4f, %.4f], loss_batch: %.5f, used time: %.4fs" %
                    (epoch, idx, min(lr_list), max(lr_list), loss.item(), (time.time() - batch_st)))
            lr_scheduler.step()
            loss_epoch += loss.item()
            batch_cnt += 1
            iters_cnt += 1
            if iters_cnt > cfg.SOLVER.MAX_ITER:
                break

        epoch_et = time.time()
        loss_epoch = loss_epoch/batch_cnt
        # validate the model
        cls_model.eval()
        p_cnt = 0
        n_cnt = 0
        for idx, data in enumerate(valid_loader, start=0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                preds = cls_model(images)
            pred_la = preds.argmax(dim=1)
            p_cnt += (labels == pred_la).sum().item()
            n_cnt += (labels != pred_la).sum().item()
        cls_model.train()

        valid_acc = p_cnt/(p_cnt+n_cnt)
        print("epoch: %04d, loss_epoch: %.5f, valid_acc:%.5f, used time: %.4fs" %
              (epoch, loss_epoch, valid_acc, (epoch_et - epoch_st)))
        if epoch >= 40 and epoch % 20 == 0:
            model_file = os.path.join(cfg.OUTPUT_DIR, "cls_model_%04d.pth" % epoch)
            torch.save(cls_model.state_dict(), model_file)
        else:
            model_file = os.path.join(cfg.OUTPUT_DIR, "cls_model_last.pth")
            torch.save(cls_model.state_dict(), model_file)
            with open(os.path.join(cfg.OUTPUT_DIR, "last.txt"), 'w') as f:
                f.write("acc: % .6f\nepoch: % 04d\nloss_epoch: % .5f\n" % (max_acc, epoch, loss_epoch))

        if valid_acc > max_acc:
            max_acc = valid_acc
            with open(os.path.join(cfg.OUTPUT_DIR, "best_acc.txt"), "w") as f:
                f.write("acc: %.6f\nepoch: %04d\nloss_epoch: %.5f\n" % (max_acc, epoch, loss_epoch))
            model_file = os.path.join(cfg.OUTPUT_DIR, "cls_model_best.pth")
            torch.save(cls_model.state_dict(), model_file)
        if iters_cnt > cfg.SOLVER.MAX_ITER:
            break
    return cls_model


if __name__ == "__main__":
    cfg_file = "configs/call_smoke_cls/resnest200-S2.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)

    cfg.SOLVER.BASE_LR = 0.003333
    cfg.OUTPUT_DIR = "train-outs/S2/resnest200-05/"

    # model_url = "train-outs/S2/resnest269-01/cls_model_0040.pth"
    model_url = "train-outs/S2/resnest200-03/cls_model_0120.pth"
    start_epoch = 0
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "last.txt")):
        with open(os.path.join(cfg.OUTPUT_DIR, "last.txt")) as f:
            lines = f.readlines()
            start_epoch = int(lines[1].split(" ")[1])

    cls_model = train(cfg, start_epoch, epochs=320, model_url=model_url)
    model_file = os.path.join(cfg.OUTPUT_DIR, "cls_model_final.pth")
    torch.save(cls_model.state_dict(), model_file)

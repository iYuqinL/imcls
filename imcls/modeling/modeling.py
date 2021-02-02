# -*- coding:utf-8 -*-
###
# File: modeling.py
# Created Date: Saturday, November 7th 2020, 2:07:36 pm
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
import torch.nn as nn
import torchvision.transforms as tvtrans
from imcls.modeling import ClsNetwork
from imcls.data.datasets import CsvDataset
from imcls.solver import build_optimizer, build_lr_scheduler
from imcls import loss_trick
from imcls.utils.mixup_wrapper import MixUpWrapper
import imcls.data.transforms as mytrans


class ClsModel(object):
    def __init__(self, cfg, **kwargs) -> None:
        gpu = kwargs["gpu"] if "gpu" in kwargs else -1
        model_url = kwargs["model_url"] if "model_url" in kwargs else None

        self.cfg = cfg
        self.device = torch.device("cuda:%d" % gpu if gpu >= 0 else "cpu")
        print(cfg.dump())
        cls_model = ClsNetwork(cfg)
        cls_model = cls_model.to(self.device)
        if model_url is not None:
            print("load pretrained model from [%s]" % model_url)
            cls_model.load_state_dict(torch.load(model_url))
        cls_model.train()
        self.cls_model = cls_model
        self.optimizer = build_optimizer(cfg, self.cls_model)
        self.lr_scheduler = build_lr_scheduler(cfg, self.optimizer)
        if 0.0 < cfg.DATA.MIXUP < 1.0:
            self.critera = loss_trick.NLLMultiLabelSmooth(smoothing=cfg.NETWORK.LABEL_SMOOTHING)
        elif 0.0 < cfg.NETWORK.LABEL_SMOOTHING < 1.0:
            self.critera = loss_trick.CrossEntropyLabelSmooth(
                classes=cfg.NETWORK.NUM_CLASSES, smoothing=cfg.NETWORK.LABEL_SMOOTHING)
        else:
            self.critera = nn.CrossEntropyLoss()
        print(self.critera)
        with open(os.path.join(cfg.OUTPUT_DIR, "cfg.yaml"), 'w') as f:
            f.write(cfg.dump())

    def train_model(self, cfg, start_epoch=0, epochs=320):
        train_csv = cfg.DATA.DATASETS.TRAIN_CSV
        valid_csv = cfg.DATA.DATASETS.VALID_CSV
        train_loader, valid_loader = self._build_dataloader(cfg, train_csv, valid_csv)
        print_interval = int(len(train_loader)/10)
        # foo step to avoid foo lr_scheduler warning
        self.optimizer.zero_grad()
        self.optimizer.step()
        # foo loop to schedule lr
        for epoch in range(0, start_epoch):
            for i in range(len(train_loader)):
                self.lr_scheduler.step()
        lr_list = self.lr_scheduler.get_lr()
        print("lr: [%.4f, %.4f]" % (min(lr_list), max(lr_list)))

        max_acc = 0
        iters_cnt = 0
        for epoch in range(start_epoch, epochs):
            loss_epoch = 0
            batch_cnt = 0
            epoch_st = time.time()
            self.cls_model.train()
            for idx, data in enumerate(train_loader, start=0):
                batch_st = time.time()
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                preds, feat = self.cls_model(images)
                self.optimizer.zero_grad()
                loss = self.critera(preds, labels)
                loss.backward()
                self.optimizer.step()
                lr_list = self.lr_scheduler.get_lr()
                if idx % print_interval == 0:
                    print(
                        "epoch: %04d, batch: %04d, lr: [%.4f, %.4f], loss_batch: %.5f, used time: %.4fs" %
                        (epoch, idx, min(lr_list), max(lr_list), loss.item(), (time.time() - batch_st)))
                self.lr_scheduler.step()
                loss_epoch += loss.item()
                batch_cnt += 1
                iters_cnt += 1
                if iters_cnt > cfg.SOLVER.MAX_ITER:
                    break

            epoch_et = time.time()
            loss_epoch = loss_epoch/batch_cnt
            # validate the model
            self.cls_model.eval()
            p_cnt = 0
            n_cnt = 0
            for idx, data in enumerate(valid_loader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    preds = self.cls_model(images)
                pred_la = preds.argmax(dim=1)
                p_cnt += (labels == pred_la).sum().item()
                n_cnt += (labels != pred_la).sum().item()

            valid_acc = p_cnt/(p_cnt+n_cnt)
            print("epoch: %04d, loss_epoch: %.5f, valid_acc:%.5f, used time: %.4fs" %
                  (epoch, loss_epoch, valid_acc, (epoch_et - epoch_st)))

            model_file = os.path.join(cfg.OUTPUT_DIR, "cls_model_last.pth")
            torch.save(self.cls_model.state_dict(), model_file)
            with open(os.path.join(cfg.OUTPUT_DIR, "last.txt"), 'w') as f:
                f.write("acc: % .6f\nepoch: % 04d\nloss_epoch: % .5f\n" % (valid_acc, epoch, loss_epoch))

            if epoch >= 40 and epoch % 20 == 0:
                model_file = os.path.join(cfg.OUTPUT_DIR, "cls_model_%04d.pth" % epoch)
                torch.save(self.cls_model.state_dict(), model_file)

            if valid_acc > max_acc:
                max_acc = valid_acc
                with open(os.path.join(cfg.OUTPUT_DIR, "best_acc.txt"), "w") as f:
                    f.write("acc: %.6f\nepoch: %04d\nloss_epoch: %.5f\n" % (max_acc, epoch, loss_epoch))
                model_file = os.path.join(cfg.OUTPUT_DIR, "cls_model_best.pth")
                torch.save(self.cls_model.state_dict(), model_file)
            if iters_cnt > cfg.SOLVER.MAX_ITER:
                break
        return self.cls_model

    def _build_dataloader(self, cfg, train_csv, valid_csv=None):
        train_set, valid_set = self._build_dataset(cfg, train_csv, valid_csv)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=cfg.DATA.BATCHSIZE, num_workers=cfg.DATA.NUM_WORKERS,
            shuffle=True, drop_last=True, pin_memory=False)
        if 0.0 < cfg.DATA.MIXUP < 1.0:
            train_loader = MixUpWrapper(
                cfg.DATA.MIXUP, num_classes=cfg.NETWORK.NUM_CLASSES, dataloader=train_loader, device=self.device)
        valid_loader = None
        if valid_set is not None:
            valid_loader = torch.utils.data.DataLoader(
                valid_set, batch_size=cfg.DATA.BATCHSIZE, num_workers=cfg.DATA.NUM_WORKERS,
                shuffle=False, drop_last=False, pin_memory=False)
        return train_loader, valid_loader

    def _build_dataset(self, cfg, train_csv, valid_csv=None):
        train_trans, valid_trans = self._build_trans(cfg=cfg)
        # train_set = CallSmoke(, transforms=train_trans)
        train_set = CsvDataset(train_csv, root_dir=cfg.DATA.DATASETS.ROOT_DIR, transform=train_trans)
        valid_set = None
        if valid_csv is not None and valid_csv != "":
            valid_set = CsvDataset(valid_csv, cfg.DATA.DATASETS.ROOT_DIR, transform=valid_trans)
        return train_set, valid_set

    def _build_trans(self, cfg):
        train_trans = tvtrans.Compose(
            [
                # tvtrans.RandomRotation((0, 10), Image.BICUBIC, expand=True),
                # mytrans.RandomBrightness(p=0.733, min_factor=0.666, max_factor=1.2222),
                # mytrans.RandomNoise(p=0.5),
                # mytrans.RandomBlur(p=0.5),
                #  mytrans.RandomCrop(p=0.7, scale=(0.8666, 1)),
                # tvtrans.Resize(size=cfg.DATA.SIZE),
                # mytrans.ResizeFill(size=cfg.DATA.SIZE),
                mytrans.ERandomCrop(imgsize=cfg.DATA.TRAIN_SIZE[0]),
                tvtrans.RandomHorizontalFlip(),
                tvtrans.ToTensor(),
                mytrans.Lighting(0.1, mytrans._imagenet_pca['eigval'], mytrans._imagenet_pca['eigvec']),
                tvtrans.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
            ])
        valid_trans = tvtrans.Compose(
            [
                # tvtrans.Resize(size=cfg.DATA.SIZE),
                # mytrans.ResizeFill(size=cfg.DATA.SIZE),
                mytrans.ECenterCrop(imgsize=cfg.DATA.VALID_SIZE[0]),
                tvtrans.ToTensor(),
                tvtrans.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
            ])
        return train_trans, valid_trans

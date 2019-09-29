# -*- coding:utf-8 -*-
###
# File: ensemble_model.py
# Created Date: Tuesday, September 17th 2019, 7:47:54 pm
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
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import classifi_model as cmodel
import csv_dataset as csvdset
import config
import time


class EnsembleNet(nn.Module):
    def __init__(self, num_model, opt):
        super(EnsembleNet, self).__init__()
        self.nets = nn.ModuleList()
        for i in range(num_model):
            self.nets.append(
                cmodel.ClassiModel(
                    arch=opt.arch, gpus=[opt.gpu],
                    optimv=opt.optimizer, num_classes=opt.num_classes, lr=opt.lr_list[0],
                    weight_decay=opt.weight_decay, from_pretrained=opt.from_pretrained).net)
        self.num_classes = opt.num_classes

    def forward(self, x):
        # x = x.cuda()
        votes = np.zeros((x.shape[0], self.num_classes), dtype=np.float32)
        for i in range(len(self.nets)):
            with torch.no_grad():
                out = self.nets[i](x)
                # out = out.cpu().argmax(dim=1)
                out = torch.nn.softmax(out, dim=1)  # batchsize*num_classes
            # for j in range(out.shape[0]):
                votes += out
        # final_result = np.argmax(votes, axis=1)
        final_result = votes
        return final_result


class EnsembleModel(object):
    def __init__(self, num_model, opt):
        super(EnsembleModel, self).__init__()
        self.network = EnsembleNet(num_model, opt)
        if opt.gpu >= 0:
            self.device = torch.device('cuda:%d' % opt.gpu)
        else:
            self.device = torch.device('cpu')

    def test_set(self, testcsv, testroot, transform=None, extension='', bs=64, workers=32):
        self._eval()
        dataset = csvdset.CsvDatasetTest(testcsv, testroot, transform, extension)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, shuffle=False, num_workers=workers, drop_last=False, pin_memory=False)
        ims_list = []
        prelabels = []
        for i, data in enumerate(dataloader, start=0):
            images, labels, ims_name = data
            images = images.to(self.device)
            st = time.time()
            pre_label = self.test(images)
            pre_label = pre_label.argmax(1)
            et1 = time.time()
            for j in range(len(pre_label)):
                ims_list.append(ims_name[j])
                prelabels.append(pre_label[j])
            et2 = time.time()
            print("one batch use time: %f:%f, batch size is: %d" % ((et2-st), (et1-st), pre_label.shape[0]))
        return ims_list, prelabels

    def test(self, x):
        return self.network(x)

    def _eval(self):
        for i in range(len(self.network.nets)):
            self.network.nets[i].eval()

    def savemodel(self, model_name):
        torch.save(self.network.state_dict(), model_name)

    def loadmodel(self, model_url):
        if (os.path.exists(model_url)):
            print("loading model........")
            self.network.load_state_dict(torch.load(model_url))
            print("model loaded successfully!")
        else:
            print("model url is not exist")

    def load_from_separate(self, root, num_model):
        for i in range(num_model):
            filename = "%d/model_valid_best.pth" % i
            model_url = os.path.join(root, filename)
            self.network.nets[i].load_state_dict(torch.load(model_url))


if __name__ == "__main__":
    conf = config.Config()
    opt = conf.create_opt()
    opt.arch = 'efficientnet-b2'
    pmodel = EnsembleModel(opt.fold_num, opt)
    model_root = "fold-out-10-b2-seq/"
    pmodel.load_from_separate(model_root, opt.fold_num)
    pmodel.savemodel(os.path.join(model_root, "model_ensemble-10-b2-seq-vb.pth"))
    pmodel.loadmodel(os.path.join(model_root, "model_ensemble-10-b2-seq-vb.pth"))

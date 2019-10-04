# -*- coding:utf-8 -*-
###
# File: weath_recog_model.py
# Created Date: Tuesday, September 10th 2019, 9:56:15 pm
# Author: yusnows
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2019 nju-visg
#
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
import PIL
from efficientnet_pytorch import EfficientNet
import resnet_cbam as resnet
from csv_dataset import CsvDataset
import time


class ClassiModel(object):
    def __init__(
            self, arch='efficientnet-b7', gpus=[0], optimv='sgd',
            num_classes=10, multi_labels=False, lr=0.1, momentum=0.9,
            weight_decay=1e-4, from_pretrained=False, ifcbam=False):
        super(ClassiModel, self).__init__()
        cudnn.benchmark = True
        self.ifcbam = ifcbam
        self.device = self._determine_device(gpus)
        self.net = self._create_net(arch, num_classes, from_pretrained)
        self.optimizer = self._create_optimizer(optimv, lr, momentum, weight_decay)
        self.multi_labels = multi_labels
        if not multi_labels:
            print("single label classify, use CrossEntropy for loss")
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            print("multi labels classify, use BCEWithLogits for loss")
            self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.softmax_threshold = 0.3
        self.sigmod_threshold = 0.5  # maybe sigmod is more reasonable

    def get_multi_labels(self, outs, method='sigmod'):
        if method == 'sigmod':
            outs = torch.sigmoid(outs, dim=1)
            threshold = self.sigmod_threshold
        elif method == 'softmax':
            outs = torch.softmax(outs, dim=1)
            threshold = self.softmax_threshold
        outs[outs < threshold] = 0
        outs[outs >= threshold] = 1
        return outs

    def train_fold(self, trainloader, validloader, fold, opt):
        start = time.time()
        if opt.model_url is not None:
            self.loadmodel(opt.model_url, ifload_fc=opt.load_fc)
        epoch = 0
        train_acc_list, valid_acc_list = [], []
        train_score_list, valid_score_list = [], []
        valid_acc, valid_score = 0, 0
        train_acc, train_score = 0, 0
        train_acc_std = 0.801
        for lr in opt.lr_list:
            self._set_learning_rate(lr)
            train_acc_std = train_acc_std + 0.02
            print('set lr to %.6f' % lr)
            # 每次调整学习率后，重新计算当前有多少个epoch准确率未上升
            patience = 0
            while True:
                st = time.time()
                # 训练一个完整的epoch
                for i, data in enumerate(trainloader, start=0):
                    img, gt = data
                    loss = self.train(img, gt)
                    print(epoch, i, loss, end='\r')
                print('')
                train_acc, train_score = self.validate_fold(trainloader, opt)
                if validloader is not None:
                    valid_acc, valid_score = self.validate_fold(validloader, opt)
                self.savemodel_name(os.path.join(opt.fold_out, "%d/model_last.pth" % fold))
                # 这里可以控制用acc还是score来作为判断标
                # if len(train_acc_list) == 0 or train_acc > max(train_acc_list):
                if len(train_score_list) == 0 or train_score > max(train_score_list):
                    self.savemodel_name(os.path.join(opt.fold_out, "%d/model_train_best.pth" % fold))
                if len(valid_score_list) == 0 or valid_score > max(valid_score_list):
                    self.savemodel_name(os.path.join(opt.fold_out, "%d/model_valid_best.pth" % fold))
                # 计算已经连续多少个epoch训练集的准确率没有上升了
                # if len(train_acc_list) == 0 or train_acc > max(train_acc_list):
                if len(train_score_list) == 0 or train_score > max(train_score_list):
                    patience = 0
                else:
                    patience += 1
                epoch += 1
                train_acc_list.append(train_acc)
                valid_acc_list.append(valid_acc)
                train_score_list.append(train_score)
                valid_score_list.append(valid_score)
                et = time.time()
                print('train acc:', train_acc, 'valid acc:', valid_acc, '%ds' % int(et-st))
                print('train score:', train_score, 'valid score:', valid_score, '%ds' % int(et-st))
                print("patience is: %d, train_acc_std is: %f" % (patience, train_acc_std))
                if ((patience >= opt.max_N) and (train_acc >= train_acc_std)) or (epoch >= opt.epoches):
                    break
        max_N = opt.max_N
        avg_valid_acc = sum(valid_acc_list[-max_N:]) / max_N
        avg_valid_score = sum(valid_score_list[-max_N:]) / max_N
        print('average valid accuracy in last %d epochs:' % max_N, avg_valid_acc)
        print('average valid score in last %d epochs:' % max_N, avg_valid_score)
        end = time.time()
        print('time:', '%ds' % int(end - start))
        return avg_valid_acc, avg_valid_score

    def validate_fold(self, validloader, opt):
        num_correct = 0
        N = 0
        # 统计TP, FP, FN
        TP = np.zeros(opt.num_classes)
        FP = np.zeros(opt.num_classes)
        FN = np.zeros(opt.num_classes)
        for i, data in enumerate(validloader, start=0):
            print(i, end='\r')
            images, labels = data
            labels = labels.to(self.device)
            out = self.test(images)
            if self.multi_labels is False:
                labels = labels.argmax(dim=1)
                pred_label = torch.argmax(out, dim=1)
            else:
                pred_label = self.get_multi_labels(outs=out)
            N += labels.shape[0]
            for i in range(labels.shape[0]):
                # if multi labels, labels[i] and pred_label is 1x(num_classes) tensor, else, they are a scalar
                if labels[i] == pred_label[i]:
                    TP[labels[i]] += 1
                    num_correct += 1
                else:
                    FN[labels[i]] += 1
                    FP[pred_label[i]] += 1
        acc = num_correct / N
        # 计算precision和recall
        precision, recall = [], []
        for i in range(len(TP)):
            if TP[i] + FP[i] == 0:
                precision.append(0)
            else:
                precision.append(TP[i] / (TP[i] + FP[i]))
            if TP[i] + FN[i] == 0:
                recall.append(0)
            else:
                recall.append(TP[i] / (TP[i] + FN[i]))
        # 计算F1-score
        f1 = []
        for i in range(len(TP)):
            if precision[i] + recall[i] == 0:
                f1.append(0)
            else:
                f1.append((2*precision[i]*recall[i]) / (precision[i]+recall[i]))
        score = sum(f1) / len(f1)
        return acc, score

    def train(self, ims, labels):
        self.net.train()
        ims = ims.to(self.device)
        labels = labels.to(self.device)
        output = self.net(ims)  # (bs, num_classes) before sigmod
        if self.multi_labels is False:  # single label classification
            labels = labels.argmax(dim=1)
        loss = self.criterion(output, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, ims, labels):
        self.net.eval()
        with torch.no_grad():
            ims = ims.to(self.device)
            labels = labels.to(self.device)
            output = self.net(ims)
            loss = self.criterion(output, labels)
        return loss.item()

    def test(self, ims):
        self.net.eval()
        with torch.no_grad():
            ims = ims.to(self.device)
            output = self.net(ims)
        return output

    def savemodel_name(self, name):
        path, _ = os.path.split(name)
        if not (os.path.exists(path)):
            os.makedirs(path, exist_ok=True)
#         model_name = path + "/weather_model_%05d.pth" % numt
        torch.save(self.net.state_dict(), name)
        print("save mdoel {} successfully".format(name))

    def savemodel(self, path, numt=0):
        if not (os.path.exists(path)):
            os.makedirs(path, exist_ok=True)
        model_name = path + "/classi_model_%05d.pth" % numt
        torch.save(self.net.state_dict(), model_name)
        print("save mdoel {} successfully".format(model_name))

    def loadmodel(self, model_url, ifload_fc=False):
        if (os.path.exists(model_url)):
            print("loading model........")
            if ifload_fc:
                self.net.load_state_dict(torch.load(model_url))
            else:
                state_dict = torch.load(model_url)
                state_dict.pop('_fc.weight')
                state_dict.pop('_fc.bias')
                res = self.net.load_state_dict(state_dict, strict=False)
                assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
            print("model {} loaded successfully!".format(model_url))
        else:
            print("model url:{} is not exist".format(model_url))

    def adjust_learning_rate(self, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = lr * (0.1 ** (epoch // 30))
        print("adjust the learning rate, lr now is: %f" % lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _set_learning_rate(self, lr):
        print("set learning rate to %f" % lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _determine_device(self, gpus):
        if len(gpus) == 0:
            print("use cpu device")
            device = torch.device('cpu')
        else:
            if gpus[0] >= 0:
                print("use gpu device: %d" % gpus[0])
                device = torch.device('cuda:%d' % gpus[0])
            else:
                print("use cpu device")
                device = torch.device('cpu')
        return device

    def _create_net(self, arch, num_classes, from_pretrained):
        if 'efficientnet' in arch:
            if from_pretrained:
                print("create efficient net from pretrained model")
                net = EfficientNet.from_pretrained(arch, num_classes=num_classes, ifcbam=self.ifcbam)
            else:
                print("create efficient net from name")
                net = EfficientNet.from_name(arch, override_params={'num_classes': num_classes}, ifcbam=self.ifcbam)
        elif 'resnext101_32x8d' == arch:
            net = resnet.resnext101_32x8d_wsl(pretrained=from_pretrained, num_classes=num_classes)
        elif 'resnext101_32x16d' == arch:
            net = resnet.resnext101_32x16d_wsl(pretrained=from_pretrained, num_classes=num_classes)
        elif 'resnext101_32x32d' == arch:
            net = resnet.resnext101_32x32d_wsl(pretrained=from_pretrained, num_classes=num_classes)
        elif 'resnext101_32x48d' == arch:
            net = resnet.resnext101_32x48d_wsl(pretrained=from_pretrained, num_classes=num_classes)
        return net.to(self.device)

    def _create_optimizer(self, optimv, lr, momentum, weight_decay):
        print("use the %s optimizer" % optimv)
        if optimv == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
        elif optimv == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimv == 'rmsprop':
            optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer


if __name__ == "__main__":
    model = ClassiModel()
    print(model.net)

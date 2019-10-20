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
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
import numpy as np
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import PIL
from efficientnet_pytorch import EfficientNet
import resnet_cbam as resnet
from csv_dataset import CsvDataset
import network_arch
import time


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class ClassiModel(object):
    def __init__(
            self, arch='efficientnet-b7', gpus=[0], optimv='sgd', num_classes=10,
            multi_labels=False, lr=0.1, momentum=0.9, weight_decay=1e-4, from_pretrained=False,
            ifcbam=False, fix_bn_v=False, criterion_v='CrossEntropyLoss'):
        super(ClassiModel, self).__init__()
        cudnn.benchmark = True
        # network architecture options
        self.precision = 'FP32'
        self.arch = arch
        self.num_classes = num_classes
        self.from_pretrained = from_pretrained
        self.multi_labels = multi_labels
        self.optimv = optimv
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # trick options
        self.ifcbam = ifcbam
        self.fix_bn_v = fix_bn_v
        self.criterion_v = criterion_v

        self.device = self._determine_device(gpus)
        self.net = self._create_net(arch, num_classes, from_pretrained)
        if self.fix_bn_v:
            self.freeze_bn()
            # because the freeze_bn function will create optimizer to make sure bn parameters will
            # not be updated, no need to create optimizer again after freeze_bn.
        else:
            self.optimizer = self._create_optimizer(optimv, lr, momentum, weight_decay)
        self.criterion = self._create_criterion(self.multi_labels, self.criterion_v)
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
        # 保存opt, 便于复现实验结果
        opt_name = os.path.join(opt.model_save_dir, "%d/opt.pth" % fold)
        os.makedirs(os.path.join(opt.model_save_dir, "%d" % fold), exist_ok=True)
        with open(opt_name, 'w') as f:
            json.dump(opt.__dict__, f)

        if opt.model_url is not None:
            self.loadmodel(opt.model_url, ifload_fc=opt.load_fc)
        epoch = 0
        train_acc_list, valid_acc_list = [], []
        train_score_list, valid_score_list = [], []
        valid_acc, valid_score = 0, 0
        train_acc, train_score = 0, 0
        train_acc_std = 0
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
                self.savemodel_name(os.path.join(opt.model_save_dir, "%d/model_last.pth" % fold))
                # 这里可以控制用acc还是score来作为判断标
                # if len(train_acc_list) == 0 or train_acc > max(train_acc_list):
                if len(train_score_list) == 0 or train_score > max(train_score_list):
                    self.savemodel_name(os.path.join(opt.model_save_dir, "%d/model_train_best.pth" % fold))
                if len(valid_score_list) == 0 or valid_score > max(valid_score_list):
                    self.savemodel_name(os.path.join(opt.model_save_dir, "%d/model_valid_best.pth" % fold))
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
        avg_train_acc = sum(train_acc_list[-max_N:])/max_N
        avg_train_score = sum(train_score_list[-max_N:])/max_N
        print('average valid accuracy in last %d epochs:' % max_N, avg_valid_acc)
        print('average valid score in last %d epochs:' % max_N, avg_valid_score)
        end = time.time()
        print('time:', '%ds' % int(end - start))
        return avg_valid_acc, avg_valid_score, avg_train_acc, avg_train_score

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
        # model_name = path + "/weather_model_%05d.pth" % numt
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
                if 'efficientnet' in self.arch:
                    state_dict = torch.load(model_url)
                    state_dict.pop('_fc.weight')
                    state_dict.pop('_fc.bias')
                    res = self.net.load_state_dict(state_dict, strict=False)
                    assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
                else:
                    state_dict = torch.load(model_url)
                    state_dict.pop('fc.weight')
                    state_dict.pop('fc.bias')
                    res = self.net.load_state_dict(state_dict, strict=False)
                    assert str(res.missing_keys) == str(['fc.weight', 'fc.bias']), 'issue loading pretrained weights'
            print("model {} loaded successfully!".format(model_url))
        else:
            print("model url:{} is not exist".format(model_url))

    def adjust_learning_rate(self, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = lr * (0.1 ** (epoch // 30))
        print("adjust the learning rate, lr now is: %f" % lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def freeze_bn(self):
        if self.from_pretrained is False:
            print("you can not freeze the bn layer parameters, because the network is not come from the pretrained model")
            return
        print("freezing the network's bn layer(s) parameters......")
        cnt = 0
        for name, m in self.net.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                cnt += 1
        print("freezed the network's %d bn layer(s) parameters" % cnt)
        # make sure the optimizer will not update the freezed parameters
        if hasattr(self, 'optimizer') and isinstance(self.optimizer, optim.Optimizer):
            for i in range(len(self.optimizer.param_grups)):
                del self.optimizer.param_grups[0]
            self.optimizer.add_param_group({'params': filter(lambda p: p.requires_grad, self.net.parameters())})
        else:
            self.optimizer = self._create_optimizer(self.optimv, self.lr, self.momentum, self.weight_decay)
        return

    def unfreeze_bn(self):
        cnt = 0
        print("unfreezing the network's bn layer(s) parameters......")
        for name, m in self.net.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                cnt += 1
        print("unfreezed the network's %d bn layer(s) parameters" % cnt)
        # make sure the optimizer will update the unfreezed parameters
        if hasattr(self, 'optimizer') and isinstance(self.optimizer, optim.Optimizer):
            for i in range(len(self.optimizer.param_grups)):
                del self.optimizer.param_grups[0]
            self.optimizer.add_param_group({'params': filter(lambda p: p.requires_grad, self.net.parameters())})
        else:
            self.optimizer = self._create_optimizer(self.optimv, self.lr, self.momentum, self.weight_decay)
        return

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
        elif arch == 'resnext101_32x8d_wsl':
            net = resnet.resnext101_32x8d_wsl(pretrained=from_pretrained, num_classes=num_classes, ifcbam=self.ifcbam)
        elif arch == 'resnext101_32x16d_wsl':
            net = resnet.resnext101_32x16d_wsl(pretrained=from_pretrained, num_classes=num_classes, ifcbam=self.ifcbam)
        elif arch == 'resnext101_32x32d_wsl':
            net = resnet.resnext101_32x32d_wsl(pretrained=from_pretrained, num_classes=num_classes, ifcbam=self.ifcbam)
        elif arch == 'resnext101_32x48d_wsl':
            net = resnet.resnext101_32x48d_wsl(pretrained=from_pretrained, num_classes=num_classes, ifcbam=self.ifcbam)
        elif arch == 'resnet18':
            net = resnet.resnet18(pretrained=from_pretrained, num_classes=num_classes, ifcbam=self.ifcbam)
        elif arch == 'resnet34':
            net = resnet.resnet34(pretrained=from_pretrained, num_classes=num_classes, ifcbam=self.ifcbam)
        elif arch == 'resnet50':
            net = resnet.resnet50(pretrained=from_pretrained, num_classes=num_classes, ifcbam=self.ifcbam)
        elif arch == 'resnet101':
            net = resnet.resnet101(pretrained=from_pretrained, num_classes=num_classes, ifcbam=self.ifcbam)
        elif arch == 'resnet152':
            net = resnet.resnet152(pretrained=from_pretrained, num_classes=num_classes, ifcbam=self.ifcbam)
        elif arch == 'resnext50_32x4d':
            net = resnet.resnext50_32x4d(pretrained=from_pretrained, num_classes=num_classes, ifcbam=self.ifcbam)
        elif arch == 'resnext101_32x8d':
            net = resnet.resnext101_32x8d(pretrained=from_pretrained, num_classes=num_classes, ifcbam=self.ifcbam)
        elif arch == 'alexnet':
            net = network_arch.alexnet(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'vgg11':
            net = network_arch.vgg11(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'vgg11_bn':
            net = network_arch.vgg11_bn(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'vgg13':
            net = network_arch.vgg13(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'vgg13_bn':
            net = network_arch.vgg13_bn(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'vgg16':
            net = network_arch.vgg16(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'vgg16_bn':
            net = network_arch.vgg16_bn(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'vgg19':
            net = network_arch.vgg19(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'vgg19_bn':
            net = network_arch.vgg19_bn(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'inception_v3':
            net = network_arch.inception_v3(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'shufflenet_v2_x0_5':
            net = network_arch.shufflenet_v2_x0_5(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'shufflenet_v2_x1_0':
            net = network_arch.shufflenet_v2_x1_0(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'shufflenet_v2_x1_5':
            net = network_arch.shufflenet_v2_x1_5(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'shufflenet_v2_x2_0':
            net = network_arch.shufflenet_v2_x2_0(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'mobilenet_v2':
            net = network_arch.mobilenet_v2(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'googlenet':
            net = network_arch.googlenet(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'densenet121':
            net = network_arch.densenet121(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'densenet161':
            net = network_arch.densenet161(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'densenet169':
            net = network_arch.densenet169(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'densnet201':
            net = network_arch.densenet201(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'squeezenet1_0':
            net = network_arch.squeezenet1_0(pretrained=from_pretrained, num_classes=num_classes)
        elif arch == 'squeezenet1_1':
            net = network_arch.squeezenet1_1(pretrained=from_pretrained, num_classes=num_classes)
        else:
            print("not suitable architecture, please check you arch parameter")
            exit()
        return net.to(self.device)

    def _create_optimizer(self, optimv, lr, momentum, weight_decay):
        print("use the %s optimizer" % optimv)
        if optimv == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()),
                                  lr, momentum=momentum, weight_decay=weight_decay)
        elif optimv == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=lr, weight_decay=weight_decay)
        elif optimv == 'rmsprop':
            optimizer = optim.RMSprop(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=lr, weight_decay=weight_decay)
        return optimizer

    def _create_criterion(self, ifmulti_labels, criterion_v):
        if not ifmulti_labels:
            print("single label classify")
            if criterion_v == "CrossEntropyLoss":
                print("use CrossEntropy for loss")
                criterion = nn.CrossEntropyLoss().to(self.device)
            elif criterion_v == "CrossEntropyLabelSmooth":
                print("use CrossEntropyLabelSmooth for loss")
                criterion = CrossEntropyLabelSmooth(self.num_classes, smoothing=0.1)
        else:
            print("multi labels classify, use BCEWithLogits for loss")
            criterion = nn.BCEWithLogitsLoss().to(self.device)
        return criterion


if __name__ == "__main__":
    model = ClassiModel()
    print(model.net)

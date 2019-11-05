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
from apex import amp
import numpy as np
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import PIL
from efficientnet_pytorch import EfficientNet
import resnet_cbam as resnet
from csv_dataset import CsvDataset
import network_arch
import loss_trick
import time


class ClassiModel(object):
    def __init__(
            self, arch='efficientnet-b7', gpus=[0], optimv='sgd', num_classes=10,
            multi_labels=False, regress_threshold=False, lr=0.1, momentum=0.9, weight_decay=1e-4,
            from_pretrained=False, ifcbam=False, fix_bn_v=False, criterion_v='CrossEntropyLoss',
            amp_train=False, amp_opt_level='O2'):
        super(ClassiModel, self).__init__()
        cudnn.benchmark = True
        # network architecture options
        self.amp_train = amp_train
        self.amp_opt_level = amp_opt_level
        self.arch = arch
        self.num_classes = num_classes
        self.multi_labels = multi_labels
        # if multi_label is False, regress_threshold should not be true
        regress_threshold = regress_threshold and multi_labels
        self.regress_threshold = regress_threshold
        self.from_pretrained = from_pretrained
        self.optimv = optimv
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # trick options
        self.ifcbam = ifcbam
        self.fix_bn_v = fix_bn_v
        self.criterion_v = criterion_v

        self.device = self._determine_device(gpus)
        self.threshold_crit = None
        if regress_threshold:
            print("regress multi labels threshold")
            num_classes += 1  # 1 is use for threshold
            self.threshold_crit = nn.SmoothL1Loss()
        self.net = self._create_net(arch, num_classes, from_pretrained)
        if self.fix_bn_v:
            self.freeze_bn()
            # because the freeze_bn function will create optimizer to make sure bn parameters will
            # not be updated, no need to create optimizer again after freeze_bn.
        else:
            self.optimizer = self._create_optimizer(optimv, lr, momentum, weight_decay)
        if self.amp_train:
            if (self.amp_opt_level is None) or (self.amp_opt_level not in ['O0', 'O1', 'O2', 'O3']):
                self.amp_opt_level = 'O1'
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level=self.amp_opt_level)
        self.criterion = self._create_criterion(self.multi_labels, self.criterion_v)
        self.softmax_threshold = 0.3
        self.sigmod_threshold = 0.45  # maybe sigmod is more reasonable

    def get_multi_labels(self, outs, threshold=None, method='sigmoid'):
        """
        this function should call only when test or validate.
        """
        if method is None or method == 'NULL':
            if threshold is None:
                threshold = self.sigmod_threshold
        elif method == 'sigmoid':
            outs = torch.sigmoid(outs)
            if threshold is None:
                threshold = self.sigmod_threshold
            else:
                threshold = torch.sigmoid(threshold)
        elif method == 'softmax':
            outs = torch.softmax(outs, dim=1)
            if threshold is None:
                threshold = self.softmax_threshold
            else:
                threshold = torch.sigmoid(threshold)
        pred_label = torch.zeros_like(outs, device=self.device)
        pred_label[torch.arange(0, outs.shape[0]), outs.argmax(dim=1)] = 1
        # outs[outs < threshold] = 0
        pred_label[outs >= threshold] = 1
        return pred_label

    def train_fold(self, trainloader, validloader, fold, opt):
        start = time.time()
        # 保存opt, 便于复现实验结果
        opt_name = os.path.join(opt.model_save_dir, "%d/opt.json" % fold)
        os.makedirs(os.path.join(opt.model_save_dir, "%d" % fold), exist_ok=True)
        with open(opt_name, 'w') as f:
            json.dump(opt.__dict__, f)

        if opt.model_url is not None:
            self.loadmodel(opt.model_url, ifload_fc=opt.load_fc)
        self.freeze_layers(0, 23)
        layer_freeze_flag = True
        epoch = 0
        train_acc_list, valid_acc_list = [], []
        train_score_list, valid_score_list = [], []
        valid_acc, valid_score = 0, 0
        train_acc, train_score = 0, 0
        train_acc_std = 0
        for lr in opt.lr_list:
            self._set_learning_rate(lr)
            if lr <= opt.lr_list[0] / 10:
                self.unfreeze_bn()
            train_acc_std = train_acc_std + 0.02
            print('set lr to %.6f' % lr)
            # 每次调整学习率后，重新计算当前有多少个epoch准确率未上升
            patience = 0
            while True:
                st = time.time()
                if epoch > 3 and layer_freeze_flag:
                    self.unfreeze_layers(0, 23)
                    layer_freeze_flag = False
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
                self.savemodels(
                    opt.model_save_dir, fold, train_acc=train_acc, train_acc_list=train_acc_list,
                    train_score=train_score, train_score_list=train_score_list, valid_acc=valid_acc,
                    valid_acc_list=valid_acc_list, valid_score=valid_score, valid_score_list=valid_score_list)
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
        avg_train_score = sum(train_score_list[-max_N:]) / max_N
        print('average valid accuracy in last %d epochs:' % max_N, avg_valid_acc)
        print('average valid score in last %d epochs:' % max_N, avg_valid_score)
        self._save_recored(os.path.join(opt.model_save_dir, "%d" % (fold)), train_acc_list,
                           train_score_list, valid_acc_list, valid_score_list)
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
            images, labels = data
            labels = labels.to(self.device)
            out = self.test(images)
            if self.multi_labels is False:
                labels = labels.argmax(dim=1)
                pred_label = torch.argmax(out, dim=1)
                print(i, end='\r')
            elif self.regress_threshold:
                threshold = out[:, self.num_classes].view(-1, 1)  # [bs,1]
                out = out[:, 0:self.num_classes]
                pred_label = self.get_multi_labels(out, threshold)
                print(
                    i, (np.where((labels[0] == 1).cpu())[0]).shape,
                    (np.where((pred_label[0] == 1).cpu())[0]).shape,
                    threshold[0].sigmoid().cpu().item(),
                    end='\r')
            else:
                pred_label = self.get_multi_labels(out)
                print(
                    i, (np.where((labels[0] == 1).cpu())[0]).shape,
                    (np.where((pred_label[0] == 1).cpu())[0]).shape,
                    end='\r')
            # print(i, labels, pred_label, end='\r')
            N += labels.shape[0]
            for i in range(labels.shape[0]):
                # if multi labels, labels[i] and pred_label is 1x(num_classes) tensor, else, they are a scalar
                if self.multi_labels:
                    indices = np.where((labels[i] == 1).cpu())
                    if (labels[i] == pred_label[i]).sum() == labels.shape[1]:
                        TP[indices] += 1
                        num_correct += 1
                    else:
                        FN[indices] += 1
                        FP[indices] += 1
                else:
                    if labels[i] == pred_label[i]:
                        TP[labels[i]] += 1
                        num_correct += 1
                    else:
                        FN[labels[i]] += 1
                        FP[labels[i]] += 1
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
        loss_threshold = 0
        if self.multi_labels is False:  # single label classification
            labels = labels.argmax(dim=1)
        elif self.regress_threshold:  # multi label classification and regress_threshold,
            # because loss criterion is with logit, don't need to sigmoid the output when training
            # output is [bs, (num_classes+1)]
            loss_threshold = self.threshold_loss(output, labels)
            output = output[:, 0:self.num_classes]

        loss = self.criterion(output, labels) + loss_threshold
        self.optimizer.zero_grad()
        if self.amp_train:
            with amp.scale_loss(loss, self.optimizer) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()
        return loss.item()

    def threshold_loss(self, outs, gt, method='sigmoid'):
        if self.regress_threshold is False:
            print("regress_threshold should be true, but it is False. please check your config")
            exit()
        outs = outs.detach()
        gt = gt.detach()
        if method is None or method == "NULL":
            threshold = outs[:, -1].view(-1, 1)
            outs = outs[:, 0:-1]
        elif method == 'sigmoid':
            outs = torch.sigmoid(outs)
            threshold = outs[:, self.num_classes].view(-1, 1)
            outs = outs[:, 0:self.num_classes]
        elif method == 'softmax':
            threshold = outs[:, self.num_classes].view(-1, 1)
            outs = outs[:, 0:self.num_classes]
            outs = outs.softmax(dim=1)
            threshold = threshold.sigmoid()
        gt_sum = gt.sum(1).cpu().numpy().astype(np.int)
        topk_v = []
        for bs_id in range(outs.shape[0]):
            topk_tmp = (outs[bs_id].topk(int(gt_sum[bs_id])))
            topk_v.append(topk_tmp[0][-1].view(1, 1))
        topk_v = torch.cat(topk_v, dim=0).detach()
        topk_v = topk_v - topk_v * 0.05
        # print(topk.shape, threshold.shape), exit(0)
        loss = self.threshold_crit(threshold, topk_v)
        # loss = 0
        return loss

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

    def savemodels(self, model_save_dir, fold, train_acc=None, train_acc_list=None, train_score=None,
                   train_score_list=None, valid_acc=None, valid_acc_list=None, valid_score=None, valid_score_list=None):
        if ((train_acc is not None) and (train_acc_list is not None)
                and (len(train_acc_list) == 0 or train_acc > max(train_acc_list))):
            # os.system("rm " + os.path.join(model_save_dir, "%d/model_train_accur_*.pth" % fold))
            self.savemodel_name(
                os.path.join(model_save_dir, "%d/model_train_accur_best.pth" % (fold)))
        if ((train_score is not None) and (train_score_list is not None)
                and (len(train_score_list) == 0 or train_score > max(train_score_list))):
            # os.system("rm " + os.path.join(model_save_dir, "%d/model_train_score_*.pth" % fold))
            self.savemodel_name(
                os.path.join(model_save_dir, "%d/model_train_score_best.pth" % (fold)))
        if ((valid_score is not None) and (valid_score_list is not None)
                and (len(valid_score_list) == 0 or valid_score > max(valid_score_list))):
            # os.system("rm " + os.path.join(model_save_dir, "%d/model_valid_score_*.pth" % fold))
            self.savemodel_name(
                os.path.join(model_save_dir, "%d/model_valid_score_best.pth" % (fold)))
        if ((valid_acc is not None) and (valid_acc_list is not None)
                and (len(valid_acc_list) == 0 or valid_acc > max(valid_acc_list))):
            # os.system("rm " + os.path.join(model_save_dir, "%d/model_valid_accur_*.pth" % fold))
            self.savemodel_name(
                os.path.join(model_save_dir, "%d/model_valid_accur_best.pth" % (fold)))

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
        if hasattr(self.net, 'ifcbam'):
            ifcbam = self.net.ifcbam
        else:
            ifcbam = False
        if ifcbam:
            cbam_module_names = self.net.cbam_module_names
        else:
            cbam_module_names = []
        cnt = 0
        for name, m in self.net.named_modules():
            if isinstance(m, nn.BatchNorm2d) and (name not in cbam_module_names):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                cnt += 1
        print("freezed the network's %d bn layer(s) parameters" % cnt)
        # make sure the optimizer will not update the freezed parameters
        if hasattr(self, 'optimizer') and isinstance(self.optimizer, optim.Optimizer):
            for i in range(len(self.optimizer.param_groups)):
                del self.optimizer.param_groups[0]
            self.optimizer.add_param_group({'params': filter(lambda p: p.requires_grad, self.net.parameters())})
        else:
            self.optimizer = self._create_optimizer(self.optimv, self.lr, self.momentum, self.weight_decay)
        return

    def unfreeze_bn(self):
        cnt = 0
        print("unfreezing the network's bn layer(s) parameters......")
        if hasattr(self.net, 'ifcbam'):
            ifcbam = self.net.ifcbam
        else:
            ifcbam = False
        if ifcbam:
            cbam_module_names = self.net.cbam_module_names
        else:
            cbam_module_names = []
        for name, m in self.net.named_modules():
            if isinstance(m, nn.BatchNorm2d) and (name not in cbam_module_names):
                m.train()
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                cnt += 1
        print("unfreezed the network's %d bn layer(s) parameters" % cnt)
        # make sure the optimizer will update the unfreezed parameters
        if hasattr(self, 'optimizer') and isinstance(self.optimizer, optim.Optimizer):
            for i in range(len(self.optimizer.param_groups)):
                del self.optimizer.param_groups[0]
            self.optimizer.add_param_group({'params': filter(lambda p: p.requires_grad, self.net.parameters())})
        else:
            self.optimizer = self._create_optimizer(self.optimv, self.lr, self.momentum, self.weight_decay)
        return

    def freeze_layers(self, layer_begin, layer_end):
        if 'efficientnet' in self.arch:
            self.net.freeze_blocks(layer_begin, layer_end)
            # make sure the optimizer will not update the freezed parameters
            if hasattr(self, 'optimizer') and isinstance(self.optimizer, optim.Optimizer):
                for i in range(len(self.optimizer.param_groups)):
                    del self.optimizer.param_groups[0]
                self.optimizer.add_param_group({'params': filter(lambda p: p.requires_grad, self.net.parameters())})
            else:
                self.optimizer = self._create_optimizer(self.optimv, self.lr, self.momentum, self.weight_decay)
        else:
            print("the architecture: {} layer freeze is not implemented now".format(self.arch))
        return

    def unfreeze_layers(self, layer_begin, layer_end):
        if 'efficientnet' in self.arch:
            self.net.unfreeze_blocks(layer_begin, layer_end)
            # make sure the optimizer will update the unfreezed parameters
            if hasattr(self, 'optimizer') and isinstance(self.optimizer, optim.Optimizer):
                for i in range(len(self.optimizer.param_groups)):
                    del self.optimizer.param_groups[0]
                self.optimizer.add_param_group({'params': filter(lambda p: p.requires_grad, self.net.parameters())})
            else:
                self.optimizer = self._create_optimizer(self.optimv, self.lr, self.momentum, self.weight_decay)
        else:
            print("the architecture: {} layer unfreeze is not implemented now".format(self.arch))
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
                criterion = loss_trick.CrossEntropyLabelSmooth(self.num_classes, smoothing=0.1)
            elif criterion_v == 'FocalLoss':
                print("Use FocalLoss for loss")
                criterion = loss_trick.FocalLoss().to(self.device)
            else:
                print("use CrossEntropyLoss for loss")
                criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            if criterion_v == 'BCEWithLogitsLoss':
                print("multi labels classify, use BCEWithLogits for loss")
                criterion = nn.BCEWithLogitsLoss().to(self.device)
            elif criterion_v == 'BCELoss':
                print("multi labels classify, use BCE for loss")
                criterion = nn.BCELoss().to(self.device)
            else:
                print("multi labels classify, use BCEWithLogits for loss")
                criterion = nn.BCEWithLogitsLoss().to(self.device)
        return criterion

    def _save_recored(self, save_dir, train_acc_list, train_score_list, valid_acc_list, valid_score_list):
        max_valid_acc, max_valid_acc_index = max(valid_acc_list), valid_acc_list.index(max(valid_acc_list))
        max_valid_score, max_valid_score_index = max(valid_score_list), valid_score_list.index(max(valid_score_list))
        max_train_acc, max_train_acc_index = max(train_acc_list), train_acc_list.index(max(train_acc_list))
        max_train_score, max_train_score_index = max(train_score_list), train_score_list.index(max(train_score_list))
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "record.txt"), 'w') as f:
            f.write("epoch: %d, max_valid_accur: %.6f \n" % (max_valid_acc_index, max_valid_acc))
            f.write("epoch: %d, max_valid_score: %.6f \n" % (max_valid_score_index, max_valid_score))
            f.write("epoch: %d, max_train_accur: %.6f \n" % (max_train_acc_index, max_train_acc))
            f.write("epoch: %d, max_train_score: %.6f \n" % (max_train_score_index, max_train_score))


if __name__ == "__main__":
    model = ClassiModel()
    print(model.net)

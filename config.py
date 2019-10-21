# -*- coding:utf-8 -*-
###
# File: config.py
# Created Date: Saturday, September 14th 2019, 3:02:52 pm
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
import argparse


class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        config = argparse.ArgumentParser()
        # -------------------------------------------------------------------------
        # Generous options
        # -------------------------------------------------------------------------
        # network options
        config.add_argument('--arch',  default='efficientnet-b4', help='efficient net architecture')
        config.add_argument('--ifcbam', type=bool, default=False, help='if use cbam attention')
        config.add_argument('--from_pretrained', action='store_false', default=True,
                            help='if use the official pretrained model, default is True')
        config.add_argument(
            '--fix_bn', type=bool, default=True,
            help='if fix the network bn layer fpor training, if use the pretrained model, fix_bn may be good.')
        config.add_argument('--criterion_v', type=str, default='CrossEntropyLoss', help='the loss criterion')
        config.add_argument('--num_classes', type=int, default=4, help='the number of classes')
        config.add_argument('--multi_labels', action='store_true', default=False, help='multi label classify')
        config.add_argument('--regress_threshold', action='store_true', default=False,
                            help='if regress the multi label threshold')
        config.add_argument('--optimizer', type=str, default='sgd', help='the type of optimizer')
        config.add_argument('--lr_list',  nargs='+', type=float, default=[1e-3, 5e-4, 1e-4, 1e-5, 1e-6])
        config.add_argument('--momentum', type=float, default=0.9)
        config.add_argument('--weight_decay', type=float, default=1e-4)
        # dataset
        config.add_argument('--trainroot', type=str,
                            default='../data/train/', help='path to dataset')
        config.add_argument('--traincsv', type=str,
                            default='../data/train_labels.csv', help='path to dataset')
        config.add_argument('--testroot', type=str,
                            default='../data/data_merge/merge_new/test/', help='path to dataset')
        config.add_argument('--testcsv', type=str,
                            default='../data/data_merge/merge_new/test.csv', help='path to dataset')
        config.add_argument('--workers', type=int,  default=32, help='number of data loading workers')
        config.add_argument('--imageSize', type=int, default=224,
                            help='the height and width of the input image to network')
        config.add_argument('--batchSize', type=int, default=8, help='input batch size')
        config.add_argument('--gpu', type=int, default=0, help='gpu id')
        # fold integrate(ensemble)
        config.add_argument('--fold_need_gen', type=bool, default=False,
                            help='if need to gen new fold csv files or use the exist csv files')
        config.add_argument('--fold_num', type=int, default=10, help='the number of folds')
        config.add_argument('--fold_begin', type=int, default=0, help='fold num begin to train')
        config.add_argument('--fold_csv_dir', type=str, default='./datacsv/fold-csv/',
                            help='the dir for the temp fold file')
        # load pretrained model potions
        config.add_argument('--model_url', type=str, default=None, help='the pretrained model path to load')
        config.add_argument('--load_fc', action='store_true')
        # -------------------------------------------------------------------------
        # train options
        # -------------------------------------------------------------------------
        config.add_argument('--epoches', type=int, default=400, help='number of epochs to train for')
        config.add_argument('--epoch_base', type=int, default=1, help='number of epoch to begin')
        config.add_argument('--model_base_dir', type=str, default='./baseline_models/',
                            help='the directory to save model')
        config.add_argument(
            '--model_save_dir', type=str, default='./baseline_models/architecrture/fold_num',
            help='the directory to save model')  # Don't need to set this, this option will be change by program
        config.add_argument('--max_N', type=int, default=3)
        # -------------------------------------------------------------------------
        # extra options, for compasitive
        # -------------------------------------------------------------------------
        config.add_argument('--checkpoint_interval', type=int, default=20,
                            help='how many epoches interval to save model parameters')
        config.add_argument('--eval', action='store_true', default=False)
        self.config = config

    def create_opt(self):
        opt = self.config.parse_args()
        return opt

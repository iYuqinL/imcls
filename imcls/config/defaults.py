# -*- coding:utf-8 -*-
###
# File: defaults.py
# Created Date: Friday, September 4th 2020, 2:29:41 pm
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
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2
_C.DIVICE = "cuda:0"

_C.NETWORK = CN()
_C.NETWORK.META_ARCH = "GeneralClsModel"
_C.NETWORK.NUM_CLASSES = 3
_C.NETWORK.BACKBONE_ARCH = "resnet50_cbam"
_C.NETWORK.BNNECK = False
_C.NETWORK.FC_DROPOUT = 0.0
_C.NETWORK.LABEL_SMOOTHING = 0.1
_C.NETWORK.PRETRAINED = True

_C.NETWORK.BENCHMARK = True
_C.NETWORK.DETERMINISTIC = False

_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 32000
_C.SOLVER.CHECKPOINT_PERIOD = 2000
# See detectron2/solver/build.py for LR scheduler options
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
_C.SOLVER.BASE_LR = 0.00333*2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False
_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"


# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

_C.DATA = CN()
_C.DATA.TRAIN_SIZE = (224, 224)
_C.DATA.VALID_SIZE = (224, 224)
_C.DATA.TEST_SIZE = (224, 224)
_C.DATA.BATCHSIZE = 16
_C.DATA.NUM_WORKERS = 4
_C.DATA.MIXUP = 0.2

_C.DATA.DATASETS = CN()
_C.DATA.DATASETS.TRAIN_DIR = ""
_C.DATA.DATASETS.VALID_DIR = ""
_C.DATA.DATASETS.TEST_DIR = ""
# csv dataset
_C.DATA.DATASETS.ROOT_DIR = ""
_C.DATA.DATASETS.TRAIN_CSV = ""
_C.DATA.DATASETS.VALID_CSV = ""
_C.DATA.DATASETS.TEST_CSV = ""

_C.OUTPUT_DIR = ""

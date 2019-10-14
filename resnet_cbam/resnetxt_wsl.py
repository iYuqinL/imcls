# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Optional list of dependencies required by the package

'''
    Code From : https://github.com/facebookresearch/WSL-Images/blob/master/hubconf.py
'''
__all__ = ['resnext101_32x8d_wsl', 'resnext101_32x16d_wsl', 'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl']
import torch
from .resnet_cbam import ResNet, Bottleneck
dependencies = ['torch', 'torchvision']

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# from .Res import ResNet, Bottleneck

# model_urls = {
#     'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
#     'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
#     'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
#     'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
# }

model_urls = {
    'resnext101_32x8d': 'resnet_cbam/pretrained_model/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'resnet_cbam/pretrained_model/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'resnet_cbam/pretrained_model/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'resnet_cbam/pretrained_model/ig_resnext101_32x48-3e41cc8a.pth',
}


# def _resnext(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
#     model.load_state_dict(state_dict)
#     return model

# 使用部分加载
def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    ifcbam = kwargs['ifcbam']
    load_fc = (model.num_classes == 1000)
    if pretrained:
        print('load {} pretrained model'.format(arch))
        if 'http' in model_urls[arch]:
            state_dict = load_state_dict_from_url(model_urls[arch])
        else:
            state_dict = torch.load(model_urls[arch])
        if load_fc:
            print('load {} pretrained model include fc layer'.format(arch))
            if ifcbam is False:
                model.load_state_dict(state_dict)
            else:
                res = model.load_state_dict(state_dict, strict=False)
                assert(
                    str(res.missing_keys) == str(['ca.fc1.weight', 'ca.fc2.weight', 'sa.conv1.weight']),
                    'issue loading pretrained weights: ca, sa')
        else:
            print('load {} pretrained model not include fc layer'.format(arch))
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            if ifcbam is False:
                res = model.load_state_dict(state_dict, strict=False)
                assert(str(res.missing_keys) == str(['fc.weight', 'fc.bias']),
                       'issue loading pretrained weights, fc')
            else:
                res = model.load_state_dict(state_dict, strict=False)
                assert(str(res.missing_keys) ==
                       str(['ca.fc1.weight', 'ca.fc2.weight', 'sa.conv1.weight', 'fc.weight', 'fc.bias']),
                       'issue loading pretrained weights, fc')
    return model


def resnext101_32x8d_wsl(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnext101_32x16d_wsl(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:zz
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnext101_32x32d_wsl(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 32
    return _resnext('resnext101_32x32d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnext101_32x48d_wsl(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 48
    return _resnext('resnext101_32x48d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

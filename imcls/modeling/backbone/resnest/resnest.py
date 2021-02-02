# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
# Copyright (c) 2020
##
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""

import torch
from .resnet import ResNet, Bottleneck

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
]}


def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]


resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
                      name in _model_sha256.keys()
                      }


def _resnest(arch, block, layers, pretrained, progress=True, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            resnest_model_urls[arch], progress=progress, check_hash=True)
        print('load {} pretrained model not include fc layer'.format(arch))
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        res = model.load_state_dict(state_dict, strict=True)
    return model


def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = _resnest('resnest50', Bottleneck, [3, 4, 6, 3], pretrained,
                     radix=2, groups=1, bottleneck_width=64, deep_stem=True, stem_width=32,
                     avg_down=True, avd=True, avd_first=False, **kwargs)
    return model


def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = _resnest('resnest101', Bottleneck, [3, 4, 23, 3], pretrained,
                     radix=2, groups=1, bottleneck_width=64, deep_stem=True, stem_width=64,
                     avg_down=True, avd=True, avd_first=False, **kwargs)
    return model


def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = _resnest('resnest200', Bottleneck, [3, 24, 36, 3], pretrained,
                     radix=2, groups=1, bottleneck_width=64, deep_stem=True, stem_width=64,
                     avg_down=True, avd=True, avd_first=False, **kwargs)

    return model


def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = _resnest('resnest269', Bottleneck, [3, 30, 48, 8], pretrained=pretrained,
                     radix=2, groups=1, bottleneck_width=64, deep_stem=True, stem_width=64,
                     avg_down=True, avd=True, avd_first=False, **kwargs)
    return model

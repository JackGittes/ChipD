from typing import Callable
import torch.nn as nn
from model.contrib.mobilenetv2 import MobileNetV2
from model.modules import DepthwiseSeparableConv


class DefaultMobileNetV2Backbone(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.feature_idx = [13]
        self.features = net.features[:14]
        self.extra_l3 = DepthwiseSeparableConv(in_chns=72, out_chns=144, stride=1, padding=1)
        self.extra_14 = DepthwiseSeparableConv(in_chns=144, out_chns=288, stride=2, padding=1)

    def forward(self, x):
        res = list()
        x = self.features(x)
        res.append(x)
        x = self.extra_l3(x)
        res.append(x)
        x = self.extra_14(x)
        res.append(x)
        return res


def build_mobilenetv2_backbone(cfg):
    import torch
    net = MobileNetV2(num_classes=1000, width_mult=0.75)
    net.load_state_dict(torch.load('weight/mobilenetv2_0.75-dace9791.pth'))
    backbone = DefaultMobileNetV2Backbone(net)
    return backbone


def build_shufflenetv2_backbone(cfg):
    return NotImplemented


_BASE_BACKBONE = {'build_mobilenetv2_backbone': build_mobilenetv2_backbone,
                  'build_shufflenetv2_backbone': build_shufflenetv2_backbone}


class BackboneRegister(object):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, func: Callable) -> None:
        global _BASE_BACKBONE
        _BASE_BACKBONE.update({self.name: func})


def build_backbone(cfg):
    return _BASE_BACKBONE[cfg.MODEL.BACKBONE.NAME](cfg)

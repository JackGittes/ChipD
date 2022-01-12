from typing import Callable, List, Tuple
import torch.nn as nn


class LightHead(nn.Module):
    def __init__(self, in_chn: int,
                 out_chn: int,
                 stride: int = 1):
        super().__init__()
        assert stride in [1, 2], "Light head only supports strides of 1 or 2."
        self.pred = nn.Sequential(nn.Conv2d(in_channels=in_chn, out_channels=in_chn,
                                            kernel_size=3, stride=stride, groups=in_chn, padding=1),
                                  nn.BatchNorm2d(in_chn),
                                  nn.ReLU6(),
                                  nn.Conv2d(in_channels=in_chn, out_channels=out_chn,
                                            kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        return self.pred(x)


def prediction_head(backbone_channels: List[int],
                    num_anchors: List[int],
                    num_classes: int,
                    light_head: bool = False) -> Tuple[List[nn.Module]]:
    loc_layers, conf_layers = [], []
    if light_head:
        for pred_idx, chn in enumerate(backbone_channels):
            loc_layers += [nn.Sequential(nn.BatchNorm2d(chn),
                                         LightHead(chn, chn),
                                         nn.BatchNorm2d(chn),
                                         nn.ReLU6(inplace=True),
                                         LightHead(chn, num_anchors[pred_idx] * 4))]
            conf_layers += [nn.Sequential(nn.BatchNorm2d(chn),
                                          LightHead(chn, chn),
                                          nn.BatchNorm2d(chn),
                                          nn.ReLU6(inplace=True),
                                          LightHead(chn, num_anchors[pred_idx] * num_classes))]
    else:
        for pred_idx, chn in enumerate(backbone_channels):
            loc_layers += [nn.Sequential(nn.BatchNorm2d(chn),
                                         nn.Conv2d(chn, chn, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(chn),
                                         nn.ReLU6(inplace=True),
                                         nn.Conv2d(chn, num_anchors[pred_idx] * 4, kernel_size=3, padding=1))]
            conf_layers += [nn.Sequential(nn.BatchNorm2d(chn),
                                          nn.Conv2d(chn, chn, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(chn),
                                          nn.ReLU6(inplace=True),
                                          nn.Conv2d(chn, num_anchors[pred_idx] * num_classes,
                                                    kernel_size=3, padding=1))]
    return (loc_layers, conf_layers)


def build_light_head(cfg):
    head = prediction_head(cfg.MODEL.FEATURE_CHANNELS,
                           cfg.ANCHOR.NUM_PER_LEVEL,
                           cfg.DATASET.NUM_CLASSES,
                           True)
    return head


def build_normal_head(cfg):
    head = prediction_head(cfg.MODEL.FEATURE_CHANNELS,
                           cfg.ANCHOR.NUM_PER_LEVEL,
                           cfg.DATASET.NUM_CLASSES,
                           False)
    return head


_BASE_HEAD = {'light': build_light_head,
              'normal': build_normal_head}


class HeadRegister(object):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, func: Callable) -> None:
        global _BASE_HEAD
        _BASE_HEAD.update({self.name: func})


def build_head(cfg):
    return _BASE_HEAD[cfg.MODEL.HEAD.NAME](cfg)

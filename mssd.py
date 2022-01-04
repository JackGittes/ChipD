from typing import List, Tuple
import torch
import torch.nn as nn
from layers import PriorBox, Detect

from mcunet.build import build_from_config
from mcunet.tinynas.nn.networks import ProxylessNASNets


# --------------------------------------------------------------
#   SSD detector definition
# --------------------------------------------------------------


class SSD(nn.Module):
    def __init__(self,
                 cfg,
                 base: nn.Module,
                 head: nn.Module):
        super(SSD, self).__init__()

        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = nn.Parameter(self.priorbox.forward(), requires_grad=False)  # updated version

        # SSD network
        self.backbone = base

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(self.cfg)

    def forward(self, x):
        loc, conf = list(), list()
        sources = self.backbone(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        output = (loc.view(loc.size(0), -1, 4),
                  conf.view(conf.size(0), -1, self.num_classes))
        return output

    @torch.no_grad()
    def inference(self, loc, conf):
        return self.detect(loc, self.softmax(conf), self.priors)

# --------------------------------------------------------------------
#    Backbone definition
# --------------------------------------------------------------------


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_chns: int, out_chns: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=in_chns,
                                            kernel_size=3,
                                            stride=stride, padding=padding, groups=in_chns),
                                  nn.BatchNorm2d(in_chns),
                                  nn.ReLU6(),
                                  nn.Conv2d(in_channels=in_chns, out_channels=out_chns,
                                            kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm2d(out_chns),
                                  nn.ReLU6())

    def forward(self, x):
        return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.features = net.features[:14]

        self.level_2 = DepthwiseSeparableConv(in_chns=72, out_chns=144, stride=1, padding=1)
        self.level_3 = DepthwiseSeparableConv(in_chns=144, out_chns=144, stride=1, padding=1)

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.level_2(x1)
        x3 = self.level_3(x2)
        return [x1, x2, x3]


class MCUBackbone(nn.Module):
    def __init__(self, basenet: ProxylessNASNets):
        super().__init__()
        self.input = basenet.first_conv
        self.blocks = basenet.blocks[:18]

        # for final scale level features
        self.conv = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=96,
                                            kernel_size=3, padding=1, stride=1, groups=96),
                                  nn.BatchNorm2d(96),
                                  nn.ReLU6(),
                                  nn.Conv2d(in_channels=96, out_channels=96,
                                            kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm2d(96),
                                  nn.ReLU6())

    def forward(self, x: torch.Tensor):
        x = self.input(x)
        res = list()
        for idx, m in enumerate(self.blocks):
            x = m(x)
            if idx in [14, 17]:
                res.append(x)
        res.append(self.conv(x))
        return res

# -----------------------------------------------------------------
#    Prediction head
# -----------------------------------------------------------------


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
            loc_layers += [LightHead(chn, num_anchors[pred_idx] * 4)]
            conf_layers += [LightHead(chn, num_anchors[pred_idx] * num_classes)]
    else:
        for pred_idx, chn in enumerate(backbone_channels):
            loc_layers += [nn.Conv2d(chn, num_anchors[pred_idx] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(chn, num_anchors[pred_idx] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


def build_ssd(cfg):
    from models.mobilenetv2 import MobileNetV2
    net = MobileNetV2(num_classes=1000, width_mult=0.75)
    net.load_state_dict(torch.load('weight/mobilenetv2_0.75-dace9791.pth'))
    backbone = MobileNetV2Backbone(net)

    head = prediction_head(cfg.MODEL.FEATURE_CHANNELS,
                           cfg.ANCHOR.NUM_PER_LEVEL,
                           cfg.DATASET.NUM_CLASSES,
                           True)
    return SSD(cfg, backbone, head)


if __name__ == "__main__":
    class Empty:
        pass
    Arg = Empty()
    Arg.num_classes = 2
    Arg.size = 256
    Arg.light_head = True
    ppp = build_ssd(Arg)
    print(ppp)
    for name, m in ppp.named_modules():
        print(name)
    from torchsummary import summary
    summary(ppp, (3, 256, 256), device='cpu')
    torch.save(ppp.state_dict(), 'name.pth')

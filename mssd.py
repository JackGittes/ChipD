from typing import List, Tuple
import torch
import torch.nn as nn
from layers import PriorBox, Detect
from data.config import airbus

from mcunet.build import build_from_config
from mcunet.tinynas.nn.networks import ProxylessNASNets


# --------------------------------------------------------------
#   SSD detector definition
# --------------------------------------------------------------


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self,
                 size: Tuple[int],
                 base: nn.Module,
                 head: nn.Module,
                 num_classes: int):
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.cfg = airbus
        self.priorbox = PriorBox(self.cfg)
        self.priors = nn.Parameter(self.priorbox.forward(), requires_grad=False)  # updated version
        self.size = size

        # SSD network
        self.backbone = base

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, 200, 0.2, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        loc, conf = list(), list()
        sources = self.backbone(x)

        # apply multibox head to source layers
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


class MCUBackbone(nn.Module):
    def __init__(self, basenet: ProxylessNASNets):
        super().__init__()
        self.input = basenet.first_conv
        self.blocks = basenet.blocks[:18]

        # for final scale level features
        self.conv = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=96,
                                            kernel_size=3, padding=1, stride=2, groups=96),
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


# -----------------------------------------------------------------
#    Anchor head
# -----------------------------------------------------------------

mbox = {
    '256': [30, 30, 30],  # number of boxes per feature map location
}


def build_ssd(args):
    num_classes = args.num_classes
    input_size = args.size

    net = build_from_config(args)
    backbone = MCUBackbone(net)

    head = prediction_head([48, 96, 96], mbox[str(input_size)], num_classes, args.light_head)
    return SSD(input_size, backbone, head, num_classes)

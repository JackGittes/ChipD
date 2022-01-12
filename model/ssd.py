from typing import Callable
import torch
import torch.nn as nn
from layers import PriorBox, Detect
from model.backbone import build_backbone
from model.head import build_head


# --------------------------------------------------------------
#   SSD detector definition
# --------------------------------------------------------------


class SSD(nn.Module):
    def __init__(self,
                 cfg,
                 backbone: nn.Module,
                 head: nn.Module):
        super(SSD, self).__init__()

        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = nn.Parameter(self.priorbox.forward(), requires_grad=False)  # updated version

        # SSD network
        self.backbone = backbone

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


def build_default_ssd(cfg) -> SSD:
    backbone = build_backbone(cfg)
    head = build_head(cfg)
    return SSD(cfg, backbone, head)


_BASE_SSD = {'SSD': build_default_ssd}


class SSDRegister(object):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, func: Callable) -> None:
        global _BASE_SSD
        _BASE_SSD.update({self.name: func})


def build_ssd(cfg) -> SSD:
    return _BASE_SSD[cfg.MODEL.NAME](cfg)

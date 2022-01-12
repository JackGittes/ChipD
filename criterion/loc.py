from typing import Callable
from torch.nn import SmoothL1Loss
from utils.check import check_implemented


def smoothl1_loss(cfg):
    return SmoothL1Loss(reduction='sum')


_BASE_LOC_LOSS = {'smooth_l1': smoothl1_loss}


class LOCLossRegister(object):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, func: Callable) -> None:
        global _BASE_LOC_LOSS
        _BASE_LOC_LOSS.update({self.name: func})


def build_loc(cfg):
    res = _BASE_LOC_LOSS.get(cfg.LOSS.IOU.NAME, NotImplemented)
    check_implemented(res, "Expected Loc loss is not implemented given the name: {}.".format(cfg.LOSS.IOU.NAME))
    return res(cfg)

from typing import Callable
import torch
import torch.nn as nn

from utils.check import check_implemented
from ops.box import bbox_overlaps_ciou, bbox_overlaps_giou, bbox_overlaps_iou, bbox_overlaps_diou


def build_iou(cfg) -> Callable:
    def iou_loss(decoded_boxes: torch.Tensor, loc_t: torch.Tensor):
        return torch.sum(1.0 - bbox_overlaps_iou(decoded_boxes, loc_t))
    return iou_loss


def build_diou(cfg) -> Callable:
    def diou_loss(decoded_boxes: torch.Tensor, loc_t: torch.Tensor):
        return torch.sum(1.0 - bbox_overlaps_diou(decoded_boxes, loc_t))
    return diou_loss


def build_giou(cfg) -> Callable:
    def giou_loss(decoded_boxes: torch.Tensor, loc_t: torch.Tensor):
        return torch.sum(1.0 - bbox_overlaps_giou(decoded_boxes, loc_t))
    return giou_loss


def build_ciou(cfg) -> Callable:
    def ciou_loss(decoded_boxes: torch.Tensor, loc_t: torch.Tensor):
        return torch.sum(1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t))
    return ciou_loss


_BASE_IOU_LOSS = {'ciou': build_ciou,
                  'diou': build_diou,
                  'giou': build_giou,
                  'iou': build_iou}


class IOULossRegister(object):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, func: Callable) -> None:
        global _BASE_IOU_LOSS
        _BASE_IOU_LOSS.update({self.name: func})


def build_iou_loss_func(cfg):
    res = _BASE_IOU_LOSS.get(cfg.LOSS.IOU.NAME, NotImplemented)
    check_implemented(res, "Expected IoU loss is not implemented given the name: {}.".format(cfg.LOSS.IOU.NAME))
    return res(cfg)


class IOULoss(nn.Module):
    def __init__(self, cfg, size_sum: bool = True):
        super(IOULoss, self).__init__()
        self.size_sum = size_sum
        self.iou_func = build_iou_loss_func(cfg)

    def forward(self, decoded_boxes: torch.Tensor, loc_t: torch.Tensor):
        num = decoded_boxes.shape[0]
        loss = self.iou_func(decoded_boxes, loc_t)
        if self.size_sum:
            loss = loss
        else:
            loss = loss / num
        return loss


def build_iou_loss(cfg):
    return IOULoss(cfg, size_sum=cfg.LOSS.IOU.SIZE_SUM)

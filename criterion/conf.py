from criterion.modules import FocalLoss
from torch.nn import CrossEntropyLoss
from criterion.common import _BASE_LOSS, LossRegister


@LossRegister('focal')
def build_focal_loss(cfg):
    return FocalLoss(class_num=cfg.DATASET.NUM_CLASSES,
                     gamma=cfg.LOSS.CONF_LOSS.FOCAL.GAMMA,
                     alpha=cfg.LOSS.CONF_LOSS.FOCAL.ALPHA,
                     size_average=True)


@LossRegister('cross_entropy')
def build_cross_entropy(cfg):
    return CrossEntropyLoss(reduction='sum')


def build_conf(cfg):
    return _BASE_LOSS[cfg.LOSS.CONF_LOSS.NAME](cfg)

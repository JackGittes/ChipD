from typing import Callable
from torch.optim import SGD, Adam, AdamW


def build_sgd(cfg, params_dict):
    return SGD(params_dict, lr=cfg.OPTIMIZER.LR,
               momentum=cfg.OPTIMIZER.MOMENTUM,
               weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)


def build_adam(cfg, params_dict):
    return Adam(params_dict, lr=cfg.OPTIMIZER.LR,
                betas=tuple(cfg.OPTIMIZER.BETAS),
                weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)


def build_adamw(cfg, params_dict):
    return AdamW(params_dict, lr=cfg.OPTMIZER.LR,
                 betas=tuple(cfg.OPTIMIZER.BETAS),
                 weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)


_BASE_OPTIM = {'SGD': build_sgd,
               'Adam': build_adam,
               'AdamW': build_adamw}


class OptimizerRegister(object):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, func: Callable) -> None:
        global _BASE_OPTIM
        _BASE_OPTIM.update({self.name: func})


def build_optim(cfg, params_dict: dict):
    return _BASE_OPTIM[cfg.OPTIMIZER.NAME](cfg, params_dict)

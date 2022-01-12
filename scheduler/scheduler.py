"""
Author: Zhao Mingxin
Date: 01/07/2022

Description: training scheduler builder.
"""

from typing import Callable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from scheduler.cosine import CosineAnnealingWarmupRestarts


def build_multistep(cfg, optimizer: Optimizer) -> _LRScheduler:
    return MultiStepLR(optimizer,
                       milestones=cfg.SCHEDULER.MULTI_STEP.LR_STEPS,
                       gamma=cfg.SCHEDULER.MULTI_STEP.GAMMA)


def build_cosine_warmup(cfg, optimizer: Optimizer) -> _LRScheduler:
    return CosineAnnealingWarmupRestarts(optimizer,
                                         first_cycle_steps=cfg.SCHEDULER.COSINE_WARMUP.LENGTH,
                                         max_lr=cfg.SCHEDULER.COSINE_WARMUP.MAX_LR,
                                         min_lr=cfg.SCHEDULER.COSINE_WARMUP.MIN_LR,
                                         warmup_steps=cfg.SCHEDULER.COSINE_WARMUP.WARMUP_STEPS)


_BASE_SCHED = {'cosine_warmup': build_cosine_warmup,
               'multistep': build_multistep}


class SchedulerRegister(object):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, func: Callable) -> None:
        global _BASE_SCHED
        _BASE_SCHED.update({self.name: func})


def build_scheduler(cfg, optimizer):
    return _BASE_SCHED[cfg.SCHEDULER.NAME](cfg, optimizer)

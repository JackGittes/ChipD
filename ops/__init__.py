# from anchor import 

from typing import Callable


_BASE_ANCHOR = dict()


class AnchorGeneratorRegister(object):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, func: Callable) -> None:
        global _BASE_ANCHOR
        _BASE_ANCHOR.update({self.name: func})


def build_anchor(cfg):
    return _BASE_ANCHOR[cfg.ANCHOR.GEN_METHOD.NAME](cfg)

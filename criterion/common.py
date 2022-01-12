from typing import Callable

_BASE_LOSS = dict()


class LossRegister(object):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, func: Callable) -> None:
        _BASE_LOSS.update({self.name: func})

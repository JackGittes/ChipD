from typing import Any


def check_implemented(inp: Any, err_info: str = ''):
    if inp == NotImplemented:
        raise NotImplementedError(err_info)

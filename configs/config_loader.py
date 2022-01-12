"""
Author: Zhao Mingxin
Date: 01/08/2022

Description: configuration loader for loading yaml configs and converting it to
    a configuration object defined in `configs/default.py`.
"""

import os
import copy
import yaml

import sys
sys.path.append('.')

from configs.default import _C as CFG
from configs.default import ConfigTemplate


class ConfigLoader:
    def __init__(self, config_path: str) -> None:

        assert os.path.isfile(config_path)
        assert config_path.endswith('.yml')

        with open(config_path, 'r') as fp:
            config_dict: dict = yaml.safe_load(fp.read())
        self.config_dict = config_dict

    def init(self):
        cfg = copy.deepcopy(CFG)
        recursive_init(cfg, self.config_dict)
        return cfg


def recursive_init(cfg: ConfigTemplate, cfg_dict: dict) -> None:
    attrs = cfg.__dir__()
    cfg_attr = list(filter(lambda x: not x.startswith('_'), attrs))
    cfg_dict_key = list(cfg_dict.keys())
    assert set(cfg_dict_key).issubset(cfg_attr), "Configuration conflicts between "\
        "the template and yaml dict."
    if len(cfg_dict_key) == 0:
        return

    for attr in cfg_dict_key:
        child_cfg = getattr(cfg, attr)
        if isinstance(child_cfg, ConfigTemplate):
            recursive_init(child_cfg, cfg_dict.get(attr))
        else:
            valid_cfg_value = cfg_dict.get(attr, None)
            setattr(cfg, attr, valid_cfg_value)


if __name__ == "__main__":
    cfg_loader = ConfigLoader('configs/default.yml')
    my_cfg = cfg_loader.init()
    print(my_cfg)

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
    res = list(filter(lambda x: not x.startswith('_'), attrs))
    if len(res) == 0:
        return
    assert set(res) == set(cfg_dict.keys()), "Configuration conflicts between "\
        "the template and yaml dict."
    assert len(set(res)) == len(res), "Duplicated configuration may be found."

    for attr in res:
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

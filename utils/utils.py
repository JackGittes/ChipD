import os
import warnings
from datetime import datetime
import sys
sys.path.append('.')


def get_log_folder(root: str) -> str:
    assert os.path.exists(root), "Log root does not exist."
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    folder_name = "log-{}".format(time_str)
    log_folder = os.path.join(root, folder_name)
    if os.path.exists(log_folder):
        warnings.warn(RuntimeWarning("Log folder name conflicts!"))
    else:
        os.mkdir(log_folder)
    return log_folder


def load_config(config_path):
    from configs.config_loader import ConfigLoader
    cfg_loader = ConfigLoader(config_path)
    training_cfg = cfg_loader.init()
    return training_cfg

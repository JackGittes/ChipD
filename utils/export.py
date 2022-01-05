import os
import torch
import sys
sys.path.append('.')

from mssd import build_ssd
from utils import load_config


def export_onnx(cfg,
                save_path: str) -> None:
    assert os.path.isdir(save_path), "Given path does not exist."

    full_path = os.path.join(save_path, 'ssd.onnx')
    ssd = build_ssd(cfg)
    torch.onnx.export(ssd,
                      torch.randn((1, 3,
                                   cfg.MODEL.INPUT_SIZE,
                                   cfg.MODEL.INPUT_SIZE)),
                      full_path,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['loc', 'conf'])


if __name__ == "__main__":
    cfg = load_config('experiment/default.yml')
    export_onnx(cfg, 'export')

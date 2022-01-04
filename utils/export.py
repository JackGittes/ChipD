import os
import torch
import sys
sys.path.append('.')

from mssd import build_ssd
from demo import parse_args


def export_onnx(args,
                save_path: str) -> None:
    assert os.path.isdir(save_path), "Given path does not exist."

    full_path = os.path.join(save_path, 'ssd.onnx')
    ssd = build_ssd(args)
    torch.onnx.export(ssd,
                      torch.randn((1, 3, args.size, args.size)),
                      full_path,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['loc', 'conf'])


if __name__ == "__main__":
    parsed_args = parse_args()
    export_onnx(parsed_args, 'export')

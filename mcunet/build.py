import json
import argparse
import torch
from torch import nn

from mcunet.tinynas.nn.networks import ProxylessNASNets


def parse_args():
    parser = argparse.ArgumentParser()
    # architecture setting
    parser.add_argument('-a', '--arch', metavar='ARCH', default='proxyless')
    parser.add_argument('--net_config', default=r'F:\PythonProject\ssd.pytorch-master\ssd.pytorch-master\mcunet\assets\configs\mcunet-320kb-1mb_imagenet_vp.json', type=str)
    parser.add_argument('--weight_path', default='weights/mcunet-320kb-1mb_imagenet.pth', type=str)

    args = parser.parse_args()
    return args


def build_from_config(args):

    with open(args.net_config) as f:
        config = json.load(f)
        args.resolution = config['resolution']
    model = ProxylessNASNets.build_from_config(config)
    return model


class MCUBackbone(nn.Module):

    def __init__(self, basenet: ProxylessNASNets):
        super().__init__()
        self.input = basenet.first_conv
        self.blocks = basenet.blocks[:18]
        self.conv = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=96,
                                            kernel_size=3, padding=1, stride=2),
                                  nn.BatchNorm2d(96),
                                  nn.ReLU6())

    def forward(self, x: torch.Tensor):
        x = self.input(x)
        res = list()
        for idx, m in enumerate(self.blocks):
            x = m(x)
            if idx in [15, 17]:
                res.append(x)
        res.append(self.conv(x))
        return res


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    parsed_args = parse_args()
    net = build_from_config(parsed_args)
    # net.load_state_dict(torch.load(parsed_args.weight_path)["state_dict"])

    net0 = MCUBackbone(net)
    summary(net0, (3, 256, 256), device='cpu')

    # torch.onnx.export(net0,               # model being run
    #                   torch.randn((1, 3, 256, 256)),                         # model input (or a tuple for multiple inputs)
    #                   "mcunet_backbone.onnx",   # where to save the model (can be a file or file-like object)
    #                   export_params=True,        # store the trained parameter weights inside the model file
    #                   opset_version=10,          # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names = ['input'],   # the model's input names
    #                   output_names = ['output'], # the model's output names
    #                   )
    print(net0)

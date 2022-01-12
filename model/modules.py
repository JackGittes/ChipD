import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_chns: int, out_chns: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=in_chns,
                                            kernel_size=3,
                                            stride=stride, padding=padding, groups=in_chns),
                                  nn.BatchNorm2d(in_chns),
                                  nn.ReLU6(),
                                  nn.Conv2d(in_channels=in_chns, out_channels=out_chns,
                                            kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm2d(out_chns),
                                  nn.ReLU6())

    def forward(self, x):
        return self.conv(x)

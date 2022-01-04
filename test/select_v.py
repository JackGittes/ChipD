import sys
sys.path.append('.')


if __name__ == '__main__':

    from models.mobilenetv2 import MobileNetV2
    net = MobileNetV2(num_classes=1000, width_mult=0.75)

    from torchsummary import summary
    summary(net.features[:14], (3, 256, 256), device='cpu')
    # print(net.named_modules)
    for name, m in net.named_modules():
        print(name)
    # print(net)
    import torch
    torch.save(net.features[:14], 'param.pth')

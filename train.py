
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import argparse

from tensorboardX import SummaryWriter

from utils.utils import get_log_folder
from data import detection_collate
from data.airbus import AirbusDetection
from data.config import airbus, MEANS
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from mssd import SSD, build_ssd


def parse_args():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')

    # dataset settings
    parser.add_argument('--dataset', default='Airbus', choices=['Airbus'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default=r'H:\Dataset\ShipDET-Horizon',
                        help='Dataset root directory path')
    parser.add_argument('--num_classes', default=2, type=int)

    # model settings
    parser.add_argument('--net_config', default=r'mcunet\assets\configs\mcunet-320kb-1mb_imagenet_vp.json',
                        help='Pretrained base model')
    parser.add_argument('--size', default=256, type=int, help='Input image size.')
    parser.add_argument('--light_head', default=True, type=bool)

    # training settings
    parser.add_argument('--phase', default='train', choices=['train', 'test'])
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--num_workers', default=12, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda_disable', action='store_true')

    # optimizer settings
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')

    # logger settings
    parser.add_argument('--log_dir', default='log', type=str, help='Directory for saving training log.')
    args = parser.parse_args()
    return args


def train(args):
    if args.dataset == 'Airbus':
        cfg = airbus
        dataset = AirbusDetection(root=args.dataset_root,
                                  transform=SSDAugmentation(cfg['min_dim'], MEANS), train=True)
    log_dir = get_log_folder(args.log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    ssd_net: SSD = build_ssd(args)

    priors = ssd_net.priors
    net = torch.nn.DataParallel(ssd_net)

    # send to GPU
    net = net.cuda()
    priors = priors.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, not args.cuda_disable)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        t0 = time.time()
        if iteration != 0 and (iteration % epoch_size == 0):
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(args, optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        images = images.cuda()
        with torch.no_grad():
            targets = [ann.cuda() for ann in targets]

        # forward
        loc, conf = net(images)
        out = (loc, conf, priors)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()

        loc_loss += loss_l.data.item()
        conf_loss += loss_c.data.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data.item()), end=' ')
            writer.add_scalar(tag='loss', scalar_value=float(loss.data.item()), global_step=iteration)
            writer.add_scalar(tag='conf', scalar_value=float(loss_c.data.item()), global_step=iteration)
            writer.add_scalar(tag='loc', scalar_value=float(loss_l.data.item()), global_step=iteration)
        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), os.path.join(log_dir, repr(iteration) + '.pth'))
    torch.save(ssd_net.state_dict(),
               os.path.join(log_dir, "final.pth"))


def adjust_learning_rate(args, optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    parsed_args = parse_args()
    train(parsed_args)


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse

from tensorboardX import SummaryWriter

from utils.utils import get_log_folder, load_config
from data import detection_collate
from data.airbus import AirbusDetection
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from mssd import SSD, build_ssd

import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    parser.add_argument('--config_path', default='experiment/default.yml', type=str)
    args = parser.parse_args()
    return args


def train(args):
    cfg = load_config(args.config_path)
    dataset = AirbusDetection(root=cfg.DATASET.ROOT,
                              transform=SSDAugmentation(cfg.MODEL.INPUT_SIZE), train=True)
    log_dir = get_log_folder(cfg.LOGGER.ROOT)
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(log_dir, 'train.log'),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logging.getLogger().addHandler(logging.StreamHandler())

    import shutil
    logger.info('Backup training configuration to: {}'.format(log_dir))
    shutil.copy(args.config_path, os.path.join(log_dir, 'default.yml'))

    writer = SummaryWriter(log_dir=log_dir)
    ssd_net: SSD = build_ssd(cfg)

    priors = ssd_net.priors
    net = torch.nn.DataParallel(ssd_net)

    # send to GPU
    net = net.cuda()
    priors = priors.cuda()

    logger.info('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(),
                          lr=cfg.OPTIMIZER.LR,
                          weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

    criterion = MultiBoxLoss(cfg)

    net.train()

    logger.info('Loading the dataset...')
    logger.info('Training SSD on: {}'.format(dataset.name))
    logger.info('Using the specified args:')
    logger.info(cfg)

    step_index = 0
    data_loader = data.DataLoader(dataset, cfg.TRAINING.BATCH_SIZE,
                                  num_workers=cfg.TRAINING.NUM_WORKERS,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(cfg.TRAINING.START_ITER, cfg.TRAINING.MAX_ITER):
        t0 = time.time()

        if iteration in cfg.SCHEDULER.LR_STEPS:
            step_index += 1
            adjust_learning_rate(cfg.OPTIMIZER.LR, optimizer, cfg.SCHEDULER.GAMMA, step_index)

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

        if iteration % 10 == 0:
            logger.info('timer: %.4f sec.' % (t1 - t0) + 'iter ' + repr(iteration) +
                        ' || Loss: %.4f ||' % (loss.data.item()))
            writer.add_scalar(tag='loss', scalar_value=float(loss.data.item()), global_step=iteration)
            writer.add_scalar(tag='conf', scalar_value=float(loss_c.data.item()), global_step=iteration)
            writer.add_scalar(tag='loc', scalar_value=float(loss_l.data.item()), global_step=iteration)
        if iteration != 0 and iteration % cfg.LOGGER.SAVE_INTERVAL == 0:
            logger.info('Saving state, iter: {}'.format(iteration))
            torch.save(ssd_net.state_dict(), os.path.join(log_dir, repr(iteration) + '.pth'))
            torch.save(ssd_net.state_dict(), os.path.join(cfg.LOGGER.ROOT, 'latest.pth'))

    torch.save(ssd_net.state_dict(),
               os.path.join(log_dir, "final.pth"))


def adjust_learning_rate(lr, optimizer, gamma, step):
    new_lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def xavier(param):
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    parsed_args = parse_args()
    train(parsed_args)


import os
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import argparse

from tensorboardX import SummaryWriter
import logging
from utils.utils import get_log_folder, load_config

from data import detection_collate
from data.airbus import AirbusDetection
from utils.augmentations import SSDAugmentation

from model import build_ssd
from criterion import build_criterion
from optim import build_optim
from scheduler import build_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    parser.add_argument('--config_path', default='experiment/default.yml', type=str)
    args = parser.parse_args()
    return args


def train(args):
    cfg = load_config(args.config_path)

    # build logger
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

    # build model
    ssd_net = build_ssd(cfg)

    priors = ssd_net.priors
    net = torch.nn.DataParallel(ssd_net)

    net = net.cuda()
    priors = priors.cuda()
    net.train()

    if not bool(cfg.TRAINING.RESUME):
        logger.info('Initializing weights...')
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
    else:
        assert os.path.isfile(cfg.TRAINING.RESUME_WEIGHT)
        logger.info('Resuming from weight: {}.'.format(cfg.TRAINING.RESUME_WEIGHT))
        net.module.load_state_dict(torch.load(cfg.TRAINING.RESUME_WEIGHT))

    # build optimizer and loss
    optimizer = build_optim(cfg, net.parameters())
    scheduler = build_scheduler(cfg, optimizer)
    # criterion = MultiBoxLoss(cfg)
    criterion = build_criterion(cfg)

    # build dataset
    logger.info('Loading the dataset...')
    dataset = AirbusDetection(root=cfg.DATASET.ROOT,
                              transform=SSDAugmentation(cfg.MODEL.INPUT_SIZE), train=True)
    logger.info('Training SSD on: {}'.format(dataset.name))
    data_loader = data.DataLoader(dataset, cfg.TRAINING.BATCH_SIZE,
                                  num_workers=cfg.TRAINING.NUM_WORKERS,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    batch_iterator = iter(data_loader)

    # start training
    logger.info('Using the specified args:')
    logger.info(cfg)

    for iteration in range(cfg.TRAINING.START_ITER, cfg.TRAINING.MAX_ITER):
        iter_start = time.time()

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
        scheduler.step()

        iter_end = time.time()

        if iteration % 10 == 0:
            normalized_loss_c = float(loss_c.data.item() / cfg.LOSS.CONF_WEIGHT)
            normalized_loss_l = float(loss_l.data.item() / cfg.LOSS.LOC_WEIGHT)
            normalized_loss = normalized_loss_l + normalized_loss_c
            logger.info('timer: %.4f sec.' % (iter_end - iter_start) + 'iter ' + repr(iteration) +
                        ' || Loss: %.4f ||' % (normalized_loss))
            training_info = {"loss": normalized_loss, 'conf': normalized_loss_c, 'loc': normalized_loss_l,
                             "lr": scheduler.get_lr()}
            for info_key, info_value in training_info.items():
                writer.add_scalar(tag=info_key, scalar_value=info_value, global_step=iteration)
        if iteration != 0 and iteration % cfg.LOGGER.SAVE_INTERVAL == 0:
            logger.info('Saving state, iter: {}'.format(iteration))
            torch.save(ssd_net.state_dict(), os.path.join(log_dir, repr(iteration) + '.pth'))
            torch.save(ssd_net.state_dict(), os.path.join(cfg.LOGGER.ROOT, 'latest.pth'))

    torch.save(ssd_net.state_dict(),
               os.path.join(log_dir, "final.pth"))


def xavier(param):
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    parsed_args = parse_args()
    train(parsed_args)

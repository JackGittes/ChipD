from __future__ import print_function
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from data.airbus import AirbusDetection
from data.airbus import AIRBUS_CLASSES as labelmap
from data import BaseTransform


def test_net(save_folder: str,
             net: nn.Module,
             cuda: bool,
             testset: AirbusDetection,
             transform):
    # dump predictions and assoc. ground truth to text file for now
    filename = os.path.join(save_folder, 'test1.txt')
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: ' + img_id + '\n')
            for box in annotation:
                f.write('label: ' + ' || '.join(str(b) for b in box) + '\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: ' + '\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num) + ' label: ' + label_name + ' score: ' +
                            str(score) + ' ' + ' || '.join(str(c) for c in coords) + '\n')
                j += 1


@torch.no_grad()
def test_voc(log_root: str,
             num_classes: int,
             input_size: int,
             net: nn.Module,
             val_dataset: AirbusDetection,
             cuda: bool,
             visual_thresh: float = 0.6):
    # load net
    num_classes = num_classes + 1  # +1 background
    # load data
    if cuda:
        net = net.cuda()
    # evaluation
    eval_root = os.path.join(log_root, 'eval')
    if not os.path.exists(eval_root):
        os.mkdir(eval_root)
    test_net(eval_root, net, cuda, val_dataset,
             BaseTransform(input_size, (104, 117, 123)),
             thresh=visual_thresh)


if __name__ == '__main__':
    test_voc()

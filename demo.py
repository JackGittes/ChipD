import argparse
import torch
import cv2
import os
import numpy as np
from mssd import build_ssd
from torchvision.utils import make_grid


def draw_bboxes(img, boxes):
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for box in boxes:
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        score = "{:.2f}".format(box[4])
        class_id = 1
        class_name = 'ship'
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)


def parse_args():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')

    # dataset settings
    parser.add_argument('--dataset', default='Airbus', choices=['Airbus'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default=r'H:\Dataset\ShipDET-Horizon',
                        help='Dataset root directory path')
    parser.add_argument('--num_classes', default=2, type=int)

    # model settings
    parser.add_argument('--net_config', default=r'mcunet/assets/configs/mcunet-320kb-1mb_imagenet_vp.json',
                        help='Pretrained base model')
    parser.add_argument('--size', default=256, type=int, help='Input image size.')
    parser.add_argument('--light_head', default=True, type=bool)

    # training settings
    parser.add_argument('--phase', default='test', choices=['train', 'test'])

    args = parser.parse_args()
    return args


def demo(cfg_path, weight_path, src_folder='example/source', cuda=True):
    file_list = os.listdir(src_folder)

    from utils.utils import load_config
    cfg = load_config(cfg_path)
    model = build_ssd(cfg)
    model.eval()
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    if cuda:
        model = model.cuda()
        model.priors = model.priors.cuda()

    for item_ in file_list:
        full_path = os.path.join(src_folder, item_)
        if not os.path.isfile(full_path):
            continue
        img = cv2.imread(full_path)
        img = cv2.resize(img, (cfg.MODEL.INPUT_SIZE, cfg.MODEL.INPUT_SIZE))
        temp_img = img.copy()

        img = img.astype(np.float32)
        img -= cfg.DATASET.MEANS
        img /= cfg.DATASET.STD

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img = img.cuda()
        loc, conf = model(img)
        detections = model.inference(loc, conf).cpu().detach().numpy()[0, 1, :, :]
        res = list()
        for i in range(cfg.DETECT.TOP_K):
            if detections[i, 0] > cfg.DETECT.CONFIDENCE_THRESH:
                x0 = detections[i, 1] * cfg.MODEL.INPUT_SIZE
                y0 = detections[i, 2] * cfg.MODEL.INPUT_SIZE
                x1 = detections[i, 3] * cfg.MODEL.INPUT_SIZE
                y1 = detections[i, 4] * cfg.MODEL.INPUT_SIZE
                score = detections[i, 0]

                res.append([x0, y0, x1, y1, score])
        draw_bboxes(temp_img, res)
        cv2.imwrite(os.path.join('example/result', item_), temp_img)
    return detections


def demo_on_running(args, net, src_folder='example/source'):
    file_list = os.listdir(src_folder)

    final_res = list()
    for item_ in file_list:
        full_path = os.path.join(src_folder, item_)
        if not os.path.isfile(full_path):
            continue
        img = cv2.imread(full_path)
        img = cv2.resize(img, (args.size, args.size))
        temp_img: np.ndarray = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)
        img -= (116.28, 103.53, 123.675)
        img /= (57.12, 57.375, 58.395)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        img = img.cuda()
        loc, conf = net(img)
        detections = net.module.inference(loc, conf).cpu().detach().numpy()[0, 1, :, :]
        res = list()
        for i in range(200):
            if detections[i, 0] > 0.2:
                x0 = detections[i, 1] * 256
                y0 = detections[i, 2] * 256
                x1 = detections[i, 3] * 256
                y1 = detections[i, 4] * 256
                score = detections[i, 0]
                # print(x0, y0, x1, y1)
                res.append([x0, y0, x1, y1, score])
        draw_bboxes(temp_img, res)

        tensor_img = torch.tensor(temp_img.transpose(2, 0, 1))
        final_res.append(tensor_img)

    res = make_grid(final_res, nrow=5)
    return res


if __name__ == '__main__':
    import time
    s = time.time()
    cfg_path = 'experiment/default.yml'
    weight_root = 'log/log-2022-01-03-16-46-21'
    full_weight_path = os.path.join(weight_root, 'latest.pth')
    det = demo(cfg_path, weight_path=full_weight_path, cuda=True)
    e = time.time()

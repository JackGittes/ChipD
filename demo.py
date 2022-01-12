import argparse
import torch
import cv2
import os
import numpy as np
from model import build_ssd
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
                coords = [detections[i, idx] * (cfg.MODEL.INPUT_SIZE - 1) for idx in range(1, 5)]
                coords.append(detections[i, 0])
                res.append(coords)
        draw_bboxes(temp_img, res)
        cv2.imwrite(os.path.join('example/result', item_), temp_img)
    return detections


if __name__ == '__main__':
    import time
    s = time.time()
    cfg_path = 'experiment/default.yml'
    weight_root = 'log/'
    full_weight_path = os.path.join(weight_root, 'latest.pth')
    det = demo(cfg_path, weight_path=full_weight_path, cuda=True)
    e = time.time()

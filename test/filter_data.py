import json
import os
import sys
import copy
from typing import List

import cv2

sys.path.append('.')

from data.config import airbus
from utils.anchor import draw_bboxes
from tqdm import tqdm

DST_DIR = r'H:\Dataset\ShipDET-Filtered'
SMALL_THRESH = 75
SAVE_IMG = False


def filter_small_image():
    data_root = airbus['data_root']

    with open(os.path.join(data_root, 'train.json'), 'r') as fp:
        train_dict: dict = json.load(fp)
    with open(os.path.join(data_root, 'val.json'), 'r') as fp:
        val_dict: dict = json.load(fp)

    res = list()
    for k, v in train_dict.items():
        res.append([k, v])
    for k, v in val_dict.items():
        res.append([k, v])

    bar = tqdm(res)
    small_image_list = list()
    for item in bar:
        areas_small = list()
        for box in item[1]:
            areas_small.append(box[2] <= SMALL_THRESH and box[3] <= SMALL_THRESH)
        if all(areas_small):
            small_image_list.append(item[0])
            continue
        if SAVE_IMG:
            full_path = os.path.join(data_root, 'AllImages', item[0])
            img = cv2.imread(full_path)
            visual_box = [[box[1], box[0], box[3], box[2]] for box in item[1]]
            draw_bboxes(img, visual_box)
            cv2.imwrite(os.path.join(DST_DIR, 'AllImages', item[0]), img)

    print('Total small images: ', len(small_image_list))
    with open(os.path.join(DST_DIR, 'small.json'), 'w') as fp:
        json.dump(small_image_list, fp)


def get_blocked_images():
    with open(os.path.join(DST_DIR, 'small.json'), 'r') as fp:
        small_imgs: List = json.load(fp)

    maybe_bad = os.path.join(DST_DIR, 'bad.json')
    if os.path.isfile(maybe_bad):
        with open(maybe_bad, 'r') as fp:
            bad_imgs = json.load(fp)
    else:
        bad_imgs = list()
    small_imgs.extend(bad_imgs)

    data_root = airbus['data_root']

    with open(os.path.join(data_root, 'train.json'), 'r') as fp:
        train_dict: dict = json.load(fp)
    with open(os.path.join(data_root, 'val.json'), 'r') as fp:
        val_dict: dict = json.load(fp)

    res = list()
    for k, v in train_dict.items():
        res.append([k, v])
    for k, v in val_dict.items():
        res.append([k, v])

    copied_res = copy.deepcopy(res)
    for item in res:
        name = item[0]
        if name in small_imgs:
            copied_res.remove(item)

    final_len = len(copied_res)
    print("Remaining images: ", final_len)
    train_len = final_len - final_len // 5

    new_train_dict, new_val_dict = dict(), dict()
    for idx, item in enumerate(copied_res):
        if idx < train_len:
            new_train_dict.update({item[0]: item[1]})
        else:
            new_val_dict.update({item[0]: item[1]})

    with open(os.path.join(data_root, 'train_new.json'), 'w') as fp:
        json.dump(new_train_dict, fp)
    with open(os.path.join(data_root, 'val_new.json'), 'w') as fp:
        json.dump(new_val_dict, fp)


if __name__ == "__main__":
    filter_small_image()
    get_blocked_images()

import json
import os
import shutil
from datetime import datetime
import tempfile
from typing import Optional
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
sys.path.append(".")

from utils.utils import load_config


@torch.no_grad()
def evaluate(model: nn.Module,
             cfg_path: str,
             result_json_path: Optional[str] = None,
             save: bool = True):

    cfg = load_config(cfg_path)

    anno_path = os.path.join(cfg.DATASET.ROOT, "val_coco.json")
    cocoGt = COCO(annotation_file=anno_path)

    with open(anno_path, "r") as fp:
        anno_json: dict = json.load(fp)
    total_len = len(anno_json["images"])
    dataset_root = cfg.DATASET.ROOT

    # use the default path or create a temporary directory for saving results.
    eval_path = cfg.EVAL.RESULT_PATH
    if not save:
        eval_path = tempfile.mkdtemp()

    if result_json_path is None:
        eval_progress = tqdm(range(total_len))
        eval_progress.set_description(desc="COCO metric eval: ")
        res_item = list()
        for im_id in eval_progress:
            im_dict: dict = cocoGt.loadImgs(im_id)[0]
            im_name = im_dict["file_name"]
            full_path = os.path.join(dataset_root, "AllImages", im_name)
            img: np.ndarray = cv2.imread(full_path)

            im_h, im_w, _ = img.shape

            img = cv2.resize(img, (cfg.MODEL.INPUT_SIZE, cfg.MODEL.INPUT_SIZE))

            img = img.astype(np.float32)
            img -= cfg.DATASET.MEANS
            img /= cfg.DATASET.STD

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            img = img.cuda()
            loc, conf = model(img)
            if isinstance(model, nn.DataParallel):
                detections = model.module.inference(loc, conf).cpu().detach().numpy()[0, 1, :, :]
            else:
                detections = model.inference(loc, conf).cpu().detach().numpy()[0, 1, :, :]

            for i in range(cfg.DETECT.TOP_K):
                if detections[i, 0] > cfg.DETECT.CONFIDENCE_THRESH:
                    x1, y1, x2, y2 = [float(detections[i, idx]) for idx in range(1, 5)]
                    w, h = x2 - x1, y2 - y1
                    res_anno = {"image_id": im_id,
                                "category_id": 1,
                                "bbox": [x1 * im_w, y1 * im_h, w * im_w, h * im_h],
                                "score": float(detections[i, 0])}
                    res_item.append(res_anno)
        time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        result_name = "coco_result_{}.json".format(time_stamp)
        result_metric_json = os.path.join(eval_path, "coco_metric_{}.json".format(time_stamp))
        result_json_path = os.path.join(eval_path, result_name)
        with open(result_json_path, "w") as fp:
            json.dump(res_item, fp)
    else:
        assert os.path.isfile(result_json_path) and result_json_path.endswith(".json"), "Given result file not found."
        result_metric_json = result_json_path.replace("coco_result", "coco_metric")

    cocoDt = cocoGt.loadRes(result_json_path)
    imgIds = sorted(cocoGt.getImgIds())

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    coco_result_dict = {"AP(0.50:0.95)-all": None, "AP(0.50)-all": None, "AP(0.75)-all": None,
                        "AP(0.50:0.95)-small": None, "AP(0.50:0.95)-medium": None, "AP(0.50:0.95)-large": None,
                        "AR(0.50:0.95)-all-Maxdet(1)": None, "AR(0.50:0.95)-all-Maxdet(10)": None,
                        "AR(0.50:0.95)-all-Maxdet(100)": None, "AR(0.50:0.95)-small-Maxdet(100)": None,
                        "AR(0.50:0.95)-medium-Maxdet(100)": None, "AR(0.50:0.95)-large-Maxdet(100)": None}
    for idx, k in enumerate(coco_result_dict.keys()):
        coco_result_dict.update({k: cocoEval.stats[idx]})

    if not save:
        shutil.rmtree(eval_path)
    else:
        with open(result_metric_json, "w") as fp:
            json.dump(coco_result_dict, fp)

    return coco_result_dict


if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from model import build_ssd
    cfg_path = 'experiment/default.yml'
    weight_root = 'log/'
    full_weight_path = os.path.join(weight_root, 'latest.pth')

    cfg = load_config(cfg_path)
    model = build_ssd(cfg)
    model.eval()
    model.load_state_dict(torch.load(full_weight_path, map_location='cpu'))
    model = model.cuda()

    evaluate(model, cfg_path, 'result/coco_result_2022_01_13_17_29_06.json', save=False)

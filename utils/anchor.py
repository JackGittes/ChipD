import json
import os
from typing import List
import warnings
import numpy as np
import cv2
from sklearn.cluster import KMeans
import sys

sys.path.append('.')

from data.airbus import AirbusDetection
from data.config import airbus
from layers.functions.prior_box import PriorBox
from mssd import mbox


def analyze_shapes():
    save_path = airbus['anchor_dir']
    if not os.path.isdir(save_path):
        warnings.warn(RuntimeWarning("Given save path does not exist."))

    dataset = AirbusDetection(airbus['data_root'])
    anchor_num = sum(mbox[str(airbus['min_dim'])])

    box_sizes = list()
    for i in range(len(dataset)):
        _, target = dataset.pull_anno(i)
        box_sizes.extend([[item[2] - item[0], item[3] - item[1]] for item in target])
    all_boxes = np.array(box_sizes)
    all_boxes /= 768.
    kmeans = KMeans(n_clusters=anchor_num, random_state=0).fit(all_boxes)
    centroids: np.ndarray = kmeans.cluster_centers_
    centroid_list = centroids.tolist()
    sorted_centroids = sorted(centroid_list, key=lambda x: x[0] * x[1])

    if os.path.isdir(save_path):
        with open(os.path.join(save_path, 'anchor.json'), 'w') as fp:
            json.dump(sorted_centroids, fp)
    print("All anchors:")
    for item in sorted_centroids:
        print(item)


def draw_bboxes(img: np.ndarray,
                boxes: List[List[float]]) -> None:
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for box in boxes:
        size = (int(box[2]), int(box[3]))
        pt1 = (int(box[0] - size[0] / 2.), int(box[1] - size[1] / 2.))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        class_id = 1
        color = colors[class_id * 60 % 255]
        cv2.rectangle(img, pt1, pt2, color, 1)


def draw_anchor():
    size = airbus['min_dim']
    scales = airbus['feature_maps']
    num_anchor_per_scale = mbox[str(size)]

    pb = PriorBox(airbus)
    anchors = pb.forward()

    split_point = [0]
    for scale, anchor_num in zip(scales, num_anchor_per_scale):
        split_point.append(split_point[-1] + scale ** 2 * anchor_num)
    all_boxes = list()
    for idx in range(len(split_point) - 1):
        boxes = anchors[split_point[idx]: split_point[idx + 1]]
        boxes_np = boxes[:num_anchor_per_scale[idx], :].numpy()
        all_boxes.append(np.hstack([np.reshape(boxes_np[:, dim] * size, (-1, 1))
                                    for dim in range(4)]))
    all_boxes = np.vstack(all_boxes)
    all_boxes[:, 0] = size // 2
    all_boxes[:, 1] = size // 2
    empty_image = np.zeros((size, size, 3), dtype=np.uint8)
    draw_bboxes(empty_image, all_boxes)
    cv2.imwrite('anchors.png', empty_image)

    count = 0
    for idx, num_anchor in enumerate(num_anchor_per_scale):
        empty_image = np.zeros((size, size, 3), dtype=np.uint8)
        draw_bboxes(empty_image, all_boxes[count: count + num_anchor, :])
        count += num_anchor
        cv2.imwrite('level_{}.png'.format(idx), empty_image)


if __name__ == "__main__":
    analyze_shapes()

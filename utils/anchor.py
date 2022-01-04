import json
import os
from typing import List
import warnings
import numpy as np
import cv2
import sys

sys.path.append('.')

from data.airbus import AirbusDetection
from layers.functions.prior_box import PriorBox
from utils import load_config


def analyze_shapes():
    cfg = load_config('experiment/default.yml')
    save_path = cfg.ANCHOR.SAVE_PATH
    if not os.path.isdir(save_path):
        warnings.warn(RuntimeWarning("Given save path does not exist."))

    dataset = AirbusDetection(cfg.DATASET.ROOT)
    anchor_num = sum(cfg.ANCHOR.NUM_PER_LEVEL)

    box_sizes = list()
    for i in range(len(dataset)):
        _, target = dataset.pull_anno(i)
        box_sizes.extend([[item[2] - item[0], item[3] - item[1]] for item in target])
    all_boxes = np.array(box_sizes)
    all_boxes /= 768.

    centroids: np.ndarray = kmeans(all_boxes, anchor_num)
    centroid_list = centroids.tolist()
    sorted_centroids = sorted(centroid_list, key=lambda x: x[0] * x[1])

    if os.path.isdir(save_path):
        with open(os.path.join(save_path, 'anchor.json'), 'w') as fp:
            json.dump(sorted_centroids, fp)
    print("All anchors:")
    for item in sorted_centroids:
        print(item)


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


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
    cfg = load_config('experiment/default.yml')
    size = cfg.MODEL.INPUT_SIZE
    scales = cfg.ANCHOR.STEPS
    num_anchor_per_scale = cfg.ANCHOR.NUM_PER_LEVEL

    pb = PriorBox(cfg)
    anchors = pb.forward()
    print(anchors.shape)

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
    draw_anchor()

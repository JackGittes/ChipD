import json
import os
from typing import List
import warnings
import numpy as np
import cv2
import sys

sys.path.append('.')

from data.airbus import AirbusDetection
from utils import load_config


def generate_anchor_by_boxes():
    cfg = load_config('experiment/default.yml')

    def centerize(input_bins: np.ndarray):
        centers = [(input_bins[start_idx] + input_bins[start_idx + 1]) / 2.
                   for start_idx in range(0, input_bins.size - 1)]
        return np.array(centers)

    save_path = cfg.ANCHOR.SAVE_PATH
    if not os.path.isdir(save_path):
        warnings.warn(RuntimeWarning("Given save path does not exist."))

    dataset = AirbusDetection(cfg.DATASET.ROOT, train=True)

    box_sizes = list()
    for i in range(len(dataset)):
        _, target = dataset.pull_anno(i)
        box_sizes.extend([[item[2] - item[0], item[3] - item[1]] for item in target])
    all_boxes = np.array(box_sizes)
    all_boxes /= 768.

    unsorted_aspects = all_boxes[:, 0] / all_boxes[:, 1]

    all_areas = all_boxes[:, 0] * all_boxes[:, 1]
    splitted_area = np.array_split(np.sort(all_areas), len(cfg.ANCHOR.NUM_PER_LEVEL))
    sorted_aspect = unsorted_aspects[np.argsort(all_areas)]
    aspect_list = np.array_split(sorted_aspect, len(cfg.ANCHOR.NUM_PER_LEVEL))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(cfg.ANCHOR.NUM_PER_LEVEL))

    anchors = list()
    for idx, num_shape in enumerate(cfg.ANCHOR.NUM_PER_LEVEL):
        if len(cfg.ANCHOR.NUM_PER_LEVEL) == 1:
            cur_ax_asp = ax[0]
            cur_ax_area = ax[1]
        else:
            cur_ax_asp = ax[0, idx]
            cur_ax_area = ax[1, idx]
        _, aspect_bins, _ = cur_ax_asp.hist(aspect_list[idx], bins=num_shape // 2)
        _, area_bins, _ = cur_ax_area.hist(splitted_area[idx], bins=num_shape // 2)

        aspect_bins = centerize(aspect_bins)
        print(aspect_bins)
        area_bins = centerize(area_bins)
        w = np.sqrt(aspect_bins * area_bins).reshape(-1, 1)
        h = np.sqrt(area_bins / aspect_bins).reshape(-1, 1)

        unsorted_anchors = np.vstack([np.hstack([w, h]), np.hstack([h, w])])
        sorted_anchors = unsorted_anchors[np.argsort(unsorted_anchors[:, 0] * unsorted_anchors[:, 1])]
        anchors.append(sorted_anchors)
    anchor_list = np.vstack(anchors).tolist()
    if os.path.isdir(save_path):
        with open(os.path.join(save_path, 'anchor.json'), 'w') as fp:
            json.dump(anchor_list, fp)
    fig.savefig(os.path.join(cfg.ANCHOR.VISUAL_PATH, 'example.png'))


def analyze_shapes():
    cfg = load_config('experiment/default.yml')
    save_path = cfg.ANCHOR.SAVE_PATH
    if not os.path.isdir(save_path):
        warnings.warn(RuntimeWarning("Given save path does not exist."))

    dataset = AirbusDetection(cfg.DATASET.ROOT, train=True)
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
    num_anchor_per_scale = cfg.ANCHOR.NUM_PER_LEVEL

    with open(os.path.join(cfg.ANCHOR.SAVE_PATH, 'anchor.json'), 'r') as fp:
        anchors = json.load(fp)

    anchors_np = np.array(anchors)
    all_boxes = np.zeros((len(anchors), 4))
    all_boxes[:, 0] = size // 2
    all_boxes[:, 1] = size // 2
    all_boxes[:, 2] = anchors_np[:, 0] * size
    all_boxes[:, 3] = anchors_np[:, 1] * size
    empty_image = np.zeros((size, size, 3), dtype=np.uint8)
    draw_bboxes(empty_image, all_boxes)
    cv2.imwrite(os.path.join(cfg.ANCHOR.VISUAL_PATH, 'anchors.png'), empty_image)
    print("Total anchors:", anchors_np.shape[0])

    count = 0
    for idx, num_anchor in enumerate(num_anchor_per_scale):
        empty_image = np.zeros((size, size, 3), dtype=np.uint8)
        draw_bboxes(empty_image, all_boxes[count: count + num_anchor, :])
        count += num_anchor
        cv2.imwrite(os.path.join(cfg.ANCHOR.VISUAL_PATH, 'level_{}.png'.format(idx)),
                    empty_image)


if __name__ == "__main__":
    analyze_shapes()
    draw_anchor()
    # generate_anchor_by_boxes()

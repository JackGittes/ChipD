"""
Author: Zhao Mingxin
Date: 01/05/2022

Description: Generate anchor boxes and tile anchors on features.
"""

import json
from math import sqrt as sqrt
from itertools import product, chain
import os
from typing import List
import torch
from ops import AnchorGeneratorRegister, build_anchor


def tile_anchors(feature_map_sizes: List[int],
                 anchors: List[List[List[float]]]):
    assert len(feature_map_sizes) == len(anchors)
    tiled_anchors = list()
    for feature_size, anchor_shapes in zip(feature_map_sizes, anchors):
        for i, j in product(range(feature_size), repeat=2):
            # unit center x,y
            cx, cy = (j + 0.5) / feature_size, (i + 0.5) / feature_size
            for shape in anchor_shapes:
                tiled_anchors.append([cx, cy, shape[0], shape[1]])
    return tiled_anchors


def divide_list_to(input_list: List, sub_part_num: List[int]):
    assert len(input_list) == sum(sub_part_num)
    iter_list = iter(input_list)
    res = list()
    for sub_len in range(len(sub_part_num)):
        res.append([next(iter_list) for _ in range(sub_len)])
    return res


def generate_anchors_by_aspect_ratios_and_sizes(aspect_ratios: List[List[float]],
                                                base_sizes: List[float]) -> List[List[float]]:
    """
    Generate anchor shapes according to the input aspect ratios and sizes.
    The `aspect ratio` is defined by `height over width` and the `size` refers to `width`.
    Args:
        all_aspect_ratios: a list contains all aspect ratios (h/w).
        base_sizes: a list contains all normalized widths.
    Example:
        all_aspect_ratios: [[2., 0.5, 3.], [1., 1.5, 0.5]]
        base_sizes: [0.3, 0.5]

        generated_anchors:
            [[0.6, 0.3], [0.15, 0.3], [0.9, 0.3], [0.5, 0.5], [0.75, 0.5], [0.25, 0.5]]
            Total length: 3 * 2 = 6 anchors
    """
    assert len(aspect_ratios) == len(base_sizes)
    return list(chain(map(lambda bs, ars: [[ar * bs, bs] for ar in ars],
                          zip(base_sizes, aspect_ratios))))


def get_anchors_from_all_sizes(feature_map_sizes: List[int],
                               all_aspect_ratios: List[List[List[float]]],
                               all_base_sizes: List[List[float]]):
    assert len(feature_map_sizes) == len(all_aspect_ratios) == len(all_base_sizes)
    all_anchor_shapes = [generate_anchors_by_aspect_ratios_and_sizes(aspect_ratios, base_sizes)
                         for aspect_ratios, base_sizes in zip(all_aspect_ratios, all_base_sizes)]
    return tile_anchors(feature_map_sizes, all_anchor_shapes)


@AnchorGeneratorRegister('load_from_json')
def load_anchor(cfg):
    with open(cfg.ANCHOR.SAVE_PATH, 'r') as fp:
        anchors = json.load(fp)
    return anchors


@AnchorGeneratorRegister('calculate')
def calculate_anchor(cfg):
    return


class Anchor(object):
    def __init__(self, cfg):
        super(Anchor, self).__init__()
        self.image_size = cfg.MODEL.INPUT_SIZE
        self.variance = cfg.ENCODE.VARIANCE
        self.feature_maps = cfg.MODEL.FEATURE_SIZE
        self.min_sizes = cfg.ANCHOR.MIN_SIZES
        self.max_sizes = cfg.ANCHOR.MAX_SIZES
        self.steps = cfg.ANCHOR.STEPS
        self.aspect_ratios = cfg.ANCHOR.ASPECT_RATIOS
        self.clip = bool(cfg.ANCHOR.CLIP)
        self.size_step = cfg.ANCHOR.SIZE_STEP

        self.gen_type = cfg.ANCHOR.GEN_METHOD
        self.anchor_root = cfg.ANCHOR.SAVE_PATH
        self.anchor_shapes = cfg.ANCHOR.NUM_PER_LEVEL

        self.config = cfg

    @torch.no_grad()
    def forward(self):
        anchors = build_anchor(self.config)
        anchors_per_level = divide_list_to(anchors, self.anchor_shapes)
        mean = tile_anchors(self.feature_maps, anchors_per_level)
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

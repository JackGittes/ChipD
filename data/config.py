# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

airbus = {
    # -----------------------------------------
    # Dataset specification
    # -----------------------------------------
    'name': 'Airbus',
    'data_root': 'H:/Dataset/ShipDET-Horizon',
    'num_classes': 2,

    # -----------------------------------------
    # Optimizer specification
    # -----------------------------------------
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,

    'min_dim': 256,  # input image size

    # ------------------------------------------
    # Anchor specification
    # ------------------------------------------
    'feature_maps': [16, 16, 8],                # output feature map size
    'feature_map_channels': [48, 96, 96],       # output feature map channels

    'anchor_dir': 'export',
    'anchor_gen': 'auto',
    'anchor_per_level': [30, 30, 30],

    'steps': [16, 32, 64],
    'size_step': 3,
    'min_sizes': [16, 32, 64],
    'max_sizes': [32, 64, 128],
    'aspect_ratios': [[2, 3, 4, 5],
                      [2, 3, 4, 5],
                      [2, 3, 4, 5]],
    'variance': [0.1, 0.2],
    'clip': True,
}
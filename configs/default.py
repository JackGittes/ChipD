"""
Author: Zhao Mingxin
Date: 01/01/2022

Simplified SSD-based detector configuration template.
"""

from typing import Union


class ConfigTemplate:
    _SPLITTER_LEN = 50

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        leading = ['']
        self.__load_string(leading)
        splitter = "=" * ConfigTemplate._SPLITTER_LEN
        return splitter + "\n" + leading[0] + splitter

    def _yaml(self) -> str:
        leading = ['---\n']
        self.__load_string(leading)
        return leading[0]

    def __load_string(self, leading=[''], level: int = 0):
        for attr in self.__dir__():
            if not attr.startswith('_'):
                true_attr = getattr(self, attr)
                if isinstance(true_attr, ConfigTemplate):
                    leading[0] += "  " * level + attr + ': ' + '\n'
                    true_attr.__load_string(leading, level + 1)
                else:
                    leading[0] += "  " * level + attr + ': ' + str(true_attr) + '\n'
    __str__ = __repr__


def forced_float(float_num: Union[float, str]):
    return "{:f}".format(float(float_num))


class FormattedFloat(float):

    def __repr__(self) -> str:
        return forced_float(self.__float__())

    def __add__(self, __x: float) -> float:
        return FormattedFloat(super().__add__(__x))

    def __sub__(self, __x: float) -> float:
        return FormattedFloat(super().__sub__(__x))

    def __mul__(self, __x: float) -> float:
        return FormattedFloat(super().__mul__(__x))

    def __truediv__(self, __x: float) -> float:
        return FormattedFloat(super().__truediv__(__x))

    __str__ = __repr__


_C = ConfigTemplate()
_C.NAME = 'SSD'

# --------------------------------------------
# optimizer and training config
# --------------------------------------------
_C.TRAINING = ConfigTemplate()
_C.TRAINING.START_ITER = 0
_C.TRAINING.MAX_ITER = 31000
_C.TRAINING.BATCH_SIZE = 128
_C.TRAINING.NUM_WORKERS = 8
_C.TRAINING.RESUME = 0
_C.TRAINING.RESUME_WEIGHT = ''

_C.OPTIMIZER = ConfigTemplate()
_C.OPTIMIZER.NAME = "SGD"
_C.OPTIMIZER.LR = FormattedFloat(8e-3)
_C.OPTIMIZER.WEIGHT_DECAY = FormattedFloat(5e-4)
_C.OPTIMIZER.MOMENTUM = FormattedFloat(0.99)
_C.OPTIMIZER.BETAS = [FormattedFloat(0.9), FormattedFloat(0.999)]

_C.SCHEDULER = ConfigTemplate()
_C.SCHEDULER.NAME = 'multistep'

_C.SCHEDULER.MULTI_STEP = ConfigTemplate()
_C.SCHEDULER.MULTI_STEP.LR_STEPS = [20000, 25000, 30000]
_C.SCHEDULER.MULTI_STEP.GAMMA = FormattedFloat(0.1)

_C.SCHEDULER.COSINE_WARMUP = ConfigTemplate()
_C.SCHEDULER.COSINE_WARMUP.LENGTH = _C.TRAINING.MAX_ITER
_C.SCHEDULER.COSINE_WARMUP.WARMUP_STEPS = _C.TRAINING.MAX_ITER // 100
_C.SCHEDULER.COSINE_WARMUP.MAX_LR = _C.OPTIMIZER.LR
_C.SCHEDULER.COSINE_WARMUP.MIN_LR = _C.OPTIMIZER.LR / 100.

# --------------------------------------------
# logger configuration
# --------------------------------------------
_C.LOGGER = ConfigTemplate()
_C.LOGGER.ROOT = 'log'
_C.LOGGER.SAVE_INTERVAL = 100

# --------------------------------------------
# dataset config
# --------------------------------------------
_C.DATASET = ConfigTemplate()

"""
NAME: dataset name should be a string.
ROOT: dataset root
"""
_C.DATASET.NAME = "Airbus"
_C.DATASET.ROOT = "/home/username/dataset"
_C.DATASET.NUM_CLASSES = 2
_C.DATASET.MEANS = [FormattedFloat(116.28),
                    FormattedFloat(103.53),
                    FormattedFloat(123.675)]  # channel means in BGR order
_C.DATASET.STD = [FormattedFloat(57.12),
                  FormattedFloat(57.375),
                  FormattedFloat(58.395)]  # channel standard deviation in BGR order

# --------------------------------------------
# model config
# --------------------------------------------
_C.MODEL = ConfigTemplate()
_C.MODEL.NAME = 'SSD'

_C.MODEL.INPUT_SIZE = 256
_C.MODEL.FEATURE_SIZE = [16, 16, 16]
_C.MODEL.FEATURE_CHANNELS = [72, 144, 144]

_C.MODEL.BACKBONE = ConfigTemplate()
_C.MODEL.BACKBONE.NAME = 'build_mobilenetv2_backbone'

_C.MODEL.HEAD = ConfigTemplate()
_C.MODEL.HEAD.NAME = 'light'

# --------------------------------------------
# anchor config
# --------------------------------------------
_C.ANCHOR = ConfigTemplate()
_C.ANCHOR.NUM_PER_LEVEL = [24, 24, 24]
_C.ANCHOR.GEN_METHOD = 'auto'
_C.ANCHOR.SAVE_PATH = 'export'

# ## Used for manual anchor generation method ##
_C.ANCHOR.STEPS = [16, 16, 16]
_C.ANCHOR.SIZE_STEP = 3
_C.ANCHOR.MIN_SIZES = [16, 32, 64]
_C.ANCHOR.MAX_SIZES = [32, 64, 128]
_C.ANCHOR.ASPECT_RATIOS = [[2, 3, 4],
                           [2, 3, 4],
                           [2, 3, 4]]
################################################

_C.ANCHOR.CLIP = 1
_C.ANCHOR.VISUAL_PATH = 'visual'  # path for saving anchor visualization results

# --------------------------------------------
# ssd related config
# --------------------------------------------
_C.ENCODE = ConfigTemplate()
_C.ENCODE.USE = 1
_C.ENCODE.METHOD = "sqrt"
_C.ENCODE.VARIANCE = [FormattedFloat(0.1), FormattedFloat(0.2)]

_C.MATCHER = ConfigTemplate()
_C.MATCHER.NAME = "simple"


# ---------------------------------------------
# loss configuration
# ---------------------------------------------
_C.LOSS = ConfigTemplate()

_C.LOSS.NAME = 'ssd_loss'

_C.LOSS.CONF_LOSS = ConfigTemplate()
_C.LOSS.CONF_LOSS.NAME = 'focal'

_C.LOSS.CONF_LOSS.FOCAL = ConfigTemplate()
_C.LOSS.CONF_LOSS.FOCAL.GAMMA = FormattedFloat(2.)
_C.LOSS.CONF_LOSS.FOCAL.ALPHA = FormattedFloat(0.25)

_C.LOSS.LOC_LOSS = ConfigTemplate()
_C.LOSS.LOC_LOSS.NAME = 'smooth_l1'

_C.LOSS.IOU_LOSS = ConfigTemplate()
_C.LOSS.IOU_LOSS.NAME = 'iou'

_C.LOSS.NEG_POS_RATIO = 3
_C.LOSS.POS_OVERLAP_THRESH = FormattedFloat(0.5)
_C.LOSS.LOC_WEIGHT = FormattedFloat(1.0)
_C.LOSS.CONF_WEIGHT = FormattedFloat(1.0)

# ----------------------------------------------
# detection config
# ----------------------------------------------
_C.DETECT = ConfigTemplate()
_C.DETECT.CONFIDENCE_THRESH = FormattedFloat(0.25)
_C.DETECT.NMS_THRESH = FormattedFloat(0.5)
_C.DETECT.TOP_K = 200
_C.DETECT.BACKGROUND_LABEL = 0


__yaml_str = _C._yaml()
with open('configs/default.yml', 'w') as fp:
    fp.write(__yaml_str)

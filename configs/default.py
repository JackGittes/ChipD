"""
Author: Zhao Mingxin
Date: 01/01/2022

Simplified SSD-based detector configuration template.
"""


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
        leading = ['']
        self.__load_string(leading)
        return "---\n" + leading[0]

    def __load_string(self, leading=[''], level: int = 0):
        for attr in self.__dir__():
            if not attr.startswith('_'):
                true_attr = getattr(self, attr)
                if isinstance(true_attr, ConfigTemplate):
                    leading[0] += attr + ': ' + '\n'
                    true_attr.__load_string(leading, level + 1)
                else:
                    leading[0] += "  " * level + attr + ': ' + str(true_attr) + '\n'


_C = ConfigTemplate()
_C.NAME = 'Airbus'

# --------------------------------------------
# optimizer and training config
# --------------------------------------------
_allowed_optim = ('AdamW', 'SGD', 'ADAM')

_C.TRAINING = ConfigTemplate()
_C.TRAINING.START_ITER = 0
_C.TRAINING.MAX_ITER = 31000
_C.TRAINING.BATCH_SIZE = 256
_C.TRAINING.NUM_WORKERS = 8

_C.OPTIMIZER = ConfigTemplate()
_C.OPTIMIZER.NAME = "AdamW"
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 5e-4

_C.SCHEDULER = ConfigTemplate()
_C.SCHEDULER.NAME = 'MultiStep'
_C.SCHEDULER.LR_STEPS = [20000, 25000, 30000]
_C.SCHEDULER.GAMMA = 0.1

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
_C.DATASET.MEANS = [116.28, 103.53, 123.675]  # channel means in BGR order
_C.DATASET.STD = [57.12, 57.375, 58.395]  # channel standard deviation in BGR order

# --------------------------------------------
# model config
# --------------------------------------------
_C.MODEL = ConfigTemplate()
_C.MODEL.INPUT_SIZE = 256
_C.MODEL.FEATURE_SIZE = [16, 16, 16]
_C.MODEL.FEATURE_CHANNELS = [72, 144, 144]

# --------------------------------------------
# anchor config
# --------------------------------------------
_allowed_gen_method = ('auto', 'manual')

_C.ANCHOR = ConfigTemplate()
_C.ANCHOR.NUM_PER_LEVEL = [24, 24, 24]
_C.ANCHOR.GEN_METHOD = 'auto'
_C.ANCHOR.SAVE_PATH = 'export'

_C.ANCHOR.STEPS = [16, 16, 16]
_C.ANCHOR.SIZE_STEP = 3
_C.ANCHOR.MIN_SIZES = [16, 32, 64]
_C.ANCHOR.MAX_SIZES = [32, 64, 128]
_C.ANCHOR.ASPECT_RATIOS = [[2, 3, 4],
                           [2, 3, 4],
                           [2, 3, 4]]
_C.ANCHOR.CLIP = 1

# --------------------------------------------
# ssd related config
# --------------------------------------------
_allowed_encode = ('log', 'sqrt')

_C.ENCODE = ConfigTemplate()
_C.ENCODE.METHOD = "log"
_C.ENCODE.VARIANCE = [0.1, 0.2]

_C.MATCHER = ConfigTemplate()
_C.MATCHER.NAME = "simple"


# ---------------------------------------------
# loss configuration
# ---------------------------------------------
_C.LOSS = ConfigTemplate()
_C.LOSS.NAME = None
_C.LOSS.NEG_POS_RATIO = 3
_C.LOSS.POS_OVERLAP_THRESH = 0.5
_C.LOSS.LOC_WEIGHT = 1.0
_C.LOSS.CONF_WEIGHT = 1.0

# ----------------------------------------------
# detection config
# ----------------------------------------------
_C.DETECT = ConfigTemplate()
_C.DETECT.CONFIDENCE_THRESH = 0.25
_C.DETECT.NMS_THRESH = 0.5
_C.DETECT.TOP_K = 200
_C.DETECT.BACKGROUND_LABEL = 0


__yaml_str = _C._yaml()
with open('configs/default.yml', 'w') as fp:
    fp.write(__yaml_str)

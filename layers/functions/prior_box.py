from __future__ import division
import json
from math import sqrt as sqrt
from itertools import product as product
import os
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        self.size_step = cfg['size_step']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        self.gen_type = cfg['anchor_gen']
        assert self.gen_type in ['auto', 'manual']
        self.anchor_root = cfg['anchor_dir']

        if self.gen_type == 'auto':
            assert os.path.isdir(self.anchor_root), 'Anchor file path not found.'
            assert os.path.isfile(os.path.join(self.anchor_root, 'anchor.json'))
        self.anchor_shapes = cfg['anchor_per_level']

    @torch.no_grad()
    def forward(self):
        if self.gen_type == 'auto':
            mean = self.auto()
        else:
            mean = self.manual()

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    @torch.no_grad()
    def auto(self):
        mean = list()
        with open(os.path.join(self.anchor_root, 'anchor.json'), 'r') as fp:
            anchors = json.load(fp)
        assert len(anchors) == sum(self.anchor_shapes)
        anchors_iter = iter(anchors)
        for k, f in enumerate(self.feature_maps):
            # prepare anchor shapes for current level
            shapes = []
            for _ in range(self.anchor_shapes[k]):
                shapes.append(next(anchors_iter))
            # iterate all positions on feature maps
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                for shape in shapes:
                    mean += [cx, cy, shape[0], shape[1]]
        return mean

    @torch.no_grad()
    def manual(self):
        mean = list()
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                all_sizes = [self.min_sizes[k] + i * (self.max_sizes[k] - self.min_sizes[k]) / self.size_step
                             for i in range(self.size_step)]
                for s_k in all_sizes:
                    # aspect_ratio: 1
                    # rel size: min_size
                    # s_k = self.min_sizes[k] / self.image_size
                    s_k /= self.image_size
                    mean += [cx, cy, s_k, s_k]
                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]
                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                        mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        return mean


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from data.config import airbus
    pb = PriorBox(airbus)
    print(pb.forward())
    print(pb.forward().shape)

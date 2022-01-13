import json
import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np


AIRBUS_CLASSES = ('ship')


class AirbusDetection(data.Dataset):
    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_name='Airbus',
                 train=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        assert isinstance(train, bool), "Dataset status must be a boolean."

        anno_file = 'train.json' if train else 'val.json'
        with open(osp.join(root, anno_file)) as fp:
            anno_dict: dict = json.load(fp)

        # all targets are categorized to 0 for ship.
        self.ids = list()
        for k, v in anno_dict.items():
            box_lbs = [item + [0] for item in v]
            self.ids.append([k, box_lbs])

    def __getitem__(self, index):
        im, gt, _, _ = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        image_name, target = self.ids[index]
        target = [[item[1] - item[3] / 2.,
                   item[0] - item[2] / 2.,
                   item[1] + item[3] / 2.,
                   item[0] + item[2] / 2.,
                   item[4]] for item in target]

        img_full_path = osp.join(self.root, 'AllImages', image_name)
        img: np.ndarray = cv2.imread(img_full_path)
        height, width, _ = img.shape

        # normalize target
        target = [[item[0] / width, item[1] / height,
                   item[2] / width, item[3] / height, item[4]]
                  for item in target]

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        image_name, _ = self.ids[index]
        img_full_path = osp.join(self.root, 'AllImages', image_name)
        return cv2.imread(img_full_path, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        image_name, target = self.ids[index]
        target = [[item[1] - item[3] / 2.,
                   item[0] - item[2] / 2.,
                   item[1] + item[3] / 2.,
                   item[0] + item[2] / 2.,
                   item[4]] for item in target]
        return image_name, target

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

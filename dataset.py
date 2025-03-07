from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn.functional as F
import os
import numpy as np


def load_info(info_file_path):
    """
    directory/
            ├── class_x
            │   ├── xxx.npy
            │   ├── xxy.npy
            │   └── ...
            └── class_y
                ├── 123.npy
                ├── nsdf3.npy
                └── ...
                └── asd932_.npy

    Args:
        info_file_path (_type_): _description_

    Returns:
        boxes info: tensor
    """
    classes = sorted(
        entry.name for entry in os.scandir(info_file_path) if entry.is_dir()
    )
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {info_file_path}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(info_file_path, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index, fname
                instances.append(item)

                if target_class not in available_classes:
                    available_classes.add(target_class)

    return instances


class CustomImageFolder(ImageFolder):
    def __init__(self, root, info_file_path, trans_flip=False, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.info = load_info(info_file_path)
        self.trans_flip=trans_flip
        self.p=0.5

    def __getitem__(self, index):
        path, target = self.samples[index]

        img = self.loader(path)

        info, target_info, fname = self.info[index]
        box_info = np.load(info)
        assert target == target_info

        if self.trans_flip:
            if torch.rand(1) < self.p:
                img = transforms.functional.hflip(img)
                box_info = np.flip(box_info,axis=1).copy()
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, box_info, fname

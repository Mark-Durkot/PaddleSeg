# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
from paddleseg.datasets.dataset import Dataset

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
GT_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


@manager.DATASETS.add_component
class VDD(Dataset):
    """
    VDD (custom) semantic segmentation dataset.

    Expected layout under ``dataset_root``::

        dataset_root/
        ├── train/
        │   ├── src/     # RGB (or BGR-read) input images
        │   └── gt/      # single-channel **grayscale** masks (not RGB):
        │                # pixel value **is** the class id (0…6) in a normal
        │                # uint8 file (0–255 range on disk; only 0–6 used).
        │                # They look almost black in a photo viewer; that is OK.
        ├── val/
        │   ├── src/
        │   └── gt/
        └── test/
            ├── src/
            └── gt/

    Use one channel only for ``gt`` (e.g. PNG mode ``L``). Values above
    ``NUM_CLASSES - 1`` or ``ignore_index`` (255) are invalid unless you
    extend the task definition.

    Image and mask are paired by **basename without extension** (stem).
    Extensions may differ between ``src`` and ``gt`` (e.g. ``.jpg`` / ``.png``).

    Args:
        transforms (list): Transforms for the image (and label where applicable).
        dataset_root (str): Root directory of the VDD dataset.
        mode (str, optional): One of ``'train'``, ``'val'``, ``'test'``. Default: ``'train'``.
        edge (bool, optional): Whether to compute edge supervision. Default: False.
        src_subdir (str, optional): Name of the image folder under each split. Default: ``'src'``.
        gt_subdir (str, optional): Name of the label folder under each split. Default: ``'gt'``.
    """
    NUM_CLASSES = 7
    IGNORE_INDEX = 255
    IMG_CHANNELS = 3

    def __init__(self,
                 transforms,
                 dataset_root,
                 mode='train',
                 edge=False,
                 src_subdir='src',
                 gt_subdir='gt'):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = []
        mode = mode.lower()
        self.mode = mode
        self.edge = edge
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = self.IGNORE_INDEX
        self.img_channels = self.IMG_CHANNELS

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if not dataset_root or not os.path.isdir(dataset_root):
            raise FileNotFoundError(
                'dataset_root is missing or not a directory: {}'.format(
                    dataset_root))

        src_dir = os.path.join(dataset_root, mode, src_subdir)
        gt_dir = os.path.join(dataset_root, mode, gt_subdir)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(
                'VDD image directory not found: {}'.format(src_dir))
        if not os.path.isdir(gt_dir):
            raise FileNotFoundError(
                'VDD label directory not found: {}'.format(gt_dir))

        src_paths = []
        for path in glob.glob(os.path.join(src_dir, '*')):
            if os.path.isfile(path) and path.lower().endswith(IMG_EXTS):
                src_paths.append(path)
        src_paths.sort()

        gt_by_stem = {}
        for path in glob.glob(os.path.join(gt_dir, '*')):
            if os.path.isfile(path) and path.lower().endswith(GT_EXTS):
                stem = os.path.splitext(os.path.basename(path))[0]
                gt_by_stem[stem] = path

        missing = []
        for img_path in src_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = gt_by_stem.get(stem)
            if label_path is None:
                missing.append(stem)
            else:
                self.file_list.append([img_path, label_path])

        if missing:
            raise FileNotFoundError(
                'No matching ground-truth file for {} source image(s) '
                '(match by stem in {}). First missing stem(s): {}'.format(
                    len(missing), gt_dir, missing[:5]))

        if len(self.file_list) == 0:
            raise ValueError(
                'No image–label pairs found under {} / {}'.format(
                    src_dir, gt_dir))

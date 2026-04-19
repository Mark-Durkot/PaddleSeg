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

"""
VDD class names and helpers for **visualization only**.

Ground-truth masks for training are **single-channel** (e.g. PNG mode ``L``).
Pixel values are **class ids** ``0 .. NUM_CLASSES-1`` (0 = other … 6 = water),
stored in a normal 8-bit file (values can be 0–255, but you only use 0–6).
Because the largest class id is 6, masks look **almost black** in Preview,
Finder, or a photo viewer—that is expected; the file is still correct.

To inspect labels visually: use ``get_color_map()`` with
``tools/data/gray2pseudo_color.py``, or ``stretch_label_ids_for_preview()`` for
a stretched single-channel PNG (do **not** train on stretched masks).

``PALETTE`` entries are **R = G = B** grays, one triple per **class id**, spaced
across 0–255 so pseudo-color previews are easy to read. That mapping is **not**
the same as raw mask bytes (which stay 0–6).
"""

import numpy as np

NUM_CLASSES = 7

# (id, name) — id matches mask pixel intensity
CLASSES = [
    (0, 'other'),
    (1, 'wall'),
    (2, 'road'),
    (3, 'vegetation'),
    (4, 'vehicle'),
    (5, 'roof'),
    (6, 'water'),
]

# Grayscale as RGB triplets (R=G=B): class id 0..6 mapped to evenly spaced
# 0..255 so previews stay distinguishable; same information as 1-D labels.
def _vdd_gray_palette(num_classes):
    if num_classes <= 1:
        return [[0, 0, 0]]
    return [
        [int(round(255 * k / (num_classes - 1)))] * 3
        for k in range(num_classes)
    ]


PALETTE = _vdd_gray_palette(NUM_CLASSES)


def get_color_map(flat=True):
    """
    Args:
        flat (bool): If True, return a flat list [R0,G0,B0, R1,G1,B1, ...]
            compatible with ``gray2pseudo_color.py`` expectations.
    """
    if len(PALETTE) != NUM_CLASSES:
        raise RuntimeError('PALETTE length must match NUM_CLASSES')
    if flat:
        out = []
        for rgb in PALETTE:
            out.extend(rgb)
        return out
    return [list(x) for x in PALETTE]


def stretch_label_ids_for_preview(label, max_class_id=None, ignore_index=255):
    """
    **Viewing / QC only.** Map label ids ``0..max_class_id`` linearly to
    ``0..255`` so a grayscale PNG is visible in ordinary image viewers.
    Pixels equal to ``ignore_index`` are left unchanged.

    Do **not** use the returned array as training labels.

    Args:
        label: ``numpy.ndarray`` (H, W) uint8 (or int) mask, or path ``str``
            to a single-channel image file (read via ``numpy.asarray(PIL)``).
        max_class_id (int, optional): Highest valid class id. Default:
            ``NUM_CLASSES - 1``.
        ignore_index (int): Pixels to pass through unscaled (default 255).

    Returns:
        ``numpy.ndarray`` uint8, shape (H, W).
    """
    if max_class_id is None:
        max_class_id = NUM_CLASSES - 1
    if isinstance(label, str):
        from PIL import Image
        arr = np.asarray(Image.open(label))
    else:
        arr = np.asarray(label)
    if arr.ndim != 2:
        raise ValueError(
            'Expected single-channel mask (H, W); got shape {}'.format(
                arr.shape))
    out = arr.astype(np.uint8, copy=True)
    if max_class_id <= 0:
        return out
    valid = (out <= max_class_id)
    scaled = (out[valid].astype(np.float32) * (255.0 / max_class_id)).round()
    out[valid] = scaled.clip(0, 255).astype(np.uint8)
    return out

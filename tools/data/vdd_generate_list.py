#!/usr/bin/env python3
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
Generate PaddleSeg ``Dataset``-style list files for VDD layout.

Writes paths relative to ``dataset_root``, one pair per line::
    train/src/foo.jpg train/gt/foo.png

``gt`` files must be **grayscale** label maps (one channel): pixel intensity
is the class id (0–6 for VDD), not RGB class colors.

Default output files in ``dataset_root``:
    vdd_train_list.txt, vdd_val_list.txt, vdd_test_list.txt

You can then train with ``type: Dataset``, ``num_classes: 7``, and
``train_path`` / ``val_path`` / ``test_path`` pointing to these files.
"""

import argparse
import glob
import os

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
GT_EXTS = IMG_EXTS


def parse_args():
    p = argparse.ArgumentParser(description='Generate VDD file lists for PaddleSeg Dataset')
    p.add_argument(
        'dataset_root',
        type=str,
        help='Root folder containing train/, val/, test/ each with src/ and gt/')
    p.add_argument(
        '--src-subdir',
        default='src',
        help='Image subdirectory under each split (default: src)')
    p.add_argument(
        '--gt-subdir',
        default='gt',
        help='Label subdirectory under each split (default: gt)')
    p.add_argument(
        '--out-dir',
        default=None,
        help='Where to write list files (default: dataset_root)')
    p.add_argument(
        '--separator',
        default=' ',
        help='Separator between image path and label path (default: space)')
    return p.parse_args()


def collect_pairs(root, split, src_sub, gt_sub):
    src_dir = os.path.join(root, split, src_sub)
    gt_dir = os.path.join(root, split, gt_sub)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError('Missing directory: {}'.format(src_dir))
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError('Missing directory: {}'.format(gt_dir))

    gt_by_stem = {}
    for path in glob.glob(os.path.join(gt_dir, '*')):
        if os.path.isfile(path) and path.lower().endswith(GT_EXTS):
            stem = os.path.splitext(os.path.basename(path))[0]
            gt_by_stem[stem] = path

    lines = []
    for path in sorted(glob.glob(os.path.join(src_dir, '*'))):
        if not (os.path.isfile(path) and path.lower().endswith(IMG_EXTS)):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem not in gt_by_stem:
            raise FileNotFoundError(
                'No GT for {} (stem {!r} not found under {})'.format(
                    path, stem, gt_dir))
        rel_img = os.path.relpath(path, root).replace(os.sep, '/')
        rel_gt = os.path.relpath(gt_by_stem[stem], root).replace(os.sep, '/')
        lines.append((rel_img, rel_gt))
    return lines


def main():
    args = parse_args()
    root = os.path.abspath(args.dataset_root)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else root
    os.makedirs(out_dir, exist_ok=True)

    for split, out_name in [('train', 'vdd_train_list.txt'),
                            ('val', 'vdd_val_list.txt'),
                            ('test', 'vdd_test_list.txt')]:
        pairs = collect_pairs(root, split, args.src_subdir, args.gt_subdir)
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, 'w') as f:
            for rel_img, rel_gt in pairs:
                f.write('{}{}{}\n'.format(rel_img, args.separator, rel_gt))
        print('Wrote {} ({} pairs)'.format(out_path, len(pairs)))


if __name__ == '__main__':
    main()

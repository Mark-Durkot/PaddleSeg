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
Batch inference for PPLiteSeg (STDC1 backbone, VDD / 7 classes) with per-image
latency logging and a summary plot. Saved masks and overlays include a legend
(class id and name, with swatch colors matching the pseudo-color map).

Example:
  python tools/infer_pp_liteseg_stdc1_vdd.py \\
    --test_img_path /path/to/images_or_list \\
    --model_params_path /path/to/model.pdparams \\
    --output_path /path/to/out
"""

import argparse
import csv
import importlib.util
import os
import time

import cv2
import numpy as np
import paddle

from paddleseg.core import infer
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.transforms import Compose
from paddleseg import utils
from paddleseg.utils import get_image_list, logger, visualize


DEFAULT_CONFIG = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), '..',
        'configs/pp_liteseg/pp_liteseg_stdc1_vdd_512x512_160k.yml'))


def _vdd_class_definitions():
    """Load (class_id, name) from tools/data/vdd_labels.py without requiring tools as a package."""
    path = os.path.join(os.path.dirname(__file__), 'data', 'vdd_labels.py')
    spec = importlib.util.spec_from_file_location('vdd_labels', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.CLASSES)


def add_class_legend_bgr(im_bgr, flat_color_map, class_rows):
    """
    Draw a class id + name legend on a BGR image using colors from flat_color_map
    (same layout as paddleseg visualize: RGB triplets per class index).
    """
    if im_bgr is None or im_bgr.size == 0:
        return im_bgr
    h, w = im_bgr.shape[:2]
    margin = 8
    line_h = 22
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    max_tw = 0
    for cid, name in class_rows:
        txt = '{}: {}'.format(cid, name)
        (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)
        max_tw = max(max_tw, tw)
    sw = 16
    legend_w = min(w - 2 * margin, pad * 2 + sw + 6 + max_tw)
    legend_h = pad * 2 + len(class_rows) * line_h
    x0 = margin
    y0 = max(margin, h - legend_h - margin)

    cv2.rectangle(
        im_bgr, (x0, y0), (x0 + legend_w - 1, y0 + legend_h - 1), (252, 252, 252), -1)
    cv2.rectangle(
        im_bgr, (x0, y0), (x0 + legend_w - 1, y0 + legend_h - 1), (55, 55, 55), 1)

    y_text = y0 + pad + 15
    for cid, name in class_rows:
        r = int(flat_color_map[cid * 3])
        g = int(flat_color_map[cid * 3 + 1])
        b = int(flat_color_map[cid * 3 + 2])
        color_bgr = (b, g, r)
        xs = x0 + pad
        cv2.rectangle(im_bgr, (xs, y_text - 13), (xs + sw - 1, y_text + 3), color_bgr, -1)
        cv2.rectangle(im_bgr, (xs, y_text - 13), (xs + sw - 1, y_text + 3), (30, 30, 30), 1)
        txt = '{}: {}'.format(cid, name)
        cv2.putText(
            im_bgr,
            txt, (xs + sw + 5, y_text),
            font,
            font_scale, (25, 25, 25),
            thickness,
            lineType=cv2.LINE_AA)
        y_text += line_h
    return im_bgr


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if sub_dir and not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def _sync_device():
    """Wait for GPU kernels to finish so timing includes device work."""
    dev = paddle.get_device()
    if dev == 'cpu':
        return
    if dev.startswith('gpu'):
        try:
            paddle.device.cuda.synchronize()
        except Exception:
            try:
                paddle.device.synchronize()
            except Exception:
                pass


def preprocess(im_path, transforms):
    data = {'img': im_path}
    data = transforms(data)
    data['img'] = data['img'][np.newaxis, ...]
    data['img'] = paddle.to_tensor(data['img'])
    return data


def merge_test_config(cfg, args):
    test_config = cfg.test_config
    test_config = {k: v for k, v in test_config.items()}
    if 'aug_eval' in test_config:
        test_config.pop('aug_eval')
    if 'auc_roc' in test_config:
        test_config.pop('auc_roc')
    if args.aug_pred:
        test_config['aug_pred'] = True
        test_config['scales'] = args.scales
        test_config['flip_horizontal'] = args.flip_horizontal
        test_config['flip_vertical'] = args.flip_vertical
    if args.is_slide:
        test_config['is_slide'] = True
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride
    if args.custom_color:
        test_config['custom_color'] = args.custom_color
    if args.use_multilabel:
        test_config['use_multilabel'] = True
    return test_config


def parse_args():
    p = argparse.ArgumentParser(
        description='PPLiteSeg STDC1 VDD inference with masks, overlays, and latency plot.')
    p.add_argument(
        '--test_img_path',
        required=True,
        help='Image file, directory of images, or list file (same rules as tools/predict.py).')
    p.add_argument(
        '--model_params_path',
        required=True,
        help='Trained weights (.pdparams).')
    p.add_argument(
        '--output_path',
        required=True,
        help='Directory for masks/, overlays/, timing plot, and CSV.')
    p.add_argument(
        '--config',
        type=str,
        default=DEFAULT_CONFIG,
        help='Training yaml (default: pp_liteseg_stdc1_vdd_512x512_160k.yml).')
    p.add_argument('--device', default='gpu', choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'])
    p.add_argument('--device_id', default=0, type=int)
    p.add_argument('--warmup', default=1, type=int, help='Number of warmup forwards before timing.')
    p.add_argument('--aug_pred', action='store_true')
    p.add_argument('--scales', nargs='+', type=float, default=[1.0])
    p.add_argument('--flip_horizontal', action='store_true')
    p.add_argument('--flip_vertical', action='store_true')
    p.add_argument('--is_slide', action='store_true')
    p.add_argument('--crop_size', nargs=2, type=int, default=None)
    p.add_argument('--stride', nargs=2, type=int, default=None)
    p.add_argument('--custom_color', nargs='+', type=int, default=None)
    p.add_argument('--use_multilabel', action='store_true', default=False)
    return p.parse_args()


def _run_forward(model, data, tc):
    if tc.get('aug_pred'):
        return infer.aug_inference(
            model,
            data['img'],
            trans_info=data['trans_info'],
            scales=tc.get('scales', 1.0),
            flip_horizontal=tc.get('flip_horizontal', True),
            flip_vertical=tc.get('flip_vertical', False),
            is_slide=tc.get('is_slide', False),
            stride=tc.get('stride'),
            crop_size=tc.get('crop_size'),
            use_multilabel=tc.get('use_multilabel', False))
    return infer.inference(
        model,
        data['img'],
        trans_info=data['trans_info'],
        is_slide=tc.get('is_slide', False),
        stride=tc.get('stride'),
        crop_size=tc.get('crop_size'),
        use_multilabel=tc.get('use_multilabel', False))


def save_latency_plot(out_dir, names, times_ms):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            'matplotlib is not installed; skipped plot. '
            'Install matplotlib or use inference_times_ms.csv.')
        return

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.35), 5))
    x = np.arange(len(names))
    ax.bar(x, times_ms, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.set_xticks(x)
    short = [os.path.basename(n) if len(n) > 40 else n for n in names]
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Per-image inference time (model forward + restore to original size)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, 'inference_times.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info('Saved latency plot to {}'.format(plot_path))


def main():
    args = parse_args()
    cfg = Config(args.config)
    builder = SegBuilder(cfg)
    tc = merge_test_config(cfg, args)

    os.makedirs(args.output_path, exist_ok=True)
    masks_dir = os.path.join(args.output_path, 'masks')
    overlays_dir = os.path.join(args.output_path, 'overlays')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    device = args.device if args.device == 'cpu' else '{}:{}'.format(
        args.device, args.device_id)
    utils.set_device(device)

    model = builder.model
    utils.load_entire_model(model, args.model_params_path)
    model.eval()

    transforms = Compose(builder.val_transforms)
    image_list, image_dir = get_image_list(args.test_img_path)
    logger.info('Images to infer: {}'.format(len(image_list)))

    color_map = visualize.get_color_map_list(
        256, custom_color=tc.get('custom_color'))

    try:
        all_class_rows = _vdd_class_definitions()
    except Exception as e:
        logger.warning('Could not load tools/data/vdd_labels.py ({}); using built-in names.'.
                         format(e))
        all_class_rows = [(0, 'other'), (1, 'wall'), (2, 'road'), (3, 'vegetation'),
                          (4, 'vehicle'), (5, 'roof'), (6, 'water')]
    num_classes = cfg.model_cfg.get('num_classes', len(all_class_rows))
    class_rows = [row for row in all_class_rows if row[0] < num_classes]

    names = []
    times_ms = []

    with paddle.no_grad():
        # Warmup (same code path as timed runs)
        if image_list:
            for _ in range(max(0, args.warmup)):
                data = preprocess(image_list[0], transforms)
                _sync_device()
                _, _ = _run_forward(model, data, tc)
                _sync_device()

        for im_path in image_list:
            data = preprocess(im_path, transforms)
            _sync_device()
            t0 = time.perf_counter()
            pred, _ = _run_forward(model, data, tc)
            _sync_device()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file.startswith('/') or im_file.startswith('\\'):
                im_file = im_file[1:]

            use_ml = tc.get('use_multilabel', False)
            overlay = utils.visualize.visualize(
                im_path, pred, color_map, weight=0.6, use_multilabel=use_ml)
            if not use_ml and class_rows:
                add_class_legend_bgr(overlay, color_map, class_rows)
            overlay_path = os.path.join(overlays_dir, im_file)
            mkdir(overlay_path)
            cv2.imwrite(overlay_path, overlay)

            pred_mask = visualize.get_pseudo_color_map(
                pred, color_map, use_multilabel=use_ml)
            mask_path = os.path.join(
                masks_dir, os.path.splitext(im_file)[0] + '.png')
            mkdir(mask_path)
            if not use_ml and class_rows:
                mask_bgr = cv2.cvtColor(
                    np.asarray(pred_mask.convert('RGB')), cv2.COLOR_RGB2BGR)
                add_class_legend_bgr(mask_bgr, color_map, class_rows)
                cv2.imwrite(mask_path, mask_bgr)
            else:
                pred_mask.save(mask_path)

            names.append(im_path)
            times_ms.append(elapsed_ms)
            logger.info('{} : {:.2f} ms'.format(im_file, elapsed_ms))

    csv_path = os.path.join(args.output_path, 'inference_times_ms.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path', 'inference_time_ms'])
        for n, t in zip(names, times_ms):
            w.writerow([n, '{:.4f}'.format(t)])
    logger.info('Wrote {}'.format(csv_path))

    save_latency_plot(args.output_path, names, times_ms)

    logger.info('Masks: {}, overlays: {}'.format(masks_dir, overlays_dir))


if __name__ == '__main__':
    main()

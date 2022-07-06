# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import numpy as np
import itertools
import argparse

cls2catid = {
    "0": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
    "10": 11,
    "11": 13,
    "12": 14,
    "13": 15,
    "14": 16,
    "15": 17,
    "16": 18,
    "17": 19,
    "18": 20,
    "19": 21,
    "20": 22,
    "21": 23,
    "22": 24,
    "23": 25,
    "24": 27,
    "25": 28,
    "26": 31,
    "27": 32,
    "28": 33,
    "29": 34,
    "30": 35,
    "31": 36,
    "32": 37,
    "33": 38,
    "34": 39,
    "35": 40,
    "36": 41,
    "37": 42,
    "38": 43,
    "39": 44,
    "40": 46,
    "41": 47,
    "42": 48,
    "43": 49,
    "44": 50,
    "45": 51,
    "46": 52,
    "47": 53,
    "48": 54,
    "49": 55,
    "50": 56,
    "51": 57,
    "52": 58,
    "53": 59,
    "54": 60,
    "55": 61,
    "56": 62,
    "57": 63,
    "58": 64,
    "59": 65,
    "60": 67,
    "61": 70,
    "62": 72,
    "63": 73,
    "64": 74,
    "65": 75,
    "66": 76,
    "67": 77,
    "68": 78,
    "69": 79,
    "70": 80,
    "71": 81,
    "72": 82,
    "73": 84,
    "74": 85,
    "75": 86,
    "76": 87,
    "77": 88,
    "78": 89,
    "79": 90
}


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anno_file",
        type=str,
        default=None,
        help="Path of annotation file using coco format.",
        required=True)
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="Path of infer results using coco format.")
    parser.add_argument(
        "--txt_dir",
        type=str,
        default=None,
        help="Directory of infer results using txt format.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output files.")

    return parser


def draw_pr_curve(precision,
                  recall,
                  iou=0.5,
                  out_dir='pr_curve',
                  file_name='precision_recall_curve.jpg'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_path = os.path.join(out_dir, file_name)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print('Matplotlib not found, plaese install matplotlib.'
              'for example: `pip install matplotlib`.')
        raise e
    plt.cla()
    plt.figure('P-R Curve')
    plt.title('Precision/Recall Curve(IoU={})'.format(iou))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.plot(recall, precision)
    plt.savefig(output_path)


def cocoapi_eval(anno_file,
                 jsonfile,
                 style='bbox',
                 max_dets=(100, 300, 1000),
                 classwise=False,
                 sigmas=None,
                 use_area=True):
    """
    Args:
        anno_file (str): COCO annotations file.
        jsonfile (str): Evaluation json file, eg: bbox.json, mask.json.
        style (str): COCOeval style, can be `bbox` , `segm` , `proposal`, `keypoints` and `keypoints_crowd`.
        max_dets (tuple): COCO evaluation maxDets.
        classwise (bool): Whether per-category AP and draw P-R Curve or not.
        sigmas (nparray): keypoint labelling sigmas.
        use_area (bool): If gt annotations (eg. CrowdPose, AIC)
                         do not have 'area', please set use_area=False.
    """
    if style == 'keypoints_crowd':
        #please install xtcocotools==1.6
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval
    else:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(anno_file)
    print(">>> Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    elif style == 'keypoints_crowd':
        coco_eval = COCOeval(coco_gt, coco_dt, style, sigmas, use_area)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if classwise:
        # Compute per-category AP and PR curve
        try:
            from terminaltables import AsciiTable
        except Exception as e:
            print('terminaltables not found, plaese install terminaltables. '
                  'for example: `pip install terminaltables`.')
            raise e
        precisions = coco_eval.eval['precision']
        cat_ids = coco_gt.getCatIds()
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]
        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_gt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (str(nm["name"]), '{:0.3f}'.format(float(ap))))
            pr_array = precisions[0, :, idx, 0, 2]
            recall_array = np.arange(0.0, 1.01, 0.01)
            draw_pr_curve(
                pr_array,
                recall_array,
                out_dir=style + '_pr_curve',
                file_name='{}_precision_recall_curve.jpg'.format(nm["name"]))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print('Per-category of {} AP: \n{}'.format(style, table.table))
        print("per-category PR curve has output to {} folder.".format(
            style + '_pr_curve'))
    # flush coco evaluation result
    sys.stdout.flush()
    return coco_eval.stats


def json_load(file_path):
    with open(file_path) as f:
        out_list = json.load(f)
    return out_list


def json_save(out_list, file_path):
    with open(file_path, 'w') as f:
        json.dump(out_list, f)


def txt_load(file_path):
    with open(file_path) as f:
        out_list = f.readlines()
    return out_list


def format_txt2coco_results(txt_dir, cls2catid, save_dir='output'):
    """ txt_format: [cls_id, score, x1, y1, x2, y2] """
    print(">>> Start txt2coco...")
    coco_results = []
    count = 0
    for txt_file in os.listdir(txt_dir):
        if os.path.splitext(txt_file)[-1] != '.txt':
            continue
        txt_list = txt_load(os.path.join(txt_dir, txt_file))
        count += 1
        for line in txt_list:
            line = line.strip().split()
            image_id = int(os.path.splitext(txt_file)[0])
            coco_results.append({
                'image_id': image_id,
                'category_id': cls2catid[line[0]],
                'file_name': txt_file,
                'bbox': [
                    float(line[2]), float(line[3]),
                    float(line[4]) - float(line[2]),
                    float(line[5]) - float(line[3])
                ],  # xyxy -> xywh
                'score': float(line[1])
            })

    json_save(coco_results, os.path.join(save_dir, 'results.json'))
    print("txt2coco done!")
    return coco_results, os.path.join(save_dir, 'results.json')


def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')


if __name__ == '__main__':
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)

    assert FLAGS.json_file is not None or FLAGS.txt_dir is not None, \
        "--json_file or --txt_dir should be set"
    if FLAGS.json_file is not None:
        assert os.path.isfile(FLAGS.json_file), \
            f"{FLAGS.json_file} is not a file."
        eval_stats = cocoapi_eval(FLAGS.anno_file, FLAGS.json_file)
    elif FLAGS.txt_dir is not None:
        assert os.path.isdir(FLAGS.txt_dir), \
        f"{FLAGS.txt_dir} is not a directory."
        output_dir = os.path.abspath(FLAGS.output_dir)
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(output_dir)
        coco_results, json_file = format_txt2coco_results(
            FLAGS.txt_dir, cls2catid, FLAGS.output_dir)
        eval_stats = cocoapi_eval(FLAGS.anno_file, json_file)
    print("Done!")

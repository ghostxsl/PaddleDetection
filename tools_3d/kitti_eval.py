#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)


def parse_args():
    parser = argparse.ArgumentParser("KITTI mAP evaluation script")
    parser.add_argument(
        '--result_dir',
        type=str,
        default='./output_results',
        help='detection result directory to evaluate')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./kitti3d',
        help='KITTI dataset root directory')
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        help='evaluation split, default val')
    parser.add_argument(
        '--class_name',
        type=str,
        default='Car',
        help='evaluation class name, default Car')
    args = parser.parse_args()
    return args


def kitti_eval():

    args = parse_args()

    label_dir = os.path.join(args.data_dir, 'training/label_2')
    split_file = os.path.join(args.data_dir, 'ImageSets',
                              '{}.txt'.format(args.split))
    final_output_dir = args.result_dir
    name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

    from kitti_object_eval_python.evaluate import evaluate as kitti_evaluate
    ap_result_str, ap_dict = kitti_evaluate(
        label_dir,
        final_output_dir,
        label_split_file=split_file,
        current_class=name_to_class[args.class_name])

    print("KITTI evaluate: ", ap_result_str, ap_dict)


if __name__ == "__main__":
    kitti_eval()

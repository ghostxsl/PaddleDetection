#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import random
import numpy as np


def txt_load(file_path):
    with open(file_path) as f:
        out_list = f.readlines()
    return out_list


def load_img_name(dataset_dir, anno_path_list):
    out_list = []
    for anno_path in anno_path_list:
        out_list.extend([
            line.strip()
            for line in txt_load(os.path.join(dataset_dir, anno_path))
            if line.strip()
        ])
    return out_list


def init_active_dataset_miaod(all_data_id, initial_sample_ratio):
    num_init_sample = int(len(all_data_id) * initial_sample_ratio)
    random.shuffle(all_data_id)
    return all_data_id[:num_init_sample], all_data_id[
        -num_init_sample:], all_data_id[num_init_sample:]


def update_active_dataset_miaod(uncertainty, labeled_list, all_unlabeled_list,
                                delta_sample_ratio):
    assert len(uncertainty) == len(all_unlabeled_list)

    total_sample_num = len(labeled_list) + len(all_unlabeled_list)
    num_delta_sample = int(total_sample_num * delta_sample_ratio)
    assert num_delta_sample <= len(uncertainty)

    # sort uncertainty, get index
    uncertainty_ind = np.argsort(np.squeeze(uncertainty))
    # update labeled_list
    all_unlabeled_list = np.array(all_unlabeled_list)
    delta_list = all_unlabeled_list[uncertainty_ind[-num_delta_sample:]]
    delta_list = delta_list.tolist()
    labeled_list.extend(delta_list)

    # update unlabeled_list
    all_unlabeled_list = list(set(all_unlabeled_list) - set(delta_list))
    random.shuffle(all_unlabeled_list)
    unlabeled_list = all_unlabeled_list[:len(labeled_list)]
    if len(labeled_list) > len(unlabeled_list):
        random.shuffle(labeled_list)
        unlabeled_list.extend(labeled_list[:len(labeled_list) - len(
            unlabeled_list)])
    labeled_list.sort()
    unlabeled_list.sort()
    all_unlabeled_list.sort()
    return labeled_list, unlabeled_list, all_unlabeled_list

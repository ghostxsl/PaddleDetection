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

import os
import numpy as np
import xml.etree.ElementTree as ET

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from ppdet.core.workspace import register, serializable
from .dataset import DetDataset

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@register
@serializable
class MIAODVOCDataSet(DetDataset):
    """
    Load dataset with PascalVOC format.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): voc annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        label_list (str): if use_default_label is False, will load
            mapping between category and class index.
        class_name (list|None):
    """

    def __init__(self,
                 dataset_dir='dataset/voc',
                 image_dir=None,
                 anno_path='trainval.txt',
                 data_fields=('image', 'gt_bbox', 'gt_class'),
                 sample_num=-1,
                 label_list='label_list.txt',
                 class_name=None,
                 data_id_list=None):
        super(MIAODVOCDataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num)

        self.label_list = label_list
        if class_name is not None:
            self.cname2cid = {v: i for i, v in enumerate(class_name)}
        elif label_list is not None:
            txt_list = txt_load(self.get_label_list())
            self.cname2cid = {v.strip(): i for i, v in enumerate(txt_list)}
        else:
            self.cname2cid = pascalvoc_label()
        self.data_id_list = data_id_list
        self.all_data_dict = self.load_all_data()

    def load_all_data(self):
        count = 0
        all_data = {}
        anno_list = txt_load(os.path.join(self.dataset_dir, self.anno_path))
        for line in anno_list:
            if count == self.sample_num:
                break
            img_file, xml_file = line.strip().split()
            img_name = img_file.split('/')[-1].split("\\")[-1]
            img_file = os.path.join(self.dataset_dir, img_file)
            xml_file = os.path.join(self.dataset_dir, xml_file)
            if not os.path.exists(img_file):
                logger.warning('Illegal image file: {}, and it will be ignored'.
                               format(img_file))
                continue
            if not os.path.isfile(xml_file):
                logger.warning('Illegal xml file: {}, and it will be ignored'.
                               format(xml_file))
                continue
            tree = ET.parse(xml_file)
            if tree.find('id') is None:
                im_id = np.array([count])
            else:
                im_id = np.array([int(tree.find('id').text)])

            objs = tree.findall('object')
            im_w = float(tree.find('size').find('width').text)
            im_h = float(tree.find('size').find('height').text)
            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and {} will be ignored'.format(im_w, im_h,
                                                               xml_file))
                continue
            voc_rec = {
                'im_file': img_file,
                'im_id': im_id,
                'h': im_h,
                'w': im_w
            } if 'image' in self.data_fields else {}

            num_bbox, i = len(objs), 0
            gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
            gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
            gt_score = np.zeros((num_bbox, 1), dtype=np.float32)
            difficult = np.zeros((num_bbox, 1), dtype=np.int32)
            for obj in objs:
                cname = obj.find('name').text
                # user dataset may not contain difficult field
                _difficult = obj.find('difficult')
                _difficult = int(
                    _difficult.text) if _difficult is not None else 0

                x1 = float(obj.find('bndbox').find('xmin').text)
                y1 = float(obj.find('bndbox').find('ymin').text)
                x2 = float(obj.find('bndbox').find('xmax').text)
                y2 = float(obj.find('bndbox').find('ymax').text)
                x1 = max(0., x1)
                y1 = max(0., y1)
                x2 = min(im_w - 1, x2)
                y2 = min(im_h - 1, y2)
                if x2 > x1 and y2 > y1:
                    gt_bbox[i, :] = [x1, y1, x2, y2]
                    gt_class[i, 0] = self.cname2cid[cname]
                    gt_score[i, 0] = 1.
                    difficult[i, 0] = _difficult
                    i += 1
                else:
                    logger.warning(
                        'Found an invalid bbox in annotations: xml_file: {}'
                        ', x1: {}, y1: {}, x2: {}, y2: {}.'.format(xml_file, x1,
                                                                   y1, x2, y2))
            gt_bbox = gt_bbox[:i, :]
            gt_class = gt_class[:i, :]
            gt_score = gt_score[:i, :]
            difficult = difficult[:i, :]

            gt_rec = {
                'gt_class': gt_class,
                'gt_score': gt_score,
                'gt_bbox': gt_bbox,
                'difficult': difficult
            }
            for k, v in gt_rec.items():
                if k in self.data_fields:
                    voc_rec[k] = v

            all_data[count] = voc_rec
            count += 1
        assert count > 0, "Not found any voc record in `anno_path`. "
        logger.debug('{} samples in anno_file'.format(count))
        return all_data

    def parse_dataset(self):
        self.update_dataset()

    def update_dataset(self, data_id_list=None):
        if data_id_list is not None:
            self.data_id_list = data_id_list
        if self.data_id_list is None:
            self.roidbs = [v for k, v in self.all_data_dict.items()]
        else:
            self.roidbs = [self.all_data_dict[k] for k in self.data_id_list]

    def get_label_list(self):
        return os.path.join(self.dataset_dir, self.label_list)


def pascalvoc_label():
    labels_map = {
        'aeroplane': 0,
        'bicycle': 1,
        'bird': 2,
        'boat': 3,
        'bottle': 4,
        'bus': 5,
        'car': 6,
        'cat': 7,
        'chair': 8,
        'cow': 9,
        'diningtable': 10,
        'dog': 11,
        'horse': 12,
        'motorbike': 13,
        'person': 14,
        'pottedplant': 15,
        'sheep': 16,
        'sofa': 17,
        'train': 18,
        'tvmonitor': 19
    }
    return labels_map


def txt_load(file_path):
    with open(file_path) as f:
        out_list = f.readlines()
    return out_list

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import math
import numpy as np
import concurrent.futures as futures

from PIL import Image

from ppdet.core.workspace import register, serializable
from .dataset import DetDataset

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['MonoKitti3d']


@register
@serializable
class MonoKitti3d(DetDataset):
    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 data_fields=['image'],
                 sample_num=-1,
                 use_default_label=None,
                 num_worker=8,
                 category_name=('Pedestrian', 'Cyclist', 'Car'),
                 **kwargs):
        super().__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            use_default_label=use_default_label,
            **kwargs)

        [setattr(self, k, v) for k, v in locals().items()]

        self.name2catid = {
            cname: i
            for i, cname in enumerate(self.category_name)
        }
        self.clsid2catid = {
            i: cid
            for i, cid in enumerate(self.name2catid.values())
        }

    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        data_root = os.path.join(self.dataset_dir, self.image_dir)

        image_ids = [
            lin.strip() for lin in open(anno_path, 'r').readlines() if lin
        ]

        def _parse_func(idx):
            info = {}

            if 'image' in self.data_fields:
                info['im_file'] = get_image_path(data_root, idx)

            if 'label' in self.data_fields:
                label_path = get_label_path(data_root, idx)
                label = get_label_anno(label_path, self.name2catid)
                label = self.filter_kitti_anno(
                    label, used_classes=self.category_name, volume_thresh=0.001)
                info['label'] = label

            if 'calib' in self.data_fields:
                calib_path = get_calib_path(data_root, idx)
                info['calib'] = get_calib_info(calib_path)

            if 'velodyne' in self.data_fields:
                # info['velodyne'] = get_velodyne_path(data_root, idx)
                pass

            if 'label' in self.data_fields and 'calib' in self.data_fields:
                info['label'].update(
                    self.compute_centers(info['label'], info['calib']))
                info['label'].update(
                    self.compute_corners(info['label'], info['calib']))

            _info = {}
            for k in info:
                if isinstance(info[k], dict):
                    _info.update(info[k])
                else:
                    _info[k] = info[k]

            return _info

        with futures.ThreadPoolExecutor(self.num_worker) as executor:
            image_infos = executor.map(_parse_func, image_ids)

        self.roidbs = list(image_infos)
        if self.sample_num > -1:
            self.roidbs = self.roidbs[:self.sample_num]

    @staticmethod
    def filter_kitti_anno(label,
                          used_classes=None,
                          h_thresh=None,
                          score_thresh=None,
                          volume_thresh=0.001):
        if used_classes is not None:
            mask = np.array([n in used_classes for n in label['type']])
            label.update({
                k: v[mask]
                for k, v in label.items() if isinstance(v, np.ndarray)
            })

        if score_thresh is not None and 'score' in label:
            mask = label['score'] > volume_thresh
            label.update({
                k: v[mask]
                for k, v in label.items() if isinstance(v, np.ndarray)
            })

        if h_thresh is not None:
            mask = label['dimensions'][:, 1] > volume_thresh
            label.update({
                k: label[k][mask]
                for k in ['dimensions', 'location', 'rotation_y']
            })

        if volume_thresh > 0:
            mask = label['dimensions'].prod(axis=-1) > volume_thresh
            label.update({
                k: label[k][mask]
                for k in ['dimensions', 'location', 'rotation_y']
            })

        return label

    @staticmethod
    def compute_centers(label, calib):
        '''center3d and center2d
        '''
        loc = label['location']
        dim = label['dimensions']
        P0, P2 = calib['P0'], calib['P2']

        dst = np.array([0.5, 0.5, 0.5])
        src = np.array([0.5, 1.0, 0.5])
        center = loc + dim * (dst - src)

        offset = (P2[0, 3] - P0[0, 3]) / P2[0, 0]
        _center3d = center.copy()
        _center3d[0, 0] += offset  # camera_2

        _center = np.concatenate(
            (center, np.ones((center.shape[0], 1))), axis=-1)
        _center = (P2 @_center.T).T
        _depth = _center[:, 2]
        _center = _center[:, [0, 1]] / _center[:, [2]]

        return {
            'center_3d': _center3d.astype('float32'),
            'center_2d': _center.astype('float32'),
            'depth': _depth.astype('float32')
        }

    @staticmethod
    def compute_corners(label, calib, bbox2d=True):
        '''
        y z x(h w l)(kitti label file) <-> x y z(l h w)(camera)
        '''
        dims = label['dimensions']
        locs = label['location']
        rys = label['rotation_y']
        K = calib['P2']

        corners_3d = []
        corners_2d = []

        for dim, loc, ry in zip(dims, locs, rys):
            l, h, w = dim
            R = rotate_matrix(ry)

            _x = np.array([l, l, 0, 0, l, l, 0, 0]) - l / 2
            _y = np.array([h, h, h, h, 0, 0, 0, 0]) - h
            _z = np.array([w, 0, 0, w, w, 0, 0, w]) - w / 2
            _corners_3d = np.vstack([_x, _y, _z])
            _corners_3d = R @_corners_3d + np.array(loc).reshape(3, 1)  # 3 x 8

            _corners_2d = K[:3, :3] @_corners_3d
            _corners_2d = _corners_2d[[0, 1], :] / _corners_2d[[2], :]  # 2 x 8

            corners_3d.append(_corners_3d.T)
            corners_2d.append(_corners_2d.T)

        corners_2d = np.array(corners_2d)
        corners_3d = np.array(corners_3d)

        assert corners_2d.shape == (dims.shape[0], 8, 2), 'corners_2d'
        assert corners_3d.shape == (dims.shape[0], 8, 3), 'corners_3d'

        result = {
            'corners_2d': corners_2d.astype('float32'),
            'corners_3d': corners_3d.astype('float32')
        }

        if bbox2d is True:
            xmin = corners_2d[:, :, 0].min(axis=-1).reshape(-1, 1)
            xmax = corners_2d[:, :, 0].max(axis=-1).reshape(-1, 1)
            ymin = corners_2d[:, :, 1].min(axis=-1).reshape(-1, 1)
            ymax = corners_2d[:, :, 1].max(axis=-1).reshape(-1, 1)
            bbox_2d = np.concatenate([xmin, ymin, xmax, ymax], axis=-1)

            assert bbox_2d.shape == (dims.shape[0], 4), 'bbox_2d'

            result.update({'bbox_2d': bbox_2d.astype('float32')})

        return result


def rotate_matrix(ry):
    '''yaw _rotate_matrix
    '''
    R = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0],
                  [-math.sin(ry), 0, math.cos(ry)]])
    return R


def _get_image_index_str(idx):
    return "{:0>6}".format(idx)


def _get_info_path(idx, prefix, suffix, check_exists=True):
    path = os.path.join(prefix, _get_image_index_str(idx) + suffix)
    if check_exists:
        assert os.path.exists(path), f'{path} does not exists.'
    return path


def get_image_path(prefix, idx):
    return _get_info_path(idx, os.path.join(prefix, 'image_2'), '.png')


def get_label_path(prefix, idx):
    return _get_info_path(idx, os.path.join(prefix, 'label_2'), '.txt')


def get_calib_path(prefix, idx):
    return _get_info_path(idx, os.path.join(prefix, 'calib'), '.txt')


def get_velodyne_path(prefix, idx):
    return _get_info_path(idx, os.path.join(prefix, 'velodyne'), '.bin')


def get_label_anno(label_path, name2catid):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    annos = {}
    annos.update({
        'type': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })

    content = [line.strip().split(' ') for line in lines]
    num_gt = len(content)
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])

    annos['type'] = np.array(
        [name2catid[x[0]] for x in content], dtype=np.float32).reshape(-1, 1)
    annos['truncated'] = np.array(
        [float(x[1]) for x in content], dtype=np.float32).reshape(-1, 1)
    annos['occluded'] = np.array([int(x[2]) for x in content]).reshape(-1, 1)
    annos['alpha'] = np.array(
        [float(x[3]) for x in content], dtype=np.float32).reshape(-1, 1)
    annos['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content],
        dtype=np.float32).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annos['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content],
        dtype=np.float32).reshape(-1, 3)[:, [2, 0, 1]]
    annos['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content],
        dtype=np.float32).reshape(-1, 3)
    annos['rotation_y'] = np.array(
        [float(x[14]) for x in content], dtype=np.float32).reshape(-1, 1)

    if len(content) != 0 and len(content[0]) == 16:  # have score
        annos['score'] = np.array(
            [float(x[15]) for x in content], dtype=np.float32).reshape(-1, 1)
    else:
        annos['score'] = np.zeros(
            (annos['bbox'].shape[0], ), dtype=np.float32).reshape(-1, 1)

    # index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    # annos['index'] = np.array(index, dtype=np.int32)
    # annos['group_ids'] = np.arange(num_gt, dtype=np.int32)

    return annos


def _to_homo_coord(mat):
    _mat = np.eye(4)
    _j, _i = mat.shape
    _mat[:_j, :_i] = mat
    return _mat


def get_calib_info(calib_path, use_homo_coord=True):
    '''P0 P1 P2 P3 R0_rect Tr_velo_to_cam Tr_imu_to_velo
    '''
    blobs = {}
    with open(calib_path, 'r') as f:
        lines = [lin.strip() for lin in f.readlines()]
        lines = [lin for lin in lines if len(lin) > 0]

        for lin in lines:
            items = [item.strip() for item in lin.strip().split(':')]
            values = np.array(
                [float(x) for x in items[1].strip().split(' ')],
                dtype=np.float32)

            matrix = values.reshape(
                3, 3) if items[0] == 'R0_rect' else values.reshape(3, 4)

            if use_homo_coord:
                blobs[items[0]] = _to_homo_coord(matrix)
            else:
                blobs[items[0]] = matrix

    return blobs


class_to_label = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
    'Van': 3,
    'Person_sitting': 4,
    'Truck': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': -1,
}

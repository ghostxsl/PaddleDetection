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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
from ppdet.core.workspace import register

from ppdet.modeling.layers import ConvNormLayer
from ppdet.modeling.heads.fcos_head import ScaleReg
from ppdet.modeling.initializer import normal_, constant_


@register
class FCOSMono3DHead(nn.Layer):
    """
    FCOSMono3DHead
    Args:
        feat_branch (nn.Layer): Instance of 'FCOSFeat', in `fcos_head.py`
        num_classes (int): Number of classes
        fpn_strides (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        fcos_mono3d_loss (nn.Layer): Instance of 'FCOSLoss'
        norm_reg_targets (bool): Normalization the regression target if true
        centerness_on_reg (bool): The prediction of centerness on regression or clssification branch
    """
    __inject__ = ['feat_branch', 'fcos_mono3d_loss']
    __shared__ = ['num_classes']

    def __init__(
            self,
            feat_branch,
            num_classes=3,
            fpn_strides=(8, 16, 32, 64, 128),
            reg_group_out_channels=(2, 1, 3,
                                    1),  # offset, depth, size, rotation
            num_scale_regs=3,  # only for offset, depth and size regression
            disentangle_head_dims=(128, 64),
            norm_type='gn',
            prior_prob=0.01,
            fcos_mono3d_loss='FCOSMono3DLoss',
            object_sizes_boundary=[48, 96, 192, 384],
            center_sampling_radius=1.5,
            centerness_alpha=2.5,
            norm_reg_targets=True,
            centerness_on_reg=True):
        super(FCOSMono3DHead, self).__init__()
        self.feat_branch = feat_branch
        self.num_classes = num_classes
        self.bg_index = num_classes
        self.fpn_strides = fpn_strides
        self.reg_group_out_channels = reg_group_out_channels
        self.num_scale_regs = num_scale_regs
        self.disentangle_head_dims = disentangle_head_dims
        self.norm_type = norm_type
        self.bias_init_value = -math.log((1 - prior_prob) / prior_prob)
        self.fcos_mono3d_loss = fcos_mono3d_loss
        self.center_sampling_radius = center_sampling_radius
        self.centerness_alpha = centerness_alpha
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg

        object_sizes_boundary = [0] + object_sizes_boundary + [float('inf')]
        self.object_sizes_boundary = []
        for i in range(len(fpn_strides)):
            self.object_sizes_boundary.append(
                [object_sizes_boundary[i], object_sizes_boundary[i + 1]])
        self.object_sizes_boundary = paddle.to_tensor(
            self.object_sizes_boundary, dtype=paddle.float32)

        self.cls_head = self._init_cls_head(self.num_classes)
        self.reg_head_lists = self._init_reg_head()
        self.centerness_head = self._init_centerness_head()
        self.direction_cls_head = self._init_cls_head(2)

        self.scales_regs = nn.LayerList([
            nn.LayerList([ScaleReg() for _ in range(num_scale_regs)])
            for _ in self.fpn_strides
        ])

    @classmethod
    def from_config(cls, cfg, input_shape):

        return {'fpn_strides': [i.stride for i in input_shape]}

    def _init_cls_head(self, out_channels):
        layers = []
        in_channels = self.feat_branch.feat_out
        if self.disentangle_head_dims is not None:
            for channels in self.disentangle_head_dims:
                layers.append(
                    ConvNormLayer(
                        ch_in=in_channels,
                        ch_out=channels,
                        filter_size=3,
                        stride=1,
                        norm_type=self.norm_type))
                layers.append(nn.ReLU())
                in_channels = channels
            layers.append(
                nn.Conv2D(self.disentangle_head_dims[-1], out_channels, 1))
        else:
            layers.append(nn.Conv2D(in_channels, out_channels, 3, padding=1))
        normal_(layers[-1].weight, std=0.01)
        constant_(layers[-1].bias, self.bias_init_value)
        return nn.Sequential(*layers)

    def _init_reg_head(self):
        layers = nn.LayerList()
        for out_channels in self.reg_group_out_channels:
            inter_layers = []
            in_channels = self.feat_branch.feat_out
            if self.disentangle_head_dims is not None:
                for channels in self.disentangle_head_dims:
                    inter_layers.append(
                        ConvNormLayer(
                            ch_in=in_channels,
                            ch_out=channels,
                            filter_size=3,
                            stride=1,
                            norm_type=self.norm_type))
                    inter_layers.append(nn.ReLU())
                    in_channels = channels
                inter_layers.append(
                    nn.Conv2D(self.disentangle_head_dims[-1], out_channels, 1))
            else:
                layers.append(
                    nn.Conv2D(
                        in_channels, out_channels, 3, padding=1))
            layers.append(nn.Sequential(*inter_layers))
        for layer in layers:
            normal_(layer[-1].weight, std=0.01)
            constant_(layer[-1].bias)
        return layers

    def _init_centerness_head(self):
        layers = []
        in_channels = self.feat_branch.feat_out
        if self.disentangle_head_dims is not None:
            for channels in self.disentangle_head_dims:
                layers.append(
                    ConvNormLayer(
                        ch_in=in_channels,
                        ch_out=channels,
                        filter_size=3,
                        stride=1,
                        norm_type=self.norm_type))
                layers.append(nn.ReLU())
                in_channels = channels
            layers.append(nn.Conv2D(self.disentangle_head_dims[-1], 1, 1))
        else:
            layers.append(nn.Conv2D(in_channels, 1, 3, padding=1))
        normal_(layers[-1].weight, std=0.01)
        constant_(layers[-1].bias)
        return nn.Sequential(*layers)

    def _compute_anchor_points(self, fpn_feat_shapes):
        anchor_points_list = []
        for fpn_stride, (h, w) in zip(self.fpn_strides, fpn_feat_shapes):
            shift_x = paddle.arange(0, w * fpn_stride, fpn_stride)
            shift_y = paddle.arange(0, h * fpn_stride, fpn_stride)
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            points = paddle.stack(
                [shift_x, shift_y], axis=-1) + float(fpn_stride) / 2
            anchor_points_list.append(points)
        return anchor_points_list

    @staticmethod
    def _compute_bbox_lrtb(x, y, bbox):
        l = x - bbox[:, :, 0]
        r = bbox[:, :, 2] - x
        t = y - bbox[:, :, 1]
        b = bbox[:, :, 3] - y
        return paddle.stack([l, r, t, b], axis=-1)

    def _check_inside_bbox(self, reg_lrtb, gt_2d_bbox, anchor_points_x,
                           anchor_points_y, fpn_strides_tensor):
        if self.center_sampling_radius > 0:
            fpn_strides_tensor *= self.center_sampling_radius
            gt_bbox_cx = (gt_2d_bbox[:, 0] + gt_2d_bbox[:, 2]) / 2
            gt_bbox_cy = (gt_2d_bbox[:, 1] + gt_2d_bbox[:, 3]) / 2
            gt_bbox_cxcy = paddle.stack(
                [gt_bbox_cx, gt_bbox_cy], axis=-1).flatten()
            clip_bbox_xminymin = (gt_bbox_cxcy - fpn_strides_tensor).reshape(
                [fpn_strides_tensor.shape[0], -1, 2])
            clip_bbox_xmaxymax = (gt_bbox_cxcy + fpn_strides_tensor).reshape(
                [fpn_strides_tensor.shape[0], -1, 2])
            clip_bbox = paddle.concat(
                [clip_bbox_xminymin, clip_bbox_xmaxymax], axis=-1)
            gt_bbox = gt_2d_bbox.unsqueeze(0).tile(
                [fpn_strides_tensor.shape[0], 1, 1])
            clip_bbox[:, :, :2] = paddle.where(
                gt_bbox[:, :, :2] > clip_bbox[:, :, :2], gt_bbox[:, :, :2],
                clip_bbox[:, :, :2])
            clip_bbox[:, :, 2:] = paddle.where(
                gt_bbox[:, :, 2:] < clip_bbox[:, :, 2:], gt_bbox[:, :, 2:],
                clip_bbox[:, :, 2:])
            reg_lrtb = self._compute_bbox_lrtb(anchor_points_x, anchor_points_y,
                                               clip_bbox)
        return (reg_lrtb.min(axis=-1) > 0).astype(paddle.float32)

    def _check_bbox_in_scale(self, reg_max, num_points_list):
        bbox_scale = []
        for object_size, num_points in zip(self.object_sizes_boundary,
                                           num_points_list):
            object_size = object_size.unsqueeze(0).tile([num_points, 1])
            bbox_scale.append(object_size)
        bbox_scale = paddle.concat(bbox_scale)
        lower_bound = bbox_scale[:, 0:1].tile([1, reg_max.shape[1]])
        upper_bound = bbox_scale[:, 1:].tile([1, reg_max.shape[1]])
        is_match_scale = (reg_max > lower_bound).astype(paddle.float32) * \
                         (reg_max < upper_bound).astype(paddle.float32)
        return is_match_scale

    def _gt_to_target_assignment(self, anchor_points, gt_labels, gt_2d_bboxes,
                                 gt_2d_centers, gt_depths, gt_sizes,
                                 gt_rotations_y):
        targets_class = []
        targets_centerness = []
        targets_regression = []
        targets_direction = []

        anchor_points_flatten = paddle.concat(
            [a.reshape([-1, 2]) for a in anchor_points])
        anchor_points_x = anchor_points_flatten[:, 0:1]
        anchor_points_y = anchor_points_flatten[:, 1:]
        num_points_list = [a.shape[0] * a.shape[1] for a in anchor_points]
        fpn_strides_tensor = []
        for num_points, fpn_stride in zip(num_points_list, self.fpn_strides):
            fpn_stride = paddle.to_tensor([fpn_stride]).unsqueeze(0).tile(
                [num_points, 1])
            fpn_strides_tensor.append(fpn_stride)
        fpn_strides_tensor = paddle.concat(fpn_strides_tensor)

        for gt_label, gt_2d_bbox, gt_2d_center, gt_depth, gt_size, gt_rotation in zip(
                gt_labels, gt_2d_bboxes, gt_2d_centers, gt_depths, gt_sizes,
                gt_rotations_y):

            gt_2d_bbox = gt_2d_bbox.astype(paddle.float32)
            gt_2d_center = gt_2d_center.astype(paddle.float32)
            gt_depth = gt_depth.astype(paddle.float32)
            gt_size = gt_size.astype(paddle.float32)
            gt_rotation = gt_rotation.astype(paddle.float32)

            reg_star = self._compute_bbox_lrtb(anchor_points_x, anchor_points_y,
                                               gt_2d_bbox.unsqueeze(0))
            # spatial match: center sampling
            is_inside_bbox = self._check_inside_bbox(
                reg_star, gt_2d_bbox, anchor_points_x, anchor_points_y,
                fpn_strides_tensor)
            # scale match: object size range
            reg_star_max = reg_star.max(axis=-1)
            is_match_scale = self._check_bbox_in_scale(reg_star_max,
                                                       num_points_list)
            # distance-based criterion
            bbox_cx = (gt_2d_bbox[:, 0] + gt_2d_bbox[:, 2]) / 2
            bbox_cy = (gt_2d_bbox[:, 1] + gt_2d_bbox[:, 3]) / 2
            points2gt_distance_v2 = paddle.square(anchor_points_x - bbox_cx) + \
                                 paddle.square(anchor_points_y - bbox_cy)

            mask_positive = (is_inside_bbox * is_match_scale).max(axis=-1)
            min_dist_ind = points2gt_distance_v2.argmin(axis=-1)
            # make classification target
            target_class = paddle.gather(gt_label, min_dist_ind, axis=0)
            target_class[mask_positive == 0] = self.bg_index
            targets_class.append(target_class.astype(paddle.int64))
            # make regression target
            gt_2d_center_ = paddle.gather(gt_2d_center, min_dist_ind, axis=0)
            target_offset_ = gt_2d_center_ - anchor_points_flatten
            target_offset = target_offset_ / fpn_strides_tensor \
                if self.norm_reg_targets else target_offset_
            target_depth = paddle.gather(
                gt_depth, min_dist_ind, axis=0).unsqueeze(1)
            target_size = paddle.gather(gt_size, min_dist_ind, axis=0)
            gt_direction = (gt_rotation >= 0).astype(paddle.float32)
            gt_rotation = (1 - gt_direction) * (gt_rotation + math.pi
                                                ) + gt_direction * gt_rotation
            targets_rotation = paddle.gather(
                gt_rotation, min_dist_ind, axis=0).unsqueeze(1)
            targets_regression.append(
                paddle.concat(
                    [
                        target_offset, target_depth, target_size,
                        targets_rotation
                    ],
                    axis=-1))
            # make centerness target
            target_centerness = paddle.exp(
                -self.centerness_alpha *
                target_offset_.square().sum(axis=-1, keepdim=True).sqrt() /
                (1.414 * fpn_strides_tensor))
            target_centerness[target_class == self.bg_index] = 0
            targets_centerness.append(target_centerness)
            # make direction target
            target_direction = paddle.gather(
                gt_direction.astype(paddle.int64), min_dist_ind, axis=0)
            targets_direction.append(target_direction)
        targets_class = paddle.stack(targets_class)
        targets_centerness = paddle.stack(targets_centerness)
        targets_regression = paddle.stack(targets_regression)
        targets_direction = paddle.stack(targets_direction)

        return targets_class, targets_regression, targets_centerness, targets_direction

    def forward(self, fpn_feats, is_training):
        assert len(fpn_feats) == len(self.fpn_strides), \
            "The size of fpn_feats is not equal to size of fpn_strides"
        cls_logits = []
        bboxes_reg = []
        centerness_list = []
        direction_cls_logits = []
        for scale_reg, fpn_stride, fpn_feat in zip(self.scales_regs,
                                                   self.fpn_strides, fpn_feats):
            fcos_cls_feat, fcos_reg_feat = self.feat_branch(fpn_feat)
            cls_logit = self.cls_head(fcos_cls_feat)
            cls_logit = cls_logit.flatten(start_axis=2).transpose([0, 2, 1])

            if self.centerness_on_reg:
                centerness = self.centerness_head(fcos_reg_feat)
            else:
                centerness = self.centerness_head(fcos_cls_feat)

            bbox_reg = []
            for i, reg_head in enumerate(self.reg_head_lists):
                if i < self.num_scale_regs:
                    bbox_reg.append(scale_reg[i](reg_head(fcos_reg_feat)))
                    if i > 0:
                        bbox_reg[-1] = paddle.exp(bbox_reg[-1])
                    else:
                        if self.norm_reg_targets and not is_training:
                            bbox_reg[-1] = bbox_reg[-1] * fpn_stride
                else:
                    bbox_reg.append(reg_head(fcos_reg_feat))

            direction_cls_logit = self.direction_cls_head(fcos_reg_feat)
            direction_cls_logits.append(
                direction_cls_logit.flatten(start_axis=2).transpose([0, 2, 1]))
            cls_logits.append(cls_logit)
            bboxes_reg.append(
                paddle.concat(
                    bbox_reg, axis=1).flatten(start_axis=2).transpose(
                        [0, 2, 1]))
            centerness_list.append(
                centerness.flatten(start_axis=2).transpose([0, 2, 1]))

        cls_logits = paddle.concat(cls_logits, axis=1)
        bboxes_reg = paddle.concat(bboxes_reg, axis=1)
        centerness_list = paddle.concat(centerness_list, axis=1)
        direction_cls_logits = paddle.concat(direction_cls_logits, axis=1)

        anchor_points_list = self._compute_anchor_points(
            [a.shape[2:] for a in fpn_feats])
        return cls_logits, bboxes_reg, centerness_list, direction_cls_logits, anchor_points_list

    def get_loss(self, fcos_head_outs, gt_labels, gt_2d_bboxes, gt_2d_centers,
                 gt_depths, gt_sizes, gt_rotations_y):
        cls_logits, bboxes_reg, centerness, direction_logits, anchor_points = fcos_head_outs

        target_class, target_regression, target_centerness, target_direction = \
            self._gt_to_target_assignment(anchor_points,
                gt_labels, gt_2d_bboxes, gt_2d_centers, gt_depths, gt_sizes,
                gt_rotations_y)
        return self.fcos_mono3d_loss(
            cls_logits, bboxes_reg, centerness, direction_logits, target_class,
            target_regression, target_centerness, target_direction)

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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ppdet.core.workspace import register
from ..bbox_utils import bbox_center, iou_similarity
from .utils import (pad_gt, check_points_inside_bboxes, compute_max_iou_anchor,
                    compute_max_iou_gt)

__all__ = ['DTSSv2Assigner']


@register
class DTSSv2Assigner(nn.Layer):
    """Dynamic Training Sample Selection v2
    """
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 topk=10,
                 select_candidate_type='center_distance',
                 center_radius=2.5,
                 normalize=None,
                 alpha=1.0,
                 beta=1.0,
                 force_gt_matching=False,
                 eps=1e-9):
        super(DTSSv2Assigner, self).__init__()
        assert select_candidate_type in ['center_distance', 'center_radius']
        assert normalize in [None, 'minmax']

        self.num_classes = num_classes
        self.topk = topk
        self.select_candidate_type = select_candidate_type
        self.center_radius = center_radius
        self.normalize = normalize
        self.alpha = alpha
        self.beta = beta
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_minimum_pyramid(self, candidate_metrics, num_anchors_list,
                                     pad_gt_mask):
        pad_gt_mask = pad_gt_mask.tile([1, 1, self.topk]).astype(paddle.bool)
        candidate_metrics_list = paddle.split(
            candidate_metrics, num_anchors_list, axis=-1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(candidate_metrics_list,
                                            num_anchors_index):
            num_anchors = distances.shape[-1]
            topk_metrics, topk_idxs = paddle.topk(
                distances, self.topk, axis=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            topk_idxs = paddle.where(pad_gt_mask, topk_idxs,
                                     paddle.zeros_like(topk_idxs))
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
            is_in_topk = paddle.where(is_in_topk > 1,
                                      paddle.zeros_like(is_in_topk), is_in_topk)
            is_in_topk_list.append(is_in_topk.astype(candidate_metrics.dtype))
        is_in_topk_list = paddle.concat(is_in_topk_list, axis=-1)
        topk_idxs_list = paddle.concat(topk_idxs_list, axis=-1)
        return is_in_topk_list, topk_idxs_list

    def _assign(self,
                pred_scores,
                pred_bboxes,
                gt_labels,
                gt_bboxes,
                anchor_points,
                num_anchors_list,
                stride_tensor,
                pad_gt_mask,
                bg_index,
                pad_gt_scores=None):
        assert hasattr(self, 'batch_size')
        assert hasattr(self, 'num_anchors')
        assert hasattr(self, 'num_max_boxes')

        # 1. compute iou between gt and pred bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes, pred_bboxes)
        # gather pred bboxes class score
        pred_scores = pred_scores.transpose([0, 2, 1])
        batch_ind = paddle.arange(
            end=self.batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        gt_labels_ind = paddle.stack(
            [batch_ind.tile([1, self.num_max_boxes]), gt_labels.squeeze(-1)],
            axis=-1)
        conf_scores = paddle.gather_nd(pred_scores, gt_labels_ind)
        # 2. compute metrics, [B, n, L]
        if self.normalize == 'minmax':
            ious_normalize = ious / (ious.max(axis=-1, keepdim=True) + self.eps)
            conf_scores_normalize = conf_scores / (
                conf_scores.max(axis=-1, keepdim=True) + self.eps)
        else:
            ious_normalize, conf_scores_normalize = ious, conf_scores
        cost_metrics = conf_scores_normalize.pow(
            self.alpha) * ious_normalize.pow(self.beta)
        # cost_metrics = -paddle.log(cost_metrics)

        # 3. compute candidate set, [B, n, L]
        if self.select_candidate_type == 'center_radius':
            # check the positive sample's center in gt, [B, n, L]
            stride_tensor *= self.center_radius
            is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes,
                                                   stride_tensor)
            candidate_metrics = -(cost_metrics * is_in_gts)
        elif self.select_candidate_type == 'center_distance':
            gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
            candidate_metrics = (gt_centers - anchor_points.unsqueeze(0)) \
                .norm(2, axis=-1).reshape([self.batch_size, -1, self.num_anchors])
        else:
            raise Exception(f'error type: {self.select_candidate_type}!')
        is_in_topk_candidates, topk_idxs = self._gather_topk_minimum_pyramid(
            candidate_metrics, num_anchors_list, pad_gt_mask)

        # 4. get corresponding iou/metric for the these candidates,
        #    and compute these candidate's mean and std,
        #    set mean + std as the iou/metric threshold
        metrics_candidates = cost_metrics * is_in_topk_candidates
        metrics_threshold = paddle.index_sample(
            metrics_candidates.flatten(stop_axis=-2),
            topk_idxs.flatten(stop_axis=-2))
        metrics_threshold = metrics_threshold.reshape(
            [self.batch_size, self.num_max_boxes, -1])
        metrics_threshold = metrics_threshold.mean(axis=-1, keepdim=True) + \
                        metrics_threshold.std(axis=-1, keepdim=True)
        is_in_topk = paddle.where(metrics_candidates > metrics_threshold.tile(
            [1, 1, self.num_anchors]), is_in_topk_candidates,
                                  paddle.zeros_like(is_in_topk_candidates))

        # 5. select positive sample, [B, n, L]
        mask_positive = is_in_topk * pad_gt_mask

        # 6. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile(
                [1, self.num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(cost_metrics)
            mask_positive = paddle.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        # 7. make sure every gt_bbox matches the anchor
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).tile(
                [1, self.num_max_boxes, 1])
            mask_positive = paddle.where(mask_max_iou, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        assigned_gt_index = mask_positive.argmax(axis=-2)
        assert mask_positive_sum.max() == 1, \
            ("one anchor just assign one gt, but received not equals 1. "
             "Received: %f" % mask_positive_sum.max().item())

        # assigned target
        batch_ind = paddle.arange(
            end=self.batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        assigned_gt_index = assigned_gt_index + batch_ind * self.num_max_boxes
        assigned_labels = paddle.gather(
            gt_labels.flatten(), assigned_gt_index.flatten(), axis=0)
        assigned_labels = assigned_labels.reshape(
            [self.batch_size, self.num_anchors])
        assigned_labels = paddle.where(
            mask_positive_sum > 0, assigned_labels,
            paddle.full_like(assigned_labels, bg_index))

        assigned_bboxes = paddle.gather(
            gt_bboxes.reshape([-1, 4]), assigned_gt_index.flatten(), axis=0)
        assigned_bboxes = assigned_bboxes.reshape(
            [self.batch_size, self.num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, self.num_classes)
        # rescale metrics to target score
        cost_metrics *= mask_positive
        max_metrics_per_instance = cost_metrics.max(axis=-1, keepdim=True)
        max_ious_per_instance = (ious * mask_positive).max(axis=-1,
                                                           keepdim=True)
        cost_metrics = cost_metrics / (
            max_metrics_per_instance + self.eps) * max_ious_per_instance
        cost_metrics = cost_metrics.max(-2).unsqueeze(-1)
        assigned_scores = assigned_scores * cost_metrics

        ignore_mask = is_in_topk_candidates.max(-2) - mask_positive_sum

        return assigned_labels, assigned_bboxes, assigned_scores, ignore_mask

    @paddle.no_grad()
    def forward(self,
                pred_scores,
                pred_bboxes,
                anchor_points,
                num_anchors_list,
                stride_tensor,
                gt_labels,
                gt_bboxes,
                bg_index,
                gt_scores=None):
        r"""The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 4)
            anchor_points (Tensor, float32): pre-defined anchors, shape(L, 2), "cxcy" format
            num_anchors_list (List): num of anchors in each level
            stride_tensor (Tensor, float32): stride of features, shape(L, 1)
            gt_labels (Tensor|List[Tensor], int64): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor|List[Tensor], float32): Ground truth bboxes, shape(B, n, 4)
            bg_index (int): background index
            gt_scores (Tensor|List[Tensor]|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C)
            ignore_mask (Tensor): (B, L)
        """
        gt_labels, gt_bboxes, pad_gt_scores, pad_gt_mask = pad_gt(
            gt_labels, gt_bboxes, gt_scores)
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3

        self.num_anchors, _ = anchor_points.shape
        self.batch_size, self.num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if self.num_max_boxes == 0:
            assigned_labels = paddle.full(
                [self.batch_size, self.num_anchors],
                bg_index,
                dtype=gt_labels.dtype)
            assigned_bboxes = paddle.zeros(
                [self.batch_size, self.num_anchors, 4])
            assigned_scores = paddle.zeros(
                [self.batch_size, self.num_anchors, self.num_classes])
            ignore_mask = paddle.zeros([self.batch_size, self.num_anchors])
            out_list = [
                assigned_labels, assigned_bboxes, assigned_scores, ignore_mask
            ]
            return out_list, out_list

        # static assign
        assigned_labels, assigned_bboxes, assigned_scores, ignore_mask = \
            self._assign(pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchor_points,
                        num_anchors_list, stride_tensor, pad_gt_mask, bg_index, pad_gt_scores)

        return assigned_labels, assigned_bboxes, assigned_scores

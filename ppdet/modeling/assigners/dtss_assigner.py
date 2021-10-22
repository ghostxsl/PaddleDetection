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

__all__ = ['DTSSAssigner']


@register
class DTSSAssigner(nn.Layer):
    """Dynamic Training Sample Selection
    """
    __shared__ = ['num_classes']

    def __init__(self,
                 topk=15,
                 num_classes=80,
                 alpha=1.0,
                 beta=3.0,
                 force_gt_matching=False,
                 eps=1e-9):
        super(DTSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
                             pad_gt_mask):
        pad_gt_mask = pad_gt_mask.tile([1, 1, self.topk]).astype(paddle.bool)
        gt2anchor_distances_list = paddle.split(
            gt2anchor_distances, num_anchors_list, axis=-1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list,
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
            is_in_topk_list.append(is_in_topk.astype(gt2anchor_distances.dtype))
        is_in_topk_list = paddle.concat(is_in_topk_list, axis=-1)
        topk_idxs_list = paddle.concat(topk_idxs_list, axis=-1)
        return is_in_topk_list, topk_idxs_list

    def _static_assign(self,
                       gt_labels,
                       gt_bboxes,
                       anchor_bboxes,
                       num_anchors_list,
                       pad_gt_mask,
                       bg_index,
                       pad_gt_scores=None):
        assert hasattr(self, 'batch_size')
        assert hasattr(self, 'num_anchors')
        assert hasattr(self, 'num_max_boxes')

        # 1. compute iou between gt and anchor bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes, anchor_bboxes.unsqueeze(0))

        # 2. compute center distance between all anchors and gt, [B, n, L]
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)) \
            .norm(2, axis=-1).reshape([self.batch_size, -1, self.num_anchors])

        # 3. on each pyramid level, selecting topk closest candidates
        # based on the center distance, [B, n, L]
        is_in_topk_candidates, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask)

        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk_candidates
        iou_threshold = paddle.index_sample(
            iou_candidates.flatten(stop_axis=-2),
            topk_idxs.flatten(stop_axis=-2))
        iou_threshold = iou_threshold.reshape(
            [self.batch_size, self.num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(axis=-1, keepdim=True) + \
                        iou_threshold.std(axis=-1, keepdim=True)
        is_in_topk = paddle.where(
            iou_candidates > iou_threshold.tile([1, 1, self.num_anchors]),
            is_in_topk_candidates, paddle.zeros_like(is_in_topk_candidates))

        # 6. check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # 7. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile(
                [1, self.num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = paddle.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        # 8. make sure every gt_bbox matches the anchor
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
        if pad_gt_scores is not None:
            gather_scores = paddle.gather(
                pad_gt_scores.flatten(), assigned_gt_index.flatten(), axis=0)
            gather_scores = gather_scores.reshape(
                [self.batch_size, self.num_anchors])
            gather_scores = paddle.where(mask_positive_sum > 0, gather_scores,
                                         paddle.zeros_like(gather_scores))
            assigned_scores *= gather_scores.unsqueeze(-1)

        ignore_mask = is_in_topk_candidates.max(-2) - mask_positive_sum

        return [assigned_labels, assigned_bboxes, assigned_scores, ignore_mask], \
               is_in_topk_candidates, topk_idxs, is_in_gts

    def _dynamic_assign(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes,
                        is_in_gts, is_in_topk_candidates, topk_idxs,
                        pad_gt_mask, bg_index):
        assert hasattr(self, 'batch_size')
        assert hasattr(self, 'num_anchors')
        assert hasattr(self, 'num_max_boxes')

        # compute iou between gt and pred bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes, pred_bboxes)
        # gather pred bboxes class score
        pred_scores = pred_scores.transpose([0, 2, 1])
        batch_ind = paddle.arange(
            end=self.batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        gt_labels_ind = paddle.stack(
            [batch_ind.tile([1, self.num_max_boxes]), gt_labels.squeeze(-1)],
            axis=-1)
        conf_scores = paddle.gather_nd(pred_scores, gt_labels_ind)
        # compute metrics, [B, n, L]
        loss_metrics = conf_scores.pow(self.alpha) * ious.pow(self.beta)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold, like ATSS,
        # but use loss metrics
        metrics_candidates = loss_metrics * is_in_topk_candidates
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

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile(
                [1, self.num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = paddle.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        # make sure every gt_bbox matches the anchor
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
        # rescale metrics
        loss_metrics *= mask_positive
        max_metrics_per_instance = loss_metrics.max(axis=-1, keepdim=True)
        max_ious_per_instance = (ious * mask_positive).max(axis=-1,
                                                           keepdim=True)
        alignment_metrics = loss_metrics / (
            max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics = alignment_metrics.max(-2).unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        ignore_mask = is_in_topk_candidates.max(-2) - mask_positive_sum

        return [assigned_labels, assigned_bboxes, assigned_scores, ignore_mask]

    def _bipartite_assign(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes,
                          is_in_topk_in_gt_candidates, pad_gt_mask, bg_index):
        assert hasattr(self, 'batch_size')
        assert hasattr(self, 'num_anchors')
        assert hasattr(self, 'num_max_boxes')

        # compute iou between gt and pred bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes, pred_bboxes)
        # gather pred bboxes class score
        pred_scores = pred_scores.transpose([0, 2, 1])
        batch_ind = paddle.arange(
            end=self.batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        gt_labels_ind = paddle.stack(
            [batch_ind.tile([1, self.num_max_boxes]), gt_labels.squeeze(-1)],
            axis=-1)
        conf_scores = paddle.gather_nd(pred_scores, gt_labels_ind)
        # compute metrics, [B, n, L]
        loss_metrics = conf_scores.pow(self.alpha) * ious.pow(self.beta)

        # bipartite match
        metrics_candidates = loss_metrics * is_in_topk_in_gt_candidates * pad_gt_mask
        match_indices = [
            linear_sum_assignment(m.numpy(), True) for m in metrics_candidates
        ]
        # assigned target
        mask_positive = paddle.zeros_like(metrics_candidates)
        assigned_labels = paddle.full(
            [self.batch_size, self.num_anchors],
            bg_index,
            dtype=gt_labels.dtype)
        assigned_bboxes = paddle.zeros([self.batch_size, self.num_anchors, 4])
        assigned_scores = paddle.zeros(
            [self.batch_size, self.num_anchors, self.num_classes])
        for b_ind, (i_arr, j_arr) in enumerate(match_indices):
            for i, j in zip(i_arr, j_arr):
                if pad_gt_mask[b_ind, i] == 1:
                    mask_positive[b_ind, i, j] = 1
                    assigned_labels[b_ind, j] = gt_labels[b_ind, i, 0]
                    assigned_bboxes[b_ind, j] = gt_bboxes[b_ind, i]
                    assigned_scores[b_ind, j, int(gt_labels[
                        b_ind, i, 0])] = ious[b_ind, i, j]
        ignore_mask = is_in_topk_in_gt_candidates.max(-2) - mask_positive.sum(
            axis=-2)

        return [assigned_labels, assigned_bboxes, assigned_scores, ignore_mask]

    @paddle.no_grad()
    def forward(self,
                pred_scores,
                pred_bboxes,
                anchor_bboxes,
                num_anchors_list,
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
            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level
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

        self.num_anchors, _ = anchor_bboxes.shape
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
        static_assigned_list, is_in_topk_candidates, topk_idxs, is_in_gts = \
            self._static_assign(gt_labels, gt_bboxes, anchor_bboxes,
                                num_anchors_list, pad_gt_mask, bg_index, pad_gt_scores)
        # # dynamic assign
        # dynamic_assigned_list = self._dynamic_assign(
        #     pred_scores, pred_bboxes, gt_labels, gt_bboxes, is_in_gts,
        #     is_in_topk_candidates, topk_idxs, pad_gt_mask, bg_index)

        # bipartite assign
        dynamic_assigned_list = self._bipartite_assign(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes,
            is_in_topk_candidates * is_in_gts, pad_gt_mask, bg_index)

        return static_assigned_list, dynamic_assigned_list

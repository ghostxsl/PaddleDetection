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
import paddle.nn.functional as F
from paddle.fluid import core
from paddle.fluid.dygraph import parallel_helper

from ppdet.core.workspace import register
from ..initializer import constant_, bias_init_with_prob, normal_
from ppdet.modeling.bbox_utils import bbox_center, batch_distance2bbox
from ..losses import GIoULoss
from ppdet.modeling.ops import get_static_shape, mish
from ppdet.modeling.assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.layers import ConvNormLayer

__all__ = ['TaskAttnHead']


class MultiHeadAttn(nn.Layer):
    def __init__(self, in_channels, num_heads=8, act='mish'):
        super(MultiHeadAttn, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.act = act
        self.se_attn = nn.Conv2D(in_channels, num_heads, 1)
        self.conv = ConvNormLayer(
            in_channels, in_channels, 1, 1, norm_type='gn')

        self._init_weights()

    def _init_weights(self):
        normal_(self.se_attn.weight, std=0.001)

    def forward(self, feat, avg_feat=None):
        b, _, h, w = get_static_shape(feat)
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = feat.reshape([b, self.num_heads, -1, h, w])

        se_feat = F.sigmoid(self.se_attn(avg_feat)).unsqueeze(-1)
        out = (feat * se_feat).reshape([b, self.in_channels, h, w])
        out = self.conv(out)
        if self.act == 'mish':
            out = mish(out)
        elif self.act == 'relu':
            out = F.relu(out)
        return out


class AttnModule(nn.Layer):
    def __init__(self, in_channels, num_heads=8, num_points=4):
        super(AttnModule, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.total_points = self.num_heads * self.num_points

        self.conv_offset = nn.Conv2D(
            self.in_channels, self.total_points * 2, 3, padding=1)
        self.conv_attn = nn.Conv2D(
            self.in_channels, self.total_points, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        normal_(self.conv_attn.weight, std=0.01)

        constant_(self.conv_offset.weight)
        thetas = paddle.arange(
            self.num_heads,
            dtype=paddle.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init.reshape([self.num_heads, 1, 2]).tile(
            [1, self.num_points, 1])
        scaling = paddle.arange(
            1, self.num_points + 1,
            dtype=paddle.float32).reshape([1, 1, -1, 1])
        grid_init *= scaling
        self.conv_offset.bias.set_value(grid_init.flatten())

    def _attn_grid_sample(self, feat, offset, attn_weight, ref_points):
        b, _, h, w = get_static_shape(feat)
        feat = feat.reshape([b * self.num_heads, -1, h, w])
        offset = offset.reshape([-1, self.num_points, 2, h, w]).transpose(
            [0, 1, 3, 4, 2]).flatten(1, 2)
        grid_shape = paddle.concat([w, h]).astype('float32')
        ref_points = ref_points.reshape([1, h, w, 2]).tile(
            [1, self.num_points, 1, 1])
        grid = (offset + ref_points) / grid_shape
        grid = 2 * grid.clip(0., 1.) - 1
        feat = F.grid_sample(feat, grid)
        feat = paddle.reshape(feat,
                              [b, self.num_heads, -1, self.num_points, h, w])
        attn_weight = attn_weight.reshape(
            [b, self.num_heads, 1, self.num_points, h, w])
        out = (feat * attn_weight).sum(3).reshape([b, -1, h, w])
        return out

    def forward(self, query_feat, value_feat, ref_points):
        offset = self.conv_offset(query_feat)
        attn_weight = F.sigmoid(self.conv_attn(query_feat))
        out = self._attn_grid_sample(value_feat, offset, attn_weight,
                                     ref_points)
        return out


@register
class TaskAttnHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['nms', 'static_assigner', 'assigner']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 num_heads=8,
                 num_points=4,
                 grid_cell_scale=5,
                 grid_cell_offset=0.5,
                 static_assigner_epoch=4,
                 use_attn=False,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.0,
                 }):
        super(TaskAttnHead, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.num_heads = num_heads
        self.num_points = num_points
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.giou_loss = GIoULoss()
        self.loss_weight = loss_weight

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        self.use_attn = use_attn
        self.use_varifocal_loss = use_varifocal_loss

        self.feat_cls = nn.LayerList()
        self.feat_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.feat_cls.append(MultiHeadAttn(in_c, self.num_heads))
            self.feat_reg.append(MultiHeadAttn(in_c, self.num_heads))

        if self.use_attn:
            self.attn_cls = nn.LayerList()
            self.attn_reg = nn.LayerList()
            for in_c in self.in_channels:
                self.attn_cls.append(
                    AttnModule(in_c, self.num_heads, self.num_points))
                self.attn_reg.append(
                    AttnModule(in_c, self.num_heads, self.num_points))

        self.conv_cls = nn.LayerList()
        self.conv_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.conv_cls.append(nn.Conv2D(in_c, self.num_classes, 1))
            self.conv_reg.append(nn.Conv2D(in_c, 4, 1))
        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_reg = math.log(self.grid_cell_scale / 2)
        for cls_, reg_ in zip(self.conv_cls, self.conv_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, bias_reg)

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"
        anchors, num_anchors_list, stride_tensor_list = \
            generate_anchors_for_grid_cell(feats,
                                           self.fpn_strides,
                                           self.grid_cell_scale,
                                           self.grid_cell_offset)

        pred_logit, pred_dist = [], []
        if self.use_attn:
            for feat, feat_cls, feat_reg, attn_cls, attn_reg,\
                conv_cls, conv_reg, anchor, stride in zip(
                    feats, self.feat_cls, self.feat_reg,
                    self.attn_cls, self.attn_reg, self.conv_cls,
                    self.conv_reg, anchors, self.fpn_strides):
                value_feat_cls = feat_cls(feat)
                value_feat_reg = feat_reg(feat)
                anchor_centers = bbox_center(anchor) / stride
                cls_logit = conv_cls(
                    attn_cls(feat, value_feat_cls, anchor_centers))
                reg_dist = conv_reg(
                    attn_reg(feat, value_feat_reg, anchor_centers))
                pred_logit.append(cls_logit.flatten(2).transpose([0, 2, 1]))
                pred_dist.append(reg_dist.flatten(2).transpose([0, 2, 1]))
        else:
            for feat, feat_cls, feat_reg, conv_cls, conv_reg in zip(
                    feats, self.feat_cls, self.feat_reg, self.conv_cls,
                    self.conv_reg):
                cls_logit = conv_cls(feat_cls(feat))
                reg_dist = conv_reg(feat_reg(feat))
                pred_logit.append(cls_logit.flatten(2).transpose([0, 2, 1]))
                pred_dist.append(reg_dist.flatten(2).transpose([0, 2, 1]))
        pred_logit = paddle.concat(pred_logit, axis=1)
        pred_dist = paddle.concat(pred_dist, axis=1).exp()

        anchors = paddle.concat(anchors)
        anchors.stop_gradient = True
        stride_tensor_list = paddle.concat(stride_tensor_list)
        stride_tensor_list.stop_gradient = True

        if self.training:
            return self.get_loss([
                pred_logit, pred_dist, anchors, num_anchors_list,
                stride_tensor_list
            ], targets)
        else:
            pred_scores = F.sigmoid(pred_logit).transpose([0, 2, 1])
            pred_dist *= stride_tensor_list
            return pred_scores, pred_dist, anchors

    @staticmethod
    def _focal_loss(logit, label, alpha=0.25, gamma=2.0):
        score = F.sigmoid(logit)
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy_with_logits(
            logit, label, weight=weight, reduction='sum')
        return loss

    @staticmethod
    def _varifocal_loss(pred_logit, gt_score, label, alpha=0.75, gamma=2.0):
        pred_score = F.sigmoid(pred_logit)
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy_with_logits(
            pred_logit, gt_score, weight=weight, reduction='sum')
        return loss

    def get_loss(self, head_outs, gt_meta):
        pred_logit, pred_dist, anchors, num_anchors_list, stride_tensor_list = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None

        # distance2bbox
        anchor_centers = bbox_center(anchors)
        pred_bboxes = batch_distance2bbox(anchor_centers,
                                          pred_dist * stride_tensor_list)
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores, assigned_ious = self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                gt_scores=gt_scores,
                pred_bboxes=pred_bboxes.detach())
            alpha_l = 0.25
            if self.use_varifocal_loss:
                assigned_scores = assigned_ious
        else:
            pred_scores = F.sigmoid(pred_logit.detach())
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores,
                pred_bboxes.detach(),
                anchor_centers,
                num_anchors_list,
                stride_tensor_list,
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                gt_scores=gt_scores)
            alpha_l = -1

        # rescale bbox
        assigned_bboxes /= stride_tensor_list
        pred_bboxes /= stride_tensor_list
        # classification loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels, self.num_classes)
            loss_cls = self._varifocal_loss(pred_logit, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(
                pred_logit, assigned_scores, alpha=alpha_l)

        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.astype(paddle.float32).sum()
        assigned_scores_sum = assigned_scores.sum()
        if core.is_compiled_with_dist(
        ) and parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        # bbox regression loss
        if num_pos > 0:
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bbox_pos = paddle.masked_select(pred_bboxes,
                                                 bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            # iou loss
            loss_iou = self.giou_loss(pred_bbox_pos,
                                      assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum
            # l1 loss
            loss_l1 = F.l1_loss(pred_bbox_pos, assigned_bboxes_pos)
        else:
            loss_iou = paddle.zeros([1])
            loss_l1 = paddle.zeros([1])

        loss_cls /= assigned_scores_sum
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou

        return {
            'loss': loss,
            'loss_class': loss_cls,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1,
        }

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_dist, anchors = head_outs

        pred_bboxes = batch_distance2bbox(
            bbox_center(anchors), pred_dist, img_shape)

        # scale bbox to origin
        scale_factor = scale_factor.flip([1]).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num

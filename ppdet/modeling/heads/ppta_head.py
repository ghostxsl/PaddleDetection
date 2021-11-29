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
from ..initializer import normal_, constant_, bias_init_with_prob
from ppdet.modeling.bbox_utils import bbox_center, batch_distance2bbox
from ..losses import GIoULoss
from ppdet.modeling.backbones.darknet import ConvBNLayer
from ppdet.modeling.assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.layers import ConvNormLayer

__all__ = ['PPTAHead']


class SEHeadFeat(nn.Layer):
    def __init__(self, feat_in=(1024, 512, 256), se_down_rate=8):
        super(SEHeadFeat, self).__init__()
        assert isinstance(feat_in, (list, tuple))
        self.feat_in = feat_in
        self.cls_se_conv = nn.LayerList()
        self.reg_se_conv = nn.LayerList()
        for in_channel in self.feat_in:
            self.cls_se_conv.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channel,
                        in_channel // se_down_rate,
                        1,
                        bias_attr=False),
                    nn.ReLU(),
                    nn.Conv2D(in_channel // se_down_rate, in_channel, 1)))
            self.reg_se_conv.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channel,
                        in_channel // se_down_rate,
                        1,
                        bias_attr=False),
                    nn.ReLU(),
                    nn.Conv2D(in_channel // se_down_rate, in_channel, 1)))

    def forward(self, fpn_feats):
        assert len(self.feat_in) == len(fpn_feats)
        cls_feats = []
        reg_feats = []
        for feat, cls_conv, reg_conv in zip(fpn_feats, self.cls_se_conv,
                                            self.reg_se_conv):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))

            se_feat_cls = F.sigmoid(cls_conv(avg_feat))
            cls_feats.append(feat * se_feat_cls)

            se_feat_reg = F.sigmoid(reg_conv(avg_feat))
            reg_feats.append(feat * se_feat_reg)

        return cls_feats, reg_feats


class EBFModule(nn.Layer):
    """
    extreme box feature (EBF) module

    """

    def __init__(self, feat_channels, act="relu"):
        super(EBFModule, self).__init__()
        self.feat_channels = feat_channels

        self.conv_feat = ConvBNLayer(
            self.feat_channels, self.feat_channels, padding=1, act=act)
        self.reg_conv = nn.Conv2D(self.feat_channels, 4, 3, padding=1)
        self.points_conv = nn.Conv2D(self.feat_channels, 4, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        constant_(self.reg_conv.weight)
        constant_(self.reg_conv.bias, math.log(5.0 / 2))
        normal_(self.points_conv.weight, std=0.001)
        normal_(self.points_conv.bias, std=0.001)

    def _make_points(self, points_delta, reg_bbox):
        x1, y1, x2, y2 = reg_bbox.split(4, axis=-1)
        h, w = y2 - y1, x2 - x1
        extreme_points = [(x1 + x2) / 2, (y1 + y2) / 2,
                          x1 + points_delta[..., 0:1] * w, y1, x1,
                          y1 + points_delta[..., 1:2] * h,
                          x1 + points_delta[..., 2:3] * w, y2, x2,
                          y1 + points_delta[..., 3:4] * h]
        return paddle.concat(extreme_points, axis=-1)

    def forward(self, feat, anchor_points, scale_weight=None):
        feat = self.conv_feat(feat)
        reg_dist = self.reg_conv(feat).exp()
        reg_dist = reg_dist.flatten(2).transpose([0, 2, 1])
        if scale_weight is not None:
            reg_dist = scale_weight(reg_dist)
        # [b, L, 4]
        reg_bbox = batch_distance2bbox(anchor_points, reg_dist)

        points_delta = F.sigmoid(self.points_conv(feat))
        points_delta = points_delta.flatten(2).transpose([0, 2, 1])
        # [b, L, 10]
        reg_points = self._make_points(points_delta, reg_bbox)

        return reg_bbox, reg_points


@register
class PPTAHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['nms', 'static_assigner', 'assigner']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5,
                 grid_cell_offset=0.5,
                 static_assigner_epoch=60,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.0,
                 }):
        super(PPTAHead, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.giou_loss = GIoULoss()
        self.loss_weight = loss_weight

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms

        self.feat_conv = SEHeadFeat(self.in_channels)
        # reg
        self.ebf_modules = nn.LayerList(
            [EBFModule(in_channel) for in_channel in self.in_channels])
        # cls
        self.cls_weights = nn.ParameterList([
            self.create_parameter(shape=[self.num_classes, in_channel * 5])
            for in_channel in self.in_channels
        ])
        self.cls_biases = self.create_parameter(
            [len(self.fpn_strides), self.num_classes])

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for w in self.cls_weights:
            constant_(w)
        constant_(self.cls_biases, bias_cls)

    def _extreme_sample_conv(self, feat, weight, offset_points, bias=None):
        """
        Args:
            offset_points (Tensor): shape[b, l, num_points * 2], "x,y"format
        """
        b, _, h, w = feat.shape
        offset_points = offset_points.reshape(
            [b, h, w, -1, 2]).transpose([0, 3, 1, 2, 4]).flatten(1, 2)
        normalize_shape = paddle.to_tensor([w, h], dtype='float32')
        reg_coord = offset_points / normalize_shape
        reg_coord = reg_coord.clip(0., 1.) * 2 - 1
        post_feat = F.grid_sample(feat, reg_coord)
        post_feat = paddle.matmul(weight, post_feat.reshape([b, -1, h * w]))
        if bias is not None:
            post_feat = post_feat + bias
        return post_feat

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"
        anchors, num_anchors_list, stride_tensor_list = \
            generate_anchors_for_grid_cell(feats,
                                           self.fpn_strides,
                                           self.grid_cell_scale,
                                           self.grid_cell_offset)
        cls_feats, reg_feats = self.feat_conv(feats)
        cls_logit_list, bbox_pred_list = [], []
        for cls_feat, reg_feat, ebf_module, cls_weight, cls_bias, anchor, stride in zip(
                cls_feats, reg_feats, self.ebf_modules, self.cls_weights,
                self.cls_biases, anchors, self.fpn_strides):
            # reg branch
            anchor_centers = bbox_center(anchor).unsqueeze(0) / stride
            reg_bbox, reg_points = ebf_module(reg_feat, anchor_centers)
            if not self.training:
                reg_bbox *= stride
            bbox_pred_list.append(reg_bbox)
            # cls branch
            cls_logit = self._extreme_sample_conv(cls_feat, cls_weight,
                                                  reg_points, cls_bias[:, None])
            cls_logit = cls_logit.transpose([0, 2, 1])
            cls_logit_list.append(cls_logit)

        cls_logit_list = paddle.concat(cls_logit_list, axis=1)
        bbox_pred_list = paddle.concat(bbox_pred_list, axis=1)
        anchors = paddle.concat(anchors)
        anchors.stop_gradient = True
        stride_tensor_list = paddle.concat(stride_tensor_list)
        stride_tensor_list.stop_gradient = True

        if self.training:
            return self.get_loss([
                cls_logit_list, bbox_pred_list, anchors, num_anchors_list,
                stride_tensor_list
            ], targets)
        else:
            return [
                cls_logit_list, bbox_pred_list, anchors, num_anchors_list,
                stride_tensor_list
            ]

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
        cls_logit, pred_bboxes, anchors, num_anchors_list, stride_tensor_list = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None

        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores, assigned_ious = self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                gt_scores=gt_scores,
                pred_bboxes=pred_bboxes.detach() * stride_tensor_list)
            assigned_scores = assigned_ious
        else:
            pred_scores = F.sigmoid(cls_logit.detach())
            # distance2bbox
            anchor_centers = bbox_center(anchors)
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores,
                pred_bboxes.detach() * stride_tensor_list,
                anchor_centers,
                num_anchors_list,
                stride_tensor_list,
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                gt_scores=gt_scores)

        # rescale bbox
        assigned_bboxes /= stride_tensor_list
        # classification loss
        one_hot_label = F.one_hot(assigned_labels, self.num_classes)
        loss_cls = self._varifocal_loss(cls_logit, assigned_scores,
                                        one_hot_label)
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
        cls_logit, pred_bboxes, _, _, _ = head_outs
        pred_scores = F.sigmoid(cls_logit).transpose([0, 2, 1])

        for i in range(len(pred_bboxes)):
            pred_bboxes[i, :, 0] = pred_bboxes[i, :, 0].clip(
                min=0, max=img_shape[i, 1])
            pred_bboxes[i, :, 1] = pred_bboxes[i, :, 1].clip(
                min=0, max=img_shape[i, 0])
            pred_bboxes[i, :, 2] = pred_bboxes[i, :, 2].clip(
                min=0, max=img_shape[i, 1])
            pred_bboxes[i, :, 3] = pred_bboxes[i, :, 3].clip(
                min=0, max=img_shape[i, 0])
        # scale bbox to origin
        scale_factor = scale_factor.flip([1]).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num

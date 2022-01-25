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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

import math
from paddle.fluid import core
from paddle.fluid.dygraph import parallel_helper
from ..bbox_utils import batch_distance2bbox
from ..losses import GIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.layers import ConvNormLayer
from ppdet.modeling.ops import get_static_shape

__all__ = ['PPYOLOEHead']


class ESEAttn(nn.Layer):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.act = act
        self.fc = nn.Conv2D(feat_channels, feat_channels, 1)
        self.conv = ConvNormLayer(feat_channels, feat_channels, 1, 1)

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        out = self.conv(feat * weight)
        return getattr(F, self.act)(out)


@register
class PPYOLOEHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=[14, 12, 10],
                 reg_num=None,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 use_cls_attn=False,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_input_size=[640, 640],
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 }):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.reg_num = reg_num if reg_num is not None else max(reg_max) + 1
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.use_cls_attn = use_cls_attn
        self.eval_input_size = eval_input_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        # stem
        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c))
            self.stem_reg.append(ESEAttn(in_c))
        # pred head
        self.pred_cls = nn.LayerList()
        self.pred_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2D(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2D(
                    in_c, 4 * self.reg_num, 3, padding=1))
        if self.use_cls_attn:
            # cls spatial attn
            self.attn_head = nn.LayerList()
            for in_c in self.in_channels:
                self.attn_head.append(nn.Conv2D(in_c, 1, 1))

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            normal_(reg_.weight, std=0.01)

        if self.use_cls_attn:
            for attn_ in self.attn_head:
                constant_(attn_.weight)
                constant_(attn_.bias, bias_cls)

        # register buffer tensor
        anchor_points, stride_tensor, proj_tensor = self._generate_anchors()
        self.register_buffer('anchor_points', anchor_points)
        self.register_buffer('stride_tensor', stride_tensor)
        self.register_buffer('proj_tensor', proj_tensor)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor, proj_tensor, step_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset, self.reg_max, self.reg_num)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat))
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            if self.use_cls_attn:
                cls_attn = F.sigmoid(self.attn_head[i](feat))
                cls_score = (F.sigmoid(cls_logit) * cls_attn).sqrt()
            else:
                cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor, proj_tensor, step_tensor
        ], targets)

    def _generate_anchors(self):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        proj_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            h = int(self.eval_input_size[0] / stride)
            w = int(self.eval_input_size[1] / stride)
            shift_x = paddle.arange(end=w) + self.grid_cell_offset
            shift_y = paddle.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype='float32')
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                paddle.full(
                    [h * w, 1], stride, dtype='float32'))
            proj_tensor.append(
                paddle.linspace(0, self.reg_max[i], self.reg_num).reshape(
                    [1, -1, 1]).tile([h * w, 1, 1]))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        proj_tensor = paddle.concat(proj_tensor)
        return anchor_points, stride_tensor, proj_tensor

    def forward_eval(self, feats):
        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat))
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            if self.use_cls_attn:
                cls_attn = F.sigmoid(self.attn_head[i](feat))
                cls_score = (F.sigmoid(cls_logit) * cls_attn).sqrt()
            else:
                cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_distri_list.append(reg_distri.reshape([b, 4 * self.reg_num, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=-1)

        return cls_score_list, reg_distri_list

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist, proj_tensor):
        b, l, _ = get_static_shape(pred_dist)
        pred_dist = F.softmax(
            pred_dist.reshape([b, l, 4, self.reg_num]),
            axis=-1).matmul(proj_tensor).reshape([b, l, 4])
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox, num_anchors_list):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        ltrb = paddle.concat([lt, rb], -1).split(num_anchors_list, -2)
        for i, ltrb_ in enumerate(ltrb):
            ltrb_.clip_(0, self.reg_max[i] - 0.01)
        return paddle.concat(ltrb, -2)

    def _df_loss(self, pred_dist, target, step_tensor):
        r"""
        Args:
            pred_dist: [n, 4, reg_num]
            target: [n, 4]
            step_tensor: [n, 4]
        """
        # make target_left and target_right
        target_left = (target / step_tensor).astype('int64')
        target_right = target_left + 1
        # make weight_left and weight_right
        weight_right = target - target_left.astype('float32') * step_tensor
        weight_left = target_right.astype('float32') * step_tensor - target
        # compute loss
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum,
                   num_anchors_list, step_tensor):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum
            # dfl
            dist_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, (self.reg_num) * 4])
            pred_dist_pos = paddle.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_num])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes,
                                                num_anchors_list)
            assigned_ltrb_pos = paddle.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])

            b, _ = get_static_shape(mask_positive)
            step_tensor = step_tensor.reshape([1, -1]).tile([b, 1])
            step_tensor_pos = paddle.masked_select(
                step_tensor, mask_positive).reshape([-1, 1]).tile([1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos,
                                     step_tensor_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_dfl = paddle.zeros([1])
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, anchors,\
        anchor_points, num_anchors_list, stride_tensor, proj_tensor, step_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri,
                                        proj_tensor)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    gt_scores=gt_scores,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                stride_tensor,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                gt_scores=gt_scores)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels, self.num_classes)
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(
                pred_scores, assigned_scores, alpha=alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if core.is_compiled_with_dist(
        ) and parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum, num_anchors_list, step_tensor)
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_distri = head_outs
        pred_bboxes = self._bbox_decode(self.anchor_points,
                                        pred_distri.transpose([0, 2, 1]),
                                        self.proj_tensor)
        pred_bboxes *= self.stride_tensor
        # scale bbox to origin
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num

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
from paddle.vision.ops import deform_conv2d
from ppdet.modeling.backbones.darknet import ConvBNLayer
from ppdet.modeling.assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.heads.tood_head import ScaleReg

__all__ = ['OATHead']


class HeadFeat(nn.Layer):
    """
    Head interior feature extractor

    Args:
        feat_in (List): Input channels of each scale, if scale_shared=False.
        feat_channels (int):
    """

    def __init__(self,
                 feat_in=(),
                 feat_channels=256,
                 num_stacked_convs=3,
                 scale_shared=True,
                 cls_residual=False,
                 act="relu",
                 norm_type='gn',
                 norm_groups=32):
        super(HeadFeat, self).__init__()
        assert isinstance(feat_in, (list, tuple))
        assert isinstance(feat_channels, int)

        self.feat_in = feat_in
        self.feat_channels = feat_channels
        self.num_stacked_convs = num_stacked_convs
        self.scale_shared = scale_shared
        self.cls_residual = cls_residual
        self.act = act

        if self.scale_shared:
            # scale shared
            self.cls_convs = nn.Sequential(*[
                ConvBNLayer(
                    self.feat_channels,
                    self.feat_channels,
                    padding=1,
                    groups=norm_groups,
                    norm_type=norm_type,
                    act=self.act) for _ in range(self.num_stacked_convs)
            ])
            self.reg_convs = nn.Sequential(*[
                ConvBNLayer(
                    self.feat_channels,
                    self.feat_channels,
                    padding=1,
                    groups=norm_groups,
                    norm_type=norm_type,
                    act=self.act) for _ in range(self.num_stacked_convs)
            ])
        else:
            # scale not shared
            assert len(self.feat_in) > 0
            self.cls_1x1 = nn.LayerList()
            self.cls_convs = nn.LayerList()
            self.reg_1x1 = nn.LayerList()
            self.reg_convs = nn.LayerList()
            for in_channel in self.feat_in:
                self.cls_1x1.append(
                    ConvBNLayer(
                        in_channel,
                        self.feat_channels,
                        filter_size=1,
                        padding=0,
                        groups=norm_groups,
                        norm_type=norm_type,
                        act=self.act))
                self.cls_convs.append(
                    nn.Sequential(*[
                        ConvBNLayer(
                            self.feat_channels,
                            self.feat_channels,
                            padding=1,
                            groups=norm_groups,
                            norm_type=norm_type,
                            act=self.act) for _ in range(self.num_stacked_convs)
                    ]))
                self.reg_1x1.append(
                    ConvBNLayer(
                        in_channel,
                        self.feat_channels,
                        filter_size=1,
                        padding=0,
                        groups=norm_groups,
                        norm_type=norm_type,
                        act=self.act))
                self.reg_convs.append(
                    nn.Sequential(*[
                        ConvBNLayer(
                            self.feat_channels,
                            self.feat_channels,
                            padding=1,
                            groups=norm_groups,
                            norm_type=norm_type,
                            act=self.act) for _ in range(self.num_stacked_convs)
                    ]))

    def forward(self, fpn_feats):
        cls_feats = []
        reg_feats = []
        if self.scale_shared:
            # scale shared
            for feat in fpn_feats:
                if self.cls_residual:
                    cls_feats.append(feat + self.cls_convs(feat))
                else:
                    cls_feats.append(self.cls_convs(feat))
                reg_feats.append(self.reg_convs(feat))
        else:
            # scale not shared
            assert len(self.feat_in) == len(fpn_feats)
            for feat, cls_1x1, cls_conv, reg_1x1, reg_conv in zip(
                    fpn_feats, self.cls_1x1, self.cls_convs, self.reg_1x1,
                    self.reg_convs):
                cls_feat = cls_1x1(feat)
                if self.cls_residual:
                    cls_feats.append(cls_feat + cls_conv(cls_feat))
                else:
                    cls_feats.append(cls_conv(cls_feat))
                reg_feats.append(reg_conv(reg_1x1(feat)))

        return cls_feats, reg_feats


class RFAModule(nn.Layer):
    """
    regression feature alignment (RFA) module

    """

    def __init__(self, feat_channels=256, num_inter_points=5):
        super(RFAModule, self).__init__()
        self.feat_channels = feat_channels
        self.num_inter_points = num_inter_points

        self.reg_conv = nn.Conv2D(feat_channels, 4, 3, padding=1)
        self.points_conv = nn.Conv2D(
            feat_channels, 4 + self.num_inter_points * 2, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        constant_(self.reg_conv.weight)
        constant_(self.reg_conv.bias, math.log(5.0 / 2))
        normal_(self.points_conv.weight, std=0.001)
        normal_(self.points_conv.bias, std=0.001)

    def _make_points(self, points_delta, reg_bbox):
        x1, y1, x2, y2 = reg_bbox.split(4, axis=-1)
        h, w = y2 - y1, x2 - x1
        extreme_points = [
            x1 + points_delta[..., 0:1] * w, y1, x1,
            y1 + points_delta[..., 1:2] * h, x1 + points_delta[..., 2:3] * w,
            y2, x2, y1 + points_delta[..., 3:4] * h
        ]
        inter_points = []
        for i in range(self.num_inter_points):
            inter_points.append(x1 + points_delta[..., 4 + i * 2:4 + i * 2 + 1]
                                * w)
            inter_points.append(y1 + points_delta[..., 4 + i * 2 + 1:4 + i * 2 +
                                                  2] * h)

        return paddle.concat(extreme_points + inter_points, axis=-1)

    def forward(self, feat, anchor_points, scale_weight=None):
        reg_dist = self.reg_conv(feat).exp()
        reg_dist = reg_dist.flatten(2).transpose([0, 2, 1])
        if scale_weight is not None:
            reg_dist = scale_weight(reg_dist)
        reg_bbox = batch_distance2bbox(anchor_points, reg_dist)

        points_delta = F.sigmoid(self.points_conv(feat))
        points_delta = points_delta.flatten(2).transpose([0, 2, 1])
        reg_points = self._make_points(points_delta, reg_bbox)

        return reg_bbox, reg_points


@register
class OATHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['nms', 'coarse_assigner', 'refined_assigner']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 feat_channels=256,
                 stacked_convs=3,
                 fpn_strides=(32, 16, 8),
                 dcn_kernel_size=3,
                 scale_shared=False,
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 norm_type='gn',
                 norm_groups=32,
                 act='relu',
                 loss_weight={
                     'class': 1.0,
                     'bbox': 1.0,
                     'iou': 2.0,
                     'coarse': 1.0,
                     'refined': 2.0,
                 },
                 nms='MultiClassNMS',
                 coarse_assigner='ATSSAssigner',
                 refined_assigner='ATSSAssigner'):
        super(OATHead, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        assert dcn_kernel_size > 2
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.fpn_strides = fpn_strides
        self.dcn_kernel_size = dcn_kernel_size
        # make base conv offset, "y,x"format
        half_dcn_kernel_size = self.dcn_kernel_size // 2
        shift_x = paddle.arange(
            start=-half_dcn_kernel_size, end=half_dcn_kernel_size + 1)
        shift_y = paddle.arange(
            start=-half_dcn_kernel_size, end=half_dcn_kernel_size + 1)
        shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
        self.conv_offset = paddle.stack(
            [shift_y, shift_x],
            axis=-1).reshape([-1, 2, 1, 1]).astype('float32')

        self.scale_shared = scale_shared
        self.act = act

        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.coarse_assigner = coarse_assigner
        self.refined_assigner = refined_assigner
        self.loss_weight = loss_weight
        self.giou_loss = GIoULoss()
        self.nms = nms

        self.inter_convs = HeadFeat(self.in_channels, self.feat_channels,
                                    self.stacked_convs, self.scale_shared, True,
                                    self.act, norm_type, norm_groups)

        if self.scale_shared:
            # reg branch
            self.rfa_modules = RFAModule(self.feat_channels,
                                         self.dcn_kernel_size**2 - 4)
            self.rfa_dcn_weights = self.create_parameter(shape=[
                4, self.feat_channels, self.dcn_kernel_size,
                self.dcn_kernel_size
            ])
            self.rfa_dcn_biases = self.create_parameter(shape=[4])
            self.coarse_scale_regs = nn.LayerList(
                [ScaleReg() for _ in self.fpn_strides])
            self.refined_scale_regs = nn.LayerList(
                [ScaleReg() for _ in self.fpn_strides])
            # cls branch
            self.cfa_modules = nn.Conv2D(
                self.feat_channels,
                self.dcn_kernel_size * self.dcn_kernel_size * 2,
                3,
                padding=1)
            self.cfa_dcn_weights = self.create_parameter(shape=[
                self.num_classes, self.feat_channels, self.dcn_kernel_size,
                self.dcn_kernel_size
            ])
            self.cfa_dcn_biases = self.create_parameter(
                shape=[self.num_classes])
        else:
            # reg branch
            self.rfa_modules = nn.LayerList([
                RFAModule(self.feat_channels, self.dcn_kernel_size**2 - 4)
                for _ in self.fpn_strides
            ])
            self.rfa_dcn_weights = self.create_parameter(shape=[
                len(self.fpn_strides), 4, self.feat_channels,
                self.dcn_kernel_size, self.dcn_kernel_size
            ])
            self.rfa_dcn_biases = self.create_parameter(
                shape=[len(self.fpn_strides), 4])
            # cls branch
            self.cfa_modules = nn.LayerList([
                nn.Conv2D(
                    self.feat_channels,
                    self.dcn_kernel_size * self.dcn_kernel_size * 2,
                    3,
                    padding=1) for _ in self.fpn_strides
            ])
            self.cfa_dcn_weights = self.create_parameter(shape=[
                len(self.fpn_strides), self.num_classes, self.feat_channels,
                self.dcn_kernel_size, self.dcn_kernel_size
            ])
            self.cfa_dcn_biases = self.create_parameter(
                shape=[len(self.fpn_strides), self.num_classes])

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        constant_(self.rfa_dcn_weights)
        constant_(self.rfa_dcn_biases, math.log(5.0 / 2))
        constant_(self.cfa_dcn_weights)
        constant_(self.cfa_dcn_biases, bias_cls)
        if self.scale_shared:
            constant_(self.cfa_modules.weight)
        else:
            for cfa_module in self.cfa_modules:
                constant_(cfa_module.weight)

    def _constraint_deform_conv(self, feat, weight, offset_points, bias=None):
        """
        Args:
            offset_points (Tensor): shape[b, l, num_points * 2], "x,y"format
        """
        b, c, h, w = feat.shape
        offset_points = offset_points.reshape(
            [b, h, w, -1, 2]).flip(-1).transpose([0, 3, 4, 1, 2])
        shift_x = paddle.arange(end=w)
        shift_y = paddle.arange(end=h)
        shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
        coordinate_yx = paddle.stack([shift_y, shift_x]).astype(feat.dtype)
        deform_offset = offset_points - coordinate_yx - self.conv_offset
        deform_offset = deform_offset.flatten(1, 2)
        y = deform_conv2d(feat, deform_offset, weight, bias, padding=1)
        return y

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        anchors, num_anchors_list, stride_tensor_list = \
            generate_anchors_for_grid_cell(feats,
                                           self.fpn_strides,
                                           self.grid_cell_scale,
                                           self.grid_cell_offset)
        cls_feats, reg_feats = self.inter_convs(feats)
        cls_logit_list, coarse_bbox_list, refined_bbox_list = [], [], []
        if self.scale_shared:
            for cls_feat, reg_feat, coarse_scale, refined_scale, anchor, stride in zip(
                    cls_feats, reg_feats, self.coarse_scale_regs,
                    self.refined_scale_regs, anchors, self.fpn_strides):
                # reg branch
                anchor_centers = bbox_center(anchor).unsqueeze(0) / stride
                coarse_reg_bbox, reg_points = self.rfa_modules(
                    reg_feat, anchor_centers, coarse_scale)
                coarse_bbox_list.append(coarse_reg_bbox)

                refined_reg_dist = self._constraint_deform_conv(
                    reg_feat, self.rfa_dcn_weights, reg_points,
                    self.rfa_dcn_biases)
                refined_reg_dist = refined_reg_dist.exp().flatten(2).transpose(
                    [0, 2, 1])
                refined_reg_bbox = batch_distance2bbox(
                    anchor_centers, refined_scale(refined_reg_dist))
                if not self.training:
                    refined_reg_bbox *= stride
                refined_bbox_list.append(refined_reg_bbox)
                # cls branch
                cls_points_delta = self.cfa_modules(cls_feat).exp()
                cls_points_delta = cls_points_delta.flatten(2).transpose(
                    [0, 2, 1])
                cls_points = reg_points.detach() * cls_points_delta
                cls_logit = self._constraint_deform_conv(
                    cls_feat, self.cfa_dcn_weights, cls_points,
                    self.cfa_dcn_biases)
                cls_logit = cls_logit.flatten(2).transpose([0, 2, 1])
                cls_logit_list.append(cls_logit)
        else:
            for cls_feat, reg_feat, rfa_module, rfa_dcn_weight, rfa_dcn_bias,\
                cfa_module, cfa_dcn_weight, cfa_dcn_bias, anchor, stride in zip(
                    cls_feats, reg_feats,
                    self.rfa_modules, self.rfa_dcn_weights, self.rfa_dcn_biases,
                    self.cfa_modules, self.cfa_dcn_weights, self.cfa_dcn_biases,
                    anchors, self.fpn_strides):
                # reg branch
                anchor_centers = bbox_center(anchor).unsqueeze(0) / stride
                coarse_reg_bbox, reg_points = rfa_module(reg_feat,
                                                         anchor_centers)
                coarse_bbox_list.append(coarse_reg_bbox)

                refined_reg_dist = self._constraint_deform_conv(
                    reg_feat, rfa_dcn_weight, reg_points, rfa_dcn_bias)
                refined_reg_dist = refined_reg_dist.exp().flatten(2).transpose(
                    [0, 2, 1])
                refined_reg_bbox = batch_distance2bbox(anchor_centers,
                                                       refined_reg_dist)
                if not self.training:
                    refined_reg_bbox *= stride
                refined_bbox_list.append(refined_reg_bbox)
                # cls branch
                cls_points_delta = cfa_module(cls_feat).exp()
                cls_points_delta = cls_points_delta.flatten(2).transpose(
                    [0, 2, 1])
                cls_points = reg_points.detach() * cls_points_delta
                cls_logit = self._constraint_deform_conv(
                    cls_feat, cfa_dcn_weight, cls_points, cfa_dcn_bias)
                cls_logit = cls_logit.flatten(2).transpose([0, 2, 1])
                cls_logit_list.append(cls_logit)

        cls_logit_list = paddle.concat(cls_logit_list, axis=1)
        coarse_bbox_list = paddle.concat(coarse_bbox_list, axis=1)
        refined_bbox_list = paddle.concat(refined_bbox_list, axis=1)
        anchors = paddle.concat(anchors)
        anchors.stop_gradient = True
        stride_tensor_list = paddle.concat(stride_tensor_list).unsqueeze(0)
        stride_tensor_list.stop_gradient = True

        if self.training:
            return self.get_loss([
                cls_logit_list, coarse_bbox_list, refined_bbox_list, anchors,
                num_anchors_list, stride_tensor_list
            ], targets)
        else:
            return [
                cls_logit_list, coarse_bbox_list, refined_bbox_list, anchors,
                num_anchors_list, stride_tensor_list
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
        cls_logit, coarse_bbox, refined_bbox, anchors, num_anchors_list, stride_tensor_list = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None
        # label assignment
        # TODO: when coarse_assigner and refined_assigner is different, use different assigner
        assigned_labels, assigned_bboxes, assigned_scores, _ = self.coarse_assigner(
            anchors,
            num_anchors_list,
            gt_labels,
            gt_bboxes,
            bg_index=self.num_classes,
            pred_bboxes=refined_bbox.detach() * stride_tensor_list)

        # rescale bbox
        assigned_bboxes /= stride_tensor_list
        # classification loss
        loss_cls = self._focal_loss(cls_logit, assigned_scores, alpha=-1)
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.astype(paddle.float32).sum()
        if core.is_compiled_with_dist(
        ) and parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.all_reduce(num_pos)
            num_pos = paddle.clip(
                num_pos / paddle.distributed.get_world_size(), min=1)
        # bbox regression loss
        if num_pos > 0:
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            coarse_bbox_pos = paddle.masked_select(coarse_bbox,
                                                   bbox_mask).reshape([-1, 4])
            refined_bbox_pos = paddle.masked_select(refined_bbox,
                                                    bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            # iou loss
            loss_iou_coarse = self.giou_loss(coarse_bbox_pos,
                                             assigned_bboxes_pos)
            loss_iou_coarse = loss_iou_coarse.sum() / num_pos
            loss_iou_refined = self.giou_loss(refined_bbox_pos,
                                              assigned_bboxes_pos)
            loss_iou_refined = loss_iou_refined.sum() / num_pos
            # l1 loss
            loss_l1_coarse = F.l1_loss(coarse_bbox_pos, assigned_bboxes_pos)
            loss_l1_refined = F.l1_loss(refined_bbox_pos, assigned_bboxes_pos)
        else:
            loss_iou_coarse = paddle.zeros([1])
            loss_iou_refined = paddle.zeros([1])
            loss_l1_coarse = paddle.zeros([1])
            loss_l1_refined = paddle.zeros([1])

        loss_cls /= num_pos
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['coarse'] * loss_iou_coarse + \
               self.loss_weight['refined'] * loss_iou_refined

        return {
            'loss': loss,
            'loss_class': loss_cls,
            'loss_iou_coarse': loss_iou_coarse,
            'loss_l1_coarse': loss_l1_coarse,
            'loss_iou_refined': loss_iou_refined,
            'loss_l1_refined': loss_l1_refined,
        }

    def post_process(self, head_outs, img_shape, scale_factor):
        cls_logit, coarse_bbox, refined_bbox, _, _, _ = head_outs
        pred_scores = F.sigmoid(cls_logit).transpose([0, 2, 1])

        pred_bboxes = refined_bbox
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

    def __init__(self,
                 feat_channels=256,
                 act="mish",
                 norm_type='gn',
                 norm_groups=32):
        super(EBFModule, self).__init__()
        self.feat_channels = feat_channels

        self.conv_feat = ConvBNLayer(
            self.feat_channels,
            self.feat_channels,
            padding=1,
            groups=norm_groups,
            norm_type=norm_type,
            act=act)
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
    __shared__ = ['num_classes', 'data_format']
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
                 act='mish',
                 nms='MultiClassNMS',
                 loss_weight={
                     'class': 1.0,
                     'coarse': 1.0,
                     'refined': 2.0,
                 },
                 data_format='NCHW'):
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
        self.act = act
        self.data_format = data_format

        self.feat_conv = SEHeadFeat(self.in_channels)
        # reg
        self.ebf_modules = nn.LayerList(
            [EBFModule(in_channel) for in_channel in self.in_channels])
        self.ebf_weights = nn.ParameterList([
            self.create_parameter(shape=[4, in_channel * 5])
            for in_channel in self.in_channels
        ])
        self.ebf_biases = self.create_parameter([len(self.fpn_strides), 4])
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
        for w in self.ebf_weights:
            constant_(w)
        constant_(self.ebf_biases, math.log(5.0 / 2))
        for w in self.cls_weights:
            constant_(w)
        constant_(self.cls_biases, bias_cls)

    def _extreme_sample_conv(self, feat, weight, offset_points, bias=None):
        """
        Args:
            offset_points (Tensor): shape[b, l, num_points * 2], "x,y"format
        """
        b, c, h, w = feat.shape
        offset_points = offset_points.reshape(
            [b, h, w, -1, 2]).transpose([0, 3, 1, 2, 4]).flatten(1, 2)
        normalize_shape = paddle.to_tensor([w, h], dtype='float32')
        reg_coord = offset_points / normalize_shape * 2 - 1
        reg_coord = reg_coord.clip(-1, 1)
        post_feat = F.grid_sample(feat, reg_coord, align_corners=False)
        post_feat = paddle.matmul(weight, post_feat.reshape([b, -1, h * w]))
        return post_feat.reshape([b, -1, h, w])

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"
        anchors, num_anchors_list, stride_tensor_list = \
            generate_anchors_for_grid_cell(feats,
                                           self.fpn_strides,
                                           self.grid_cell_scale,
                                           self.grid_cell_offset)
        cls_feats, reg_feats = self.feat_conv(feats)
        cls_logit_list, coarse_bbox_list, refined_bbox_list = [], [], []
        for cls_feat, reg_feat, ebf_module, ebf_weight, ebf_bias,\
            cls_weight, cls_bias, anchor, stride in zip(
                cls_feats, reg_feats,
                self.ebf_modules, self.ebf_weights, self.ebf_biases,
                self.cls_weights, self.cls_biases,
                anchors, self.fpn_strides):
            # reg branch
            anchor_centers = bbox_center(anchor).unsqueeze(0) / stride
            coarse_reg_bbox, reg_points = ebf_module(reg_feat, anchor_centers)
            coarse_bbox_list.append(coarse_reg_bbox)

            refined_reg_dist = self._extreme_sample_conv(reg_feat, ebf_weight,
                                                         reg_points, ebf_bias)
            refined_reg_dist = refined_reg_dist.exp().flatten(2).transpose(
                [0, 2, 1])
            refined_reg_bbox = batch_distance2bbox(anchor_centers,
                                                   refined_reg_dist)
            if not self.training:
                refined_reg_bbox *= stride
            refined_bbox_list.append(refined_reg_bbox)
            # cls branch
            cls_logit = self._extreme_sample_conv(cls_feat, cls_weight,
                                                  reg_points, cls_bias)
            cls_logit = cls_logit.flatten(2).transpose([0, 2, 1])
            cls_logit_list.append(cls_logit)

        cls_logit_list = paddle.concat(cls_logit_list, axis=1)
        coarse_bbox_list = paddle.concat(coarse_bbox_list, axis=1)
        refined_bbox_list = paddle.concat(refined_bbox_list, axis=1)
        anchors = paddle.concat(anchors)
        anchors.stop_gradient = True
        stride_tensor_list = paddle.concat(stride_tensor_list).unsqueeze(0)
        stride_tensor_list.stop_gradient = True

        if self.training:
            return self.get_loss([
                cls_logit_list, coarse_bbox_list, refined_bbox_list, anchors,
                num_anchors_list, stride_tensor_list
            ], targets)
        else:
            return [
                cls_logit_list, coarse_bbox_list, refined_bbox_list, anchors,
                num_anchors_list, stride_tensor_list
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
        cls_logit, coarse_bbox, refined_bbox, anchors, num_anchors_list, stride_tensor_list = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None

        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores, _ = self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                pred_bboxes=refined_bbox.detach() * stride_tensor_list)
        else:
            pred_scores = F.sigmoid(cls_logit.detach())
            # distance2bbox
            anchor_centers = bbox_center(anchors)
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores,
                refined_bbox.detach(),
                anchor_centers,
                num_anchors_list,
                stride_tensor_list.squeeze(0),
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
        if core.is_compiled_with_dist(
        ) and parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.all_reduce(num_pos)
            num_pos = paddle.clip(
                num_pos / paddle.distributed.get_world_size(), min=1)
        # bbox regression loss
        if num_pos > 0:
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            coarse_bbox_pos = paddle.masked_select(coarse_bbox,
                                                   bbox_mask).reshape([-1, 4])
            refined_bbox_pos = paddle.masked_select(refined_bbox,
                                                    bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            # iou loss
            loss_iou_coarse = self.giou_loss(coarse_bbox_pos,
                                             assigned_bboxes_pos)
            loss_iou_coarse = loss_iou_coarse.sum() / num_pos
            loss_iou_refined = self.giou_loss(refined_bbox_pos,
                                              assigned_bboxes_pos)
            loss_iou_refined = loss_iou_refined.sum() / num_pos
            # l1 loss
            loss_l1_coarse = F.l1_loss(coarse_bbox_pos, assigned_bboxes_pos)
            loss_l1_refined = F.l1_loss(refined_bbox_pos, assigned_bboxes_pos)
        else:
            loss_iou_coarse = paddle.zeros([1])
            loss_iou_refined = paddle.zeros([1])
            loss_l1_coarse = paddle.zeros([1])
            loss_l1_refined = paddle.zeros([1])

        loss_cls /= num_pos
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['coarse'] * loss_iou_coarse + \
               self.loss_weight['refined'] * loss_iou_refined

        return {
            'loss': loss,
            'loss_class': loss_cls,
            'loss_iou_coarse': loss_iou_coarse,
            'loss_l1_coarse': loss_l1_coarse,
            'loss_iou_refined': loss_iou_refined,
            'loss_l1_refined': loss_l1_refined,
        }

    def post_process(self, head_outs, img_shape, scale_factor):
        cls_logit, coarse_bbox, refined_bbox, _, _, _ = head_outs
        pred_scores = F.sigmoid(cls_logit).transpose([0, 2, 1])

        pred_bboxes = refined_bbox
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

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

# The code is based on:
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/gfl_head.py

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ppdet.core.workspace import register
from ppdet.modeling.layers import ConvNormLayer
from ppdet.modeling.bbox_utils import distance2bbox, bbox2distance, batch_distance2bbox
from ppdet.data.transform.atss_assigner import bbox_overlaps

from functools import partial


class MIAODHead(nn.Layer):
    """Base class for DenseHeads."""

    def __init__(self):
        super(MIAODHead, self).__init__()

    def get_loss(self, head_outs, img_metas):
        y_head_f_1, y_head_f_2, y_head_f_r, y_head_cls = head_outs
        # Label set training
        if img_metas['stage_id'] == 0:
            outs = (y_head_f_1, y_head_f_r, y_head_cls)
            L_det_1 = self.L_det(outs, img_metas)
            outs = (y_head_f_2, y_head_f_r, y_head_cls)
            L_det_2 = self.L_det(outs, img_metas)
            l_det_cls = (L_det_1['l_det_cls'] + L_det_2['l_det_cls']) / 2
            l_det_loc = (L_det_1['l_det_loc'] + L_det_2['l_det_loc']) / 2
            l_imgcls = (L_det_1['l_imgcls'] + L_det_2['l_imgcls']) / 2
            l_det_dfl = (L_det_1['l_det_dfl'] + L_det_2['l_det_dfl']) / 2
            L_det = dict(
                l_det_cls=l_det_cls,
                l_det_loc=l_det_loc,
                l_det_dfl=l_det_dfl,
                l_imgcls=l_imgcls)
            # outs = (y_head_f_1, y_head_f_r, y_head_cls)
            # L_det = self.L_det(outs, img_metas)
            return L_det
        # Re-weighting and minimizing instance uncertainty
        elif img_metas['stage_id'] == 1:
            outs = (y_head_f_1, y_head_f_2, y_head_f_r, y_head_cls)
            loss = self.L_wave_min(outs, img_metas)
            L_wave_min = dict(
                l_det_cls=loss['l_det_cls'],
                l_det_loc=loss['l_det_loc'],
                l_wave_dis=loss['l_wave_dis'],
                l_imgcls=loss['l_imgcls'])
            if 'l_det_dfl' in loss:
                L_wave_min['l_det_dfl'] = loss['l_det_dfl']
            return L_wave_min

        # Re-weighting and maximizing instance uncertainty
        else:
            outs = (y_head_f_1, y_head_f_2, y_head_f_r, y_head_cls)
            loss = self.L_wave_max(outs, img_metas)
            L_wave_max = dict(
                l_det_cls=loss['l_det_cls'],
                l_det_loc=loss['l_det_loc'],
                l_wave_dis_minus=loss['l_wave_dis_minus'])
            if 'l_det_dfl' in loss:
                L_wave_max['l_det_dfl'] = loss['l_det_dfl']
            return L_wave_max


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class ScaleReg(nn.Layer):
    """
    Parameter for scaling the regression outputs.
    """

    def __init__(self):
        super(ScaleReg, self).__init__()
        self.scale_reg = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=1.)),
            dtype="float32")

    def forward(self, inputs):
        out = inputs * self.scale_reg
        return out


class Integral(nn.Layer):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             paddle.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape([-1, self.reg_max + 1]), axis=1)
        x = F.linear(x, self.project)
        if self.training:
            x = x.reshape([-1, 4])
        return x


@register
class MIAODGFLHead(MIAODHead):
    """
    GFLHead
    Args:
        conv_feat (object): Instance of 'FCOSFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_class (object): Instance of QualityFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 16.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox', 'nms'
    ]
    __shared__ = ['num_classes']

    def __init__(self,
                 conv_feat='FCOSFeat',
                 in_channels=256,
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 stacked_convs=4,
                 loss_class='QualityFocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 reg_max=16,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0):
        super(MIAODGFLHead, self).__init__()
        self.conv_feat = conv_feat
        self.dgqp_module = dgqp_module
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.stacked_convs = stacked_convs
        self.loss_qfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.reg_max = reg_max
        self.in_channels = in_channels
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.feat_channels = 256
        self.use_sigmoid = self.loss_qfl.use_sigmoid
        self.param_lambda = 0.5
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        self.l_imgcls = nn.BCELoss()

        self.f_1_convs = nn.LayerList()
        self.f_2_convs = nn.LayerList()
        self.f_r_convs = nn.LayerList()
        self.f_mil_convs = nn.LayerList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.f_1_convs.append(
                ConvNormLayer(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    norm_type='bn',
                    use_dcn=False,
                    bias_on=True,
                    lr_scale=1.))
            self.f_2_convs.append(
                ConvNormLayer(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    norm_type='bn',
                    use_dcn=False,
                    bias_on=True,
                    lr_scale=1.))
            self.f_r_convs.append(
                ConvNormLayer(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    norm_type='bn',
                    use_dcn=False,
                    bias_on=True,
                    lr_scale=1.))
            self.f_mil_convs.append(
                ConvNormLayer(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    norm_type='bn',
                    use_dcn=False,
                    bias_on=True,
                    lr_scale=1.))

        #assert self.num_anchors == 1, 'anchor free version'
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.f_1_gfl = nn.Conv2D(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.01)),
            bias_attr=ParamAttr(initializer=Constant(value=bias_init_value)))
        self.f_2_gfl = nn.Conv2D(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.01)),
            bias_attr=ParamAttr(initializer=Constant(value=bias_init_value)))
        self.f_r_gfl = nn.Conv2D(
            self.feat_channels,
            4 * (self.reg_max + 1),
            3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.01)),
            bias_attr=ParamAttr(initializer=Constant(value=bias_init_value)))
        self.f_mil_gfl = nn.Conv2D(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.01)),
            bias_attr=ParamAttr(initializer=Constant(value=bias_init_value)))

        self.scales_regs = []
        for i in range(len(self.fpn_stride)):
            lvl = int(math.log(int(self.fpn_stride[i]), 2))
            feat_name = 'p{}_feat'.format(lvl)
            scale_reg = self.add_sublayer(feat_name, ScaleReg())
            self.scales_regs.append(scale_reg)

        self.distribution_project = Integral(self.reg_max)

    def forward(self, fpn_feats, stage_id=None):
        assert len(fpn_feats) == len(
            self.fpn_stride
        ), "The size of fpn_feats is not equal to size of fpn_stride"
        if stage_id is not None:
            for p in self.parameters():
                if stage_id == 1:
                    p.stop_gradient = True
                else:
                    p.stop_gradient = False

        f1_list, f2_list, fr_list, fmil_list = [], [], [], []
        for stride, scale_reg, fpn_feat in zip(self.fpn_stride,
                                               self.scales_regs, fpn_feats):
            f_1_feat = fpn_feat
            f_2_feat = fpn_feat
            f_r_feat = fpn_feat
            f_mil_feat = fpn_feat
            for cls_conv1 in self.f_1_convs:
                f_1_feat = cls_conv1(f_1_feat)
            for cls_conv2 in self.f_2_convs:
                f_2_feat = cls_conv2(f_2_feat)
            for reg_conv in self.f_r_convs:
                f_r_feat = reg_conv(f_r_feat)
            for mil_conv in self.f_mil_convs:
                f_mil_feat = mil_conv(f_mil_feat)
            y_head_f_1 = self.f_1_gfl(f_1_feat)
            y_head_f_2 = self.f_2_gfl(f_2_feat)
            y_head_f_r = scale_reg(self.f_r_gfl(f_r_feat))
            y_head_f_mil = self.f_mil_gfl(f_mil_feat)
            y_head_cls_term2 = (y_head_f_1 + y_head_f_2) / 2
            y_head_cls_term2 = y_head_cls_term2.detach()

            y_head_f_mil = y_head_f_mil.transpose([0, 2, 3, 1]).reshape(
                [y_head_f_1.shape[0], -1, self.cls_out_channels])
            y_head_cls_term2 = y_head_cls_term2.transpose([0, 2, 3, 1]).reshape(
                [y_head_f_1.shape[0], -1, self.cls_out_channels])
            y_head_cls = F.softmax(
                y_head_f_mil, axis=2) * F.softmax(
                    F.sigmoid(y_head_cls_term2).max(2, keepdim=True), axis=1)

            # cls_score = self.gfl_head_cls(conv_cls_feat)
            # bbox_pred = scale_reg(self.gfl_head_reg(conv_reg_feat))

            if not self.training:

                y_head_f_1 = F.sigmoid(y_head_f_1.transpose([0, 2, 3, 1]))
                y_head_f_r = y_head_f_r.transpose([0, 2, 3, 1])
                b, cell_h, cell_w, _ = paddle.shape(y_head_f_1)
                y, x = self.get_single_level_center_point(
                    [cell_h, cell_w], stride, cell_offset=self.cell_offset)
                center_points = paddle.stack([x, y], axis=-1)
                y_head_f_1 = y_head_f_1.reshape([b, -1, self.cls_out_channels])
                y_head_f_r = self.distribution_project(y_head_f_r) * stride
                y_head_f_r = y_head_f_r.reshape([b, cell_h * cell_w, 4])

                # NOTE: If keep_ratio=False and image shape value that
                # multiples of 32, distance2bbox not set max_shapes parameter
                # to speed up model prediction. If need to set max_shapes,
                # please use inputs['im_shape'].
                y_head_f_r = batch_distance2bbox(
                    center_points, y_head_f_r, max_shapes=None)

            f1_list.append(y_head_f_1)
            f2_list.append(y_head_f_2)
            fr_list.append(y_head_f_r)
            fmil_list.append(y_head_cls)

        return (f1_list, f2_list, fr_list, fmil_list)

    def _images_to_levels(self, target, num_level_anchors):
        """
        Convert targets by image to targets by feature level.
        """
        level_targets = []
        start = 0
        for n in num_level_anchors:
            end = start + n
            level_targets.append(target[:, start:end].squeeze(0))
            start = end
        return level_targets

    def _grid_cells_to_center(self, grid_cells):
        """
        Get center location of each gird cell
        Args:
            grid_cells: grid cells of a feature map
        Returns:
            center points
        """
        cells_cx = (grid_cells[:, 2] + grid_cells[:, 0]) / 2
        cells_cy = (grid_cells[:, 3] + grid_cells[:, 1]) / 2
        return paddle.stack([cells_cx, cells_cy], axis=-1)

    def l_det(self, gfl_head_outs, gt_meta):
        cls_logits, bboxes_reg, _ = gfl_head_outs
        decode_bbox_preds = []
        center_and_strides = []
        featmap_sizes = [[featmap.shape[-2], featmap.shape[-1]]
                         for featmap in cls_logits]
        num_imgs = gt_meta['im_id'].shape[0]
        bbox_targets = gt_meta['bbox_targets']
        labels = gt_meta['labels']
        label_weights = gt_meta['label_weights']
        for featmap_size, stride, bbox_pred in zip(featmap_sizes,
                                                   self.fpn_stride, bboxes_reg):
            # center in origin image
            yy, xx = self.get_single_level_center_point(featmap_size, stride,
                                                        self.cell_offset)
            strides = paddle.full((len(xx), ), stride)
            center_and_stride = paddle.stack([xx, yy, strides, strides],
                                             -1).tile([num_imgs, 1, 1])
            center_and_strides.append(center_and_stride)
            center_in_feature = center_and_stride.reshape(
                [-1, 4])[:, :-2] / stride
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [num_imgs, -1, 4 * (self.reg_max + 1)])
            pred_distances = self.distribution_project(bbox_pred)
            decode_bbox_pred_wo_stride = distance2bbox(
                center_in_feature, pred_distances).reshape([num_imgs, -1, 4])
            decode_bbox_preds.append(decode_bbox_pred_wo_stride * stride)

        flatten_cls_preds = [
            cls_pred.transpose([0, 2, 3, 1]).reshape(
                [num_imgs, -1, self.cls_out_channels])
            for cls_pred in cls_logits
        ]
        flatten_cls_preds = paddle.concat(flatten_cls_preds, axis=1)
        flatten_bboxes = paddle.concat(decode_bbox_preds, axis=1)
        flatten_center_and_strides = paddle.concat(center_and_strides, axis=1)
        # num_level_anchors = [
        #     featmap.shape[-2] * featmap.shape[-1] for featmap in cls_logits
        # ]
        # grid_cells_list = self._images_to_levels(gt_meta['grid_cells'],
        #                                          num_level_anchors)
        # labels_list = self._images_to_levels(gt_meta['labels'],
        #                                      num_level_anchors)
        # label_weights_list = self._images_to_levels(gt_meta['label_weights'],
        #                                             num_level_anchors)
        # bbox_targets_list = self._images_to_levels(gt_meta['bbox_targets'],
        #                                            num_level_anchors)
        num_total_pos = sum(gt_meta['pos_num'])
        # num_total_pos = sum(pos_num_l)

        # try:
        #     num_total_pos = paddle.distributed.all_reduce(num_total_pos.clone(
        #     )) / paddle.distributed.get_world_size()
        # except:
        #     num_total_pos = max(num_total_pos, 1)
        num_total_pos = paddle.clip(num_total_pos, min=1)
        # CQY
        flatten_regs = [
            reg.transpose([0, 2, 3, 1]).reshape(
                [num_imgs, -1, 4 * (self.reg_max + 1)]) for reg in bboxes_reg
        ]
        flatten_regs = paddle.concat(flatten_regs, axis=1)
        flatten_center_and_strides = flatten_center_and_strides.reshape([-1, 4])
        flatten_cls_preds = flatten_cls_preds.reshape([-1, self.num_classes])
        # flatten_cls_preds = F.sigmoid(flatten_cls_preds)
        flatten_regs = flatten_regs.reshape([-1, 4 * (self.reg_max + 1)])
        flatten_bboxes = flatten_bboxes.reshape([-1, 4])
        flatten_bbox_targets = bbox_targets.reshape([-1, 4])
        flatten_labels = labels.reshape([-1])
        flatten_label_weights = label_weights.reshape([-1])
        bg_class_ind = self.num_classes
        pos_inds = paddle.nonzero(
            paddle.logical_and((flatten_labels >= 0),
                               (flatten_labels < bg_class_ind)),
            as_tuple=False).squeeze(1)
        # qfl
        score = np.zeros(flatten_labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = paddle.gather(
                flatten_bbox_targets, pos_inds, axis=0)
            pos_decode_bbox_pred = paddle.gather(
                flatten_bboxes, pos_inds, axis=0)
            pos_reg = paddle.gather(flatten_regs, pos_inds, axis=0)
            pos_center_and_strides = paddle.gather(
                flatten_center_and_strides, pos_inds, axis=0)
            pos_stride = pos_center_and_strides[:, -1].unsqueeze(1)
            pos_centers = pos_center_and_strides[:, :-2] / pos_stride

            weight_targets = F.sigmoid(flatten_cls_preds).detach()
            weight_targets = paddle.gather(
                weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)

            pos_level_bbox_targets = pos_bbox_targets / pos_stride
            bbox_iou = bbox_overlaps(
                pos_decode_bbox_pred.detach().numpy(),
                pos_bbox_targets.detach().numpy(),
                is_aligned=True)

            # pos_labels = paddle.gather(flatten_labels, pos_inds, axis=0)
            score[pos_inds.numpy()] = bbox_iou

            pred_corners = pos_reg.reshape([-1, self.reg_max + 1])
            target_corners = bbox2distance(pos_centers, pos_level_bbox_targets,
                                           self.reg_max).reshape([-1])
            # regression loss
            loss_bbox = paddle.sum(
                self.loss_bbox(pos_decode_bbox_pred,
                               pos_bbox_targets) * weight_targets)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets.expand([-1, 4]).reshape([-1]),
                avg_factor=4.0)
        else:
            loss_bbox = paddle.zeros([1])
            loss_dfl = paddle.zeros([1])
            weight_targets = paddle.to_tensor([0], dtype='float32')
        # num_pos_avg_per_gpu = num_total_pos
        score = paddle.to_tensor(score)
        loss_qfl = self.loss_qfl(
            flatten_cls_preds, (flatten_labels, score),
            weight=flatten_label_weights,
            avg_factor=num_total_pos)

        avg_factor = weight_targets.sum()
        # try:
        #     avg_factor = paddle.distributed.all_reduce(avg_factor.clone())
        #     avg_factor = paddle.clip(
        #         avg_factor / paddle.distributed.get_world_size(), min=1)
        # except:
        #     avg_factor = max(avg_factor.item(), 1)
        avg_factor = paddle.clip(avg_factor, min=1)

        if avg_factor <= 0:
            loss_qfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
            loss_bbox = paddle.to_tensor(
                0, dtype='float32', stop_gradient=False)
            loss_dfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
        else:
            loss_bbox = loss_bbox / avg_factor
            loss_dfl = loss_dfl / avg_factor

        loss_states = dict(
            l_det_cls=loss_qfl, l_det_loc=loss_bbox, l_det_dfl=loss_dfl)

        return loss_states

    # def l_det(self, gfl_head_outs, gt_meta):
    #     cls_logits, bboxes_reg, _, = gfl_head_outs
    #     num_level_anchors = [
    #         featmap.shape[-2] * featmap.shape[-1] for featmap in cls_logits
    #     ]
    #     grid_cells_list = self._images_to_levels(gt_meta['grid_cells'],
    #                                              num_level_anchors)
    #     labels_list = self._images_to_levels(gt_meta['labels'],
    #                                          num_level_anchors)
    #     label_weights_list = self._images_to_levels(gt_meta['label_weights'],
    #                                                 num_level_anchors)
    #     bbox_targets_list = self._images_to_levels(gt_meta['bbox_targets'],
    #                                                num_level_anchors)
    #     num_total_pos = sum(gt_meta['pos_num'])
    #     try:
    #         num_total_pos = paddle.distributed.all_reduce(num_total_pos.clone(
    #         )) / paddle.distributed.get_world_size()
    #     except:
    #         num_total_pos = max(num_total_pos, 1)

    #     loss_bbox_list, loss_dfl_list, loss_qfl_list, avg_factor = [], [], [], []
    #     for cls_score, bbox_pred, grid_cells, labels, label_weights, bbox_targets, stride in zip(
    #             cls_logits, bboxes_reg, grid_cells_list, labels_list,
    #             label_weights_list, bbox_targets_list, self.fpn_stride):
    #         grid_cells = grid_cells.reshape([-1, 4])
    #         cls_score = cls_score.transpose([0, 2, 3, 1]).reshape(
    #             [-1, self.cls_out_channels])
    #         bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
    #             [-1, 4 * (self.reg_max + 1)])
    #         bbox_targets = bbox_targets.reshape([-1, 4])
    #         labels = labels.reshape([-1])
    #         label_weights = label_weights.reshape([-1])

    #         bg_class_ind = self.num_classes
    #         pos_inds = paddle.nonzero(
    #             paddle.logical_and((labels >= 0), (labels < bg_class_ind)),
    #             as_tuple=False).squeeze(1)
    #         score = np.zeros(labels.shape)
    #         if len(pos_inds) > 0:
    #             pos_bbox_targets = paddle.gather(bbox_targets, pos_inds, axis=0)
    #             pos_bbox_pred = paddle.gather(bbox_pred, pos_inds, axis=0)
    #             pos_grid_cells = paddle.gather(grid_cells, pos_inds, axis=0)
    #             pos_grid_cell_centers = self._grid_cells_to_center(
    #                 pos_grid_cells) / stride

    #             weight_targets = F.sigmoid(cls_score.detach())
    #             weight_targets = paddle.gather(
    #                 weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)
    #             pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
    #             pos_decode_bbox_pred = distance2bbox(pos_grid_cell_centers,
    #                                                  pos_bbox_pred_corners)
    #             pos_decode_bbox_targets = pos_bbox_targets / stride
    #             bbox_iou = bbox_overlaps(
    #                 pos_decode_bbox_pred.detach().numpy(),
    #                 pos_decode_bbox_targets.detach().numpy(),
    #                 is_aligned=True)
    #             score[pos_inds.numpy()] = bbox_iou
    #             pred_corners = pos_bbox_pred.reshape([-1, self.reg_max + 1])
    #             target_corners = bbox2distance(pos_grid_cell_centers,
    #                                            pos_decode_bbox_targets,
    #                                            self.reg_max).reshape([-1])
    #             # regression loss
    #             loss_bbox = paddle.sum(
    #                 self.loss_bbox(pos_decode_bbox_pred,
    #                                pos_decode_bbox_targets) * weight_targets)

    #             # dfl loss
    #             loss_dfl = self.loss_dfl(
    #                 pred_corners,
    #                 target_corners,
    #                 weight=weight_targets.expand([-1, 4]).reshape([-1]),
    #                 avg_factor=4.0)
    #         else:
    #             print('negative batch')
    #             loss_bbox = paddle.zeros([1])
    #             loss_dfl = paddle.zeros([1])
    #             weight_targets = paddle.to_tensor([0], dtype='float32')

    #         # qfl loss
    #         score = paddle.to_tensor(score)
    #         loss_qfl = self.loss_qfl(
    #             cls_score, (labels, score),
    #             weight=label_weights,
    #             avg_factor=num_total_pos)
    #         loss_bbox_list.append(loss_bbox)
    #         loss_dfl_list.append(loss_dfl)
    #         loss_qfl_list.append(loss_qfl)
    #         avg_factor.append(weight_targets.sum())

    #     avg_factor = sum(avg_factor)
    #     try:
    #         avg_factor = paddle.distributed.all_reduce(avg_factor.clone())
    #         avg_factor = paddle.clip(
    #             avg_factor / paddle.distributed.get_world_size(), min=1)
    #     except:
    #         avg_factor = max(avg_factor.item(), 1)
    #     if avg_factor <= 0:
    #         loss_qfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
    #         loss_bbox = paddle.to_tensor(
    #             0, dtype='float32', stop_gradient=False)
    #         loss_dfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
    #     else:
    #         losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
    #         losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
    #         loss_qfl = sum(loss_qfl_list)
    #         loss_bbox = sum(losses_bbox)
    #         loss_dfl = sum(losses_dfl)

    #     loss_states = dict(
    #         l_det_cls=loss_qfl, l_det_loc=loss_bbox, l_det_dfl=loss_dfl)

    #     return loss_states

    def get_single_level_center_point(self, featmap_size, stride,
                                      cell_offset=0):
        """
        Generate pixel centers of a single stage feature map.
        Args:
            featmap_size: height and width of the feature map
            stride: down sample stride of the feature map
        Returns:
            y and x of the center points
        """
        h, w = featmap_size
        x_range = (paddle.arange(w, dtype='float32') + cell_offset) * stride
        y_range = (paddle.arange(h, dtype='float32') + cell_offset) * stride
        y, x = paddle.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        return y, x

    def post_process(self, gfl_head_outs, im_shape, scale_factor):
        cls_scores, _, bboxes_reg, _ = gfl_head_outs
        bboxes = paddle.concat(bboxes_reg, axis=1)
        # rescale: [h_scale, w_scale] -> [w_scale, h_scale, w_scale, h_scale]
        im_scale = scale_factor.flip([1]).tile([1, 2]).unsqueeze(1)
        bboxes /= im_scale
        mlvl_scores = paddle.concat(cls_scores, axis=1)
        mlvl_scores = mlvl_scores.transpose([0, 2, 1])
        bbox_pred, bbox_num, _ = self.nms(bboxes, mlvl_scores)

        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output

    def L_det(self, gfl_head_outs, gt_meta):
        """Compute losses of the head.

        Args:
            y_f (list[Tensor]): Box scores for each scale level
                Has shape (n, N * C, H, W)
            y_f_r (list[Tensor]): Box energies / deltas for each scale
                level with shape (n, N * 4, H, W)
            y_loc_img (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            y_cls_img (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            y_loc_img_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        cls_logits, bboxes_reg, y_head_cls = gfl_head_outs
        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in cls_logits
        ]
        loss_states = self.l_det(gfl_head_outs, gt_meta)

        labels_list = self._images_to_levels(gt_meta['labels'],
                                             num_level_anchors)

        #compute mil loss

        y_head_cls_1level, y_cls_1level = self.get_img_gtlabel_score(
            labels_list, y_head_cls)
        l_imgcls = self.l_imgcls(y_head_cls_1level, y_cls_1level)
        loss_states['l_imgcls'] = l_imgcls
        #loss_states['l_imgcls'] = paddle.to_tensor([0.0])
        return loss_states

    def get_img_gtlabel_score(self, y_cls_img, y_head_cls):
        '''
        y_cls_img: [BS, [class of gt]]
        y_head_cls: [strides of [N, h*w*n_anchors, classes]]

        return:
            y_head_cls_1level: [BS, classes] 全图预测cls值的sum
            y_cls_1level: [BS, classes] 全图gt class的one-hot
        '''
        y_head_cls_1level = paddle.zeros(
            [y_cls_img[0].shape[0], self.cls_out_channels])
        for y_head_cls_single in y_head_cls:
            y_head_cls_1level = paddle.maximum(y_head_cls_1level,
                                               y_head_cls_single.sum(1))
        y_head_cls_1level = paddle.clip(
            y_head_cls_1level, min=1e-5, max=1.0 - 1e-5)
        y_cls_1level = F.one_hot(
            paddle.concat(
                y_cls_img, axis=1),
            self.cls_out_channels + 1)[..., :-1].sum(1).clip(max=1.0)

        return y_head_cls_1level, y_cls_1level

    def l_wave_dis(self, y_head_f_1_single, y_head_f_2_single,
                   y_head_cls_single, y_head_f_r_single):
        y_head_f_1_single = y_head_f_1_single.transpose([0, 2, 3, 1]).reshape(
            [-1, self.cls_out_channels])
        y_head_f_2_single = y_head_f_2_single.transpose([0, 2, 3, 1]).reshape(
            [-1, self.cls_out_channels])
        y_head_f_1_single = nn.Sigmoid()(y_head_f_1_single)
        y_head_f_2_single = nn.Sigmoid()(y_head_f_2_single)
        # mil weight
        w_i = y_head_cls_single.detach()
        l_det_cls_all = (paddle.abs(y_head_f_1_single - y_head_f_2_single) *
                         w_i.reshape([-1, self.cls_out_channels])).mean(
                             axis=1).sum() * 0.5
        l_det_loc = paddle.to_tensor([0.0])
        return l_det_cls_all, l_det_loc

    # Re-weighting and minimizing instance uncertainty
    def L_wave_min(self, gfl_head_outs, gt_meta):

        y_f_1, y_f_2, y_f_r, y_head_cls = gfl_head_outs
        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in y_f_1
        ]
        anchor_list = self._images_to_levels(gt_meta['grid_cells'],
                                             num_level_anchors)
        y_cls = self._images_to_levels(gt_meta['labels'], num_level_anchors)
        label_weights_list = self._images_to_levels(gt_meta['label_weights'],
                                                    num_level_anchors)
        y_loc = self._images_to_levels(gt_meta['bbox_targets'],
                                       num_level_anchors)
        num_total_pos = sum(gt_meta['pos_num'])

        l_wave_dis, l_det_loc = multi_apply(self.l_wave_dis, y_f_1, y_f_2,
                                            y_head_cls, y_f_r)
        l_wave_dis, l_det_loc = paddle.add_n(l_wave_dis), paddle.add_n(
            l_det_loc)

        gfl_head_outs = (y_f_1, y_f_r, y_head_cls)
        loss_states1 = self.l_det(gfl_head_outs, gt_meta)
        l_det_cls1, l_det_loc1, l_det_dfl1 = loss_states1[
            'l_det_cls'], loss_states1['l_det_loc'], loss_states1['l_det_dfl']

        gfl_head_outs = (y_f_2, y_f_r, y_head_cls)
        loss_states2 = self.l_det(gfl_head_outs, gt_meta)
        l_det_cls2, l_det_loc2, l_det_dfl2 = loss_states2[
            'l_det_cls'], loss_states1['l_det_loc'], loss_states1['l_det_dfl']

        if gt_meta['is_unlabeled']:
            l_det_cls = paddle.to_tensor([0.0])
            l_det_loc = paddle.to_tensor([0.0])
            l_det_dfl = paddle.to_tensor([0.0])

            # compute mil loss
            y_head_cls_1level, y_pseudo = self.get_img_pseudolabel_score(
                (y_f_1, y_f_2), y_head_cls)
            l_imgcls = self.l_imgcls(y_head_cls_1level, y_pseudo)
            #l_imgcls = paddle.to_tensor([0.0])
        else:
            l_det_cls = (l_det_cls1 + l_det_cls2) / 2
            l_det_loc = (l_det_loc1 + l_det_loc2) / 2
            l_det_dfl = (l_det_dfl1 + l_det_dfl2) / 2
            l_wave_dis = paddle.to_tensor([0.0])

            # compute mil loss
            y_head_cls_1level, y_cls_1level = self.get_img_gtlabel_score(
                y_cls, y_head_cls)
            l_imgcls = self.l_imgcls(y_head_cls_1level, y_cls_1level)

        return dict(
            l_det_cls=l_det_cls,
            l_det_loc=l_det_loc,
            l_det_dfl=l_det_dfl,
            l_wave_dis=l_wave_dis,
            l_imgcls=l_imgcls)

    def get_img_pseudolabel_score(self, y_f, y_head_cls):
        batch_size = y_head_cls[0].shape[0]
        y_head_cls_1level = paddle.zeros([batch_size, self.cls_out_channels])
        y_pseudo = paddle.zeros([batch_size, self.cls_out_channels])
        # predict image pseudo label
        with paddle.no_grad():
            for s in range(len(y_f[0])):
                y_head_f_i = F.sigmoid(y_f[0][s].transpose([
                    0, 2, 3, 1
                ]).reshape([batch_size, -1, self.cls_out_channels]))
                y_head_f_i = F.sigmoid(y_f[1][s].transpose(
                    [0, 2, 3, 1]).reshape(
                        [batch_size, -1, self.cls_out_channels])) + y_head_f_i
                y_head_f_i = y_head_f_i.max(1)[0] / 2
                y_pseudo = paddle.maximum(y_pseudo, y_head_f_i)
            y_pseudo[y_pseudo >= 0.5] = 1
            y_pseudo[y_pseudo < 0.5] = 0
        # mil image score
        for y_head_cls_single in y_head_cls:
            y_head_cls_1level = paddle.maximum(y_head_cls_1level,
                                               y_head_cls_single.sum(1))
        y_head_cls_1level = paddle.clip(
            y_head_cls_1level, min=1e-5, max=1.0 - 1e-5)
        return y_head_cls_1level, y_pseudo.detach()

    def l_wave_dis_minus(self, y_head_f_1_single, y_head_f_2_single,
                         y_head_cls_single):
        y_head_f_1_single = y_head_f_1_single.transpose([0, 2, 3, 1]).reshape(
            [-1, self.cls_out_channels])
        y_head_f_2_single = y_head_f_2_single.transpose([0, 2, 3, 1]).reshape(
            [-1, self.cls_out_channels])
        y_head_f_1_single = nn.Sigmoid()(y_head_f_1_single)
        y_head_f_2_single = nn.Sigmoid()(y_head_f_2_single)
        # mil weight
        w_i = y_head_cls_single.detach()
        l_det_cls_all = ((1 - paddle.abs(y_head_f_1_single - y_head_f_2_single))
                         * w_i.reshape([-1, self.cls_out_channels])).mean(
                             axis=1).sum() * self.param_lambda
        l_det_loc = paddle.to_tensor([0.0])
        return l_det_cls_all, l_det_loc

    # Re-weighting and maximizing instance uncertainty
    def L_wave_max(self, gfl_head_outs, gt_meta):

        y_f_1, y_f_2, y_f_r, y_head_cls = gfl_head_outs
        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in y_f_1
        ]
        anchor_list = self._images_to_levels(gt_meta['grid_cells'],
                                             num_level_anchors)
        y_cls = self._images_to_levels(gt_meta['labels'], num_level_anchors)
        label_weights_list = self._images_to_levels(gt_meta['label_weights'],
                                                    num_level_anchors)
        y_loc = self._images_to_levels(gt_meta['bbox_targets'],
                                       num_level_anchors)

        l_wave_dis_minus, l_det_loc = multi_apply(self.l_wave_dis_minus, y_f_1,
                                                  y_f_2, y_head_cls)
        l_wave_dis_minus, l_det_loc = paddle.add_n(
            l_wave_dis_minus), paddle.add_n(l_det_loc)

        gfl_head_outs = (y_f_1, y_f_r, y_head_cls)
        loss_states1 = self.l_det(gfl_head_outs, gt_meta)
        l_det_cls1, l_det_loc1, l_det_dfl1 = loss_states1[
            'l_det_cls'], loss_states1['l_det_loc'], loss_states1['l_det_dfl']

        gfl_head_outs = (y_f_2, y_f_r, y_head_cls)
        loss_states2 = self.l_det(gfl_head_outs, gt_meta)
        l_det_cls2, l_det_loc2, l_det_dfl2 = loss_states2[
            'l_det_cls'], loss_states1['l_det_loc'], loss_states1['l_det_dfl']

        if gt_meta['is_unlabeled']:
            l_det_cls = paddle.to_tensor([0.0])
            l_det_loc = paddle.to_tensor([0.0])
            l_det_dfl = paddle.to_tensor([0.0])
        else:
            l_det_cls = (l_det_cls1 + l_det_cls2) / 2
            l_det_loc = (l_det_loc1 + l_det_loc2) / 2
            l_det_dfl = (l_det_dfl1 + l_det_dfl2) / 2
            l_wave_dis_minus = paddle.to_tensor([0.0])
        return dict(
            l_det_cls=l_det_cls,
            l_det_loc=l_det_loc,
            l_det_dfl=l_det_dfl,
            l_wave_dis_minus=l_wave_dis_minus)

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register

import math
from paddle.fluid import core
from paddle.fluid.dygraph import parallel_helper
from ..bbox_utils import bbox_center, batch_distance2bbox
from ..losses import GIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.layers import ConvNormLayer
from .tood_head import TaskDecomposition, ScaleReg
from paddle.vision.ops import deform_conv2d
from ppdet.modeling.ops import get_static_shape, mish

eps = 1e-12


def _de_sigmoid(x, eps=1e-7):
    x = paddle.clip(x, eps, 1. / eps)
    x = paddle.clip(1. / x - 1., eps, 1. / eps)
    x = -paddle.log(x)
    return x


@register
class YOLOv3Head(nn.Layer):
    __shared__ = ['num_classes', 'data_format']
    __inject__ = ['loss']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 num_classes=80,
                 loss='YOLOv3Loss',
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 data_format='NCHW'):
        """
        Head for YOLOv3 network

        Args:
            num_classes (int): number of foreground classes
            anchors (list): anchors
            anchor_masks (list): anchor masks
            loss (object): YOLOv3Loss instance
            iou_aware (bool): whether to use iou_aware
            iou_aware_factor (float): iou aware factor
            data_format (str): data format, NCHW or NHWC
        """
        super(YOLOv3Head, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss = loss

        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor

        self.parse_anchor(anchors, anchor_masks)
        self.num_outputs = len(self.anchors)
        self.data_format = data_format

        self.yolo_outputs = []
        for i in range(len(self.anchors)):

            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)
            name = 'yolo_output.{}'.format(i)
            conv = nn.Conv2D(
                in_channels=self.in_channels[i],
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                data_format=data_format,
                bias_attr=ParamAttr(regularizer=L2Decay(0.)))
            conv.skip_quant = True
            yolo_output = self.add_sublayer(name, conv)
            self.yolo_outputs.append(yolo_output)

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):
            yolo_output = self.yolo_outputs[i](feat)
            if self.data_format == 'NHWC':
                yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
            yolo_outputs.append(yolo_output)

        if self.training:
            return self.loss(yolo_outputs, targets, self.anchors)
        else:
            if self.iou_aware:
                y = []
                for i, out in enumerate(yolo_outputs):
                    na = len(self.anchors[i])
                    ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                    b, c, h, w = x.shape
                    no = c // na
                    x = x.reshape((b, na, no, h * w))
                    ioup = ioup.reshape((b, na, 1, h * w))
                    obj = x[:, :, 4:5, :]
                    ioup = F.sigmoid(ioup)
                    obj = F.sigmoid(obj)
                    obj_t = (obj**(1 - self.iou_aware_factor)) * (
                        ioup**self.iou_aware_factor)
                    obj_t = _de_sigmoid(obj_t)
                    loc_t = x[:, :, :4, :]
                    cls_t = x[:, :, 5:, :]
                    y_t = paddle.concat([loc_t, obj_t, cls_t], axis=2)
                    y_t = y_t.reshape((b, c, h, w))
                    y.append(y_t)
                return y
            else:
                return yolo_outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }


@register
class PPYOLOHead(nn.Layer):
    __shared__ = ['num_classes', 'data_format']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 static_assigner_epoch=4,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 use_varifocal_loss=True,
                 loss_weight={'class': 1.0,
                              'iou': 2.0},
                 data_format='NCHW'):
        super(PPYOLOHead, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        self.data_format = data_format
        self.use_varifocal_loss = use_varifocal_loss

        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        for in_channel in self.in_channels:
            self.cls_convs.append(
                nn.Conv2D(
                    in_channels=in_channel,
                    out_channels=self.num_classes,
                    kernel_size=1,
                    data_format=data_format))
            self.reg_convs.append(
                nn.Conv2D(
                    in_channels=in_channel,
                    out_channels=4,
                    kernel_size=1,
                    data_format=data_format))

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_reg = math.log(self.grid_cell_scale / 2)
        for conv in self.cls_convs:
            constant_(conv.weight)
            constant_(conv.bias, bias_cls)
        for conv in self.reg_convs:
            constant_(conv.weight)
            constant_(conv.bias, bias_reg)

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides)
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale,
            self.grid_cell_offset)

        pred_scores, pred_dist = [], []
        for feat, conv_cls, conv_reg in zip(feats, self.cls_convs,
                                            self.reg_convs):
            cls_logit = conv_cls(feat)
            reg_dist = conv_reg(feat)
            if self.data_format == 'NCHW':
                cls_logit = cls_logit.transpose([0, 2, 3, 1])
                reg_dist = reg_dist.transpose([0, 2, 3, 1])
            pred_scores.append(cls_logit.flatten(1, 2))
            pred_dist.append(reg_dist.flatten(1, 2))
        pred_scores = F.sigmoid(paddle.concat(pred_scores, 1))
        pred_dist = paddle.concat(pred_dist, 1).exp()

        if self.training:
            return self.get_loss([
                pred_scores, pred_dist, anchors, anchor_points,
                num_anchors_list, stride_tensor
            ], targets)
        else:
            return pred_scores, pred_dist, anchor_points, stride_tensor

    def _focal_loss(self, score, label, alpha=0.25, gamma=2.0):
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

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_dist, anchors, anchor_points,\
            num_anchors_list, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points / stride_tensor,
                                          pred_dist)
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None

        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                gt_scores=gt_scores,
                pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                stride_tensor,
                gt_labels,
                gt_bboxes,
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

        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        assigned_scores_sum = assigned_scores.sum()
        if core.is_compiled_with_dist(
        ) and parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        # pos/neg loss
        if num_pos > 0:
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            # iou loss
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum
            # l1 loss
            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])

        loss_cls /= assigned_scores_sum
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1
        }
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_scores = pred_scores.transpose([0, 2, 1])
        pred_bboxes = batch_distance2bbox(anchor_points,
                                          pred_dist * stride_tensor)

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
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num


@register
class PPTOODHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 feat_channels=256,
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 stacked_convs=6,
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 norm_type='gn',
                 norm_groups=32,
                 static_assigner_epoch=4,
                 use_align_head=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 loss_weight={'class': 1.0,
                              'bbox': 1.0,
                              'iou': 2.0}):
        super(PPTOODHead, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.stacked_convs = stacked_convs
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_align_head = use_align_head

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms

        self.stem_conv = nn.LayerList()
        for in_c in self.in_channels:
            self.stem_conv.append(
                ConvNormLayer(
                    in_c,
                    self.feat_channels,
                    filter_size=1,
                    stride=1,
                    norm_type=norm_type,
                    norm_groups=norm_groups))

        self.inter_convs = nn.LayerList()
        for i in range(self.stacked_convs):
            self.inter_convs.append(
                ConvNormLayer(
                    self.feat_channels,
                    self.feat_channels,
                    filter_size=3,
                    stride=1,
                    norm_type=norm_type,
                    norm_groups=norm_groups))

        self.cls_decomp = TaskDecomposition(
            self.feat_channels,
            self.stacked_convs,
            self.stacked_convs * 8,
            norm_type=norm_type,
            norm_groups=norm_groups)
        self.reg_decomp = TaskDecomposition(
            self.feat_channels,
            self.stacked_convs,
            self.stacked_convs * 8,
            norm_type=norm_type,
            norm_groups=norm_groups)

        self.tood_cls = nn.Conv2D(
            self.feat_channels, self.num_classes, 3, padding=1)
        self.tood_reg = nn.Conv2D(self.feat_channels, 4, 3, padding=1)

        if self.use_align_head:
            self.cls_prob_conv1 = nn.Conv2D(self.feat_channels *
                                            self.stacked_convs,
                                            self.feat_channels // 4, 1)
            self.cls_prob_conv2 = nn.Conv2D(
                self.feat_channels // 4, 1, 3, padding=1)
            self.reg_offset_conv1 = nn.Conv2D(self.feat_channels *
                                              self.stacked_convs,
                                              self.feat_channels // 4, 1)
            self.reg_offset_conv2 = nn.Conv2D(
                self.feat_channels // 4, 4 * 2, 3, padding=1)

        self.scales_regs = nn.LayerList([ScaleReg() for _ in self.fpn_strides])

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        normal_(self.tood_cls.weight, std=0.01)
        constant_(self.tood_cls.bias, bias_cls)
        normal_(self.tood_reg.weight, std=0.01)

        if self.use_align_head:
            normal_(self.cls_prob_conv1.weight, std=0.01)
            normal_(self.cls_prob_conv2.weight, std=0.01)
            constant_(self.cls_prob_conv2.bias, bias_cls)
            normal_(self.reg_offset_conv1.weight, std=0.001)
            constant_(self.reg_offset_conv2.weight)
            constant_(self.reg_offset_conv2.bias)

    def _deform_sampling(self, feat, offset):
        """ Sampling the feature according to offset.
        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for for feature sampliing
        """
        # it is an equivalent implementation of bilinear interpolation
        # you can also use F.grid_sample instead
        c = feat.shape[1]
        weight = paddle.ones([c, 1, 1, 1])
        y = deform_conv2d(feat, offset, weight, deformable_groups=c, groups=c)
        return y

    def _reg_grid_sample(self, feat, offset, anchor_points):
        b, _, h, w = get_static_shape(feat)
        feat = paddle.reshape(feat, [-1, 1, h, w])
        offset = paddle.reshape(offset, [-1, 2, h, w]).transpose([0, 2, 3, 1])
        grid_shape = paddle.concat([w, h]).astype('float32')
        grid = (offset + anchor_points) / grid_shape
        grid = 2 * grid.clip(0., 1.) - 1
        feat = F.grid_sample(feat, grid)
        feat = paddle.reshape(feat, [b, -1, h, w])
        return feat

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        anchors, num_anchors_list, stride_tensor_list = generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale,
            self.grid_cell_offset)

        cls_score_list, bbox_pred_list = [], []
        for feat, conv_reduction, scale_reg, anchor, stride in zip(
                feats, self.stem_conv, self.scales_regs, anchors,
                self.fpn_strides):
            feat = F.relu(conv_reduction(feat))
            b, _, h, w = get_static_shape(feat)
            inter_feats = []
            for inter_conv in self.inter_convs:
                feat = F.relu(inter_conv(feat))
                inter_feats.append(feat)
            feat = paddle.concat(inter_feats, axis=1)

            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)

            # cls prediction and alignment
            cls_logits = self.tood_cls(cls_feat)
            if self.use_align_head:
                cls_prob = F.relu(self.cls_prob_conv1(feat))
                cls_prob = F.sigmoid(self.cls_prob_conv2(cls_prob))
                cls_score = (F.sigmoid(cls_logits) * cls_prob).sqrt()
            else:
                cls_score = F.sigmoid(cls_logits)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))

            # reg prediction and alignment
            reg_dist = scale_reg(self.tood_reg(reg_feat).exp())
            reg_dist = reg_dist.flatten(2).transpose([0, 2, 1])
            anchor_centers = bbox_center(anchor).unsqueeze(0) / stride
            reg_bbox = batch_distance2bbox(anchor_centers, reg_dist)
            if self.use_align_head:
                reg_offset = F.relu(self.reg_offset_conv1(feat))
                reg_offset = self.reg_offset_conv2(reg_offset)
                reg_bbox = reg_bbox.transpose([0, 2, 1]).reshape([b, 4, h, w])
                anchor_centers = anchor_centers.reshape([1, h, w, 2])
                bbox_pred = self._reg_grid_sample(reg_bbox, reg_offset,
                                                  anchor_centers)
                bbox_pred = bbox_pred.flatten(2).transpose([0, 2, 1])
            else:
                bbox_pred = reg_bbox

            if not self.training:
                bbox_pred *= stride
            bbox_pred_list.append(bbox_pred)
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        bbox_pred_list = paddle.concat(bbox_pred_list, axis=1)
        anchors = paddle.concat(anchors)
        anchors.stop_gradient = True
        stride_tensor_list = paddle.concat(stride_tensor_list)
        stride_tensor_list.stop_gradient = True

        if self.training:
            return self.get_loss([
                cls_score_list, bbox_pred_list, anchors, num_anchors_list,
                stride_tensor_list
            ], targets)
        else:
            return cls_score_list, bbox_pred_list, anchors, num_anchors_list, stride_tensor_list

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_bboxes, anchors, num_anchors_list, stride_tensor_list = head_outs
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
                gt_scores=gt_scores)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor_list,
                bbox_center(anchors),
                num_anchors_list,
                stride_tensor_list,
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                gt_scores=gt_scores)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor_list
        # cls loss
        loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha=alpha_l)

        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        assigned_scores_sum = assigned_scores.sum()
        if core.is_compiled_with_dist(
        ) and parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
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
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])

        loss_cls /= assigned_scores_sum
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1
        }
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_bboxes, _, _, _ = head_outs
        pred_scores = pred_scores.transpose([0, 2, 1])

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
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num


class ESEAttn(nn.Layer):
    def __init__(self, feat_channels):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2D(feat_channels, feat_channels, 1)
        self.conv = ConvNormLayer(feat_channels, feat_channels, 1, 1)

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        out = self.conv(feat * weight)
        return mish(out)


@register
class PPSimTHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 use_align_head=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 loss_weight={'class': 1.0,
                              'iou': 2.0}):
        super(PPSimTHead, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.use_align_head = use_align_head

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
        self.simT_cls = nn.LayerList()
        self.simT_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.simT_cls.append(
                nn.Conv2D(
                    in_c, self.num_classes, 3, padding=1))
            self.simT_reg.append(nn.Conv2D(in_c, 4, 3, padding=1))
        # align head
        if self.use_align_head:
            self.cls_align = nn.LayerList()
            self.reg_align = nn.LayerList()
            for in_c in self.in_channels:
                self.cls_align.append(nn.Conv2D(in_c, 1, 3, padding=1))
                self.reg_align.append(nn.Conv2D(in_c, 4 * 2, 3, padding=1))

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_reg = math.log(self.grid_cell_scale / 2)
        for cls_, reg_ in zip(self.simT_cls, self.simT_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, bias_reg)

        if self.use_align_head:
            for cls_, reg_ in zip(self.cls_align, self.reg_align):
                constant_(cls_.weight)
                constant_(cls_.bias, bias_cls)
                constant_(reg_.weight)

    def _reg_grid_sample(self, feat, offset, anchor_points):
        b, _, h, w = get_static_shape(feat)
        feat = paddle.reshape(feat, [-1, 1, h, w])
        offset = paddle.reshape(offset, [-1, 2, h, w]).transpose([0, 2, 3, 1])
        grid_shape = paddle.concat([w, h]).astype('float32')
        grid = (offset + anchor_points) / grid_shape
        grid = 2 * grid.clip(0., 1.) - 1
        feat = F.grid_sample(feat, grid)
        feat = paddle.reshape(feat, [b, -1, h, w])
        return feat

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale,
            self.grid_cell_offset)
        anchor_centers = (anchor_points / stride_tensor).split(num_anchors_list)

        cls_score_list, bbox_pred_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = get_static_shape(feat)
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.simT_cls[i](self.stem_cls[i](feat, avg_feat))
            reg_dist = self.simT_reg[i](self.stem_reg[i](feat, avg_feat)).exp()

            # cls prediction and alignment
            if self.use_align_head:
                cls_prob = F.sigmoid(self.cls_align[i](feat))
                cls_score = (F.sigmoid(cls_logit) * cls_prob + eps).sqrt()
            else:
                cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))

            # reg prediction and alignment
            reg_dist = reg_dist.flatten(2).transpose([0, 2, 1])
            reg_bbox = batch_distance2bbox(anchor_centers[i], reg_dist)
            if self.use_align_head:
                anchor_center = anchor_centers[i].reshape([1, h, w, 2])
                reg_bbox = reg_bbox.transpose([0, 2, 1]).reshape([b, 4, h, w])
                reg_offset = self.reg_align[i](feat)
                bbox_pred = self._reg_grid_sample(reg_bbox, reg_offset,
                                                  anchor_center)
                bbox_pred = bbox_pred.flatten(2).transpose([0, 2, 1])
            else:
                bbox_pred = reg_bbox
            bbox_pred_list.append(bbox_pred)
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        bbox_pred_list = paddle.concat(bbox_pred_list, axis=1)

        if self.training:
            return self.get_loss([
                cls_score_list, bbox_pred_list, anchors, anchor_points,
                num_anchors_list, stride_tensor
            ], targets)
        else:
            return cls_score_list, bbox_pred_list, stride_tensor

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

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_bboxes, anchors, anchor_points,\
        num_anchors_list, stride_tensor = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores =\
                self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
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

        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        assigned_scores_sum = assigned_scores.sum()
        if core.is_compiled_with_dist(
        ) and parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
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
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])

        loss_cls /= assigned_scores_sum
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1
        }
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_bboxes, stride_tensor = head_outs
        pred_scores = pred_scores.transpose([0, 2, 1])
        pred_bboxes *= stride_tensor
        # clip bbox to origin
        img_shape = img_shape.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes = paddle.where(pred_bboxes < img_shape, pred_bboxes,
                                   img_shape)
        pred_bboxes = paddle.where(pred_bboxes > 0, pred_bboxes,
                                   paddle.zeros_like(pred_bboxes))
        # scale bbox to origin
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num


@register
class PPSimYHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 static_assigner_epoch=4,
                 use_align_head=True,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 loss_weight={'obj': 1.0,
                              'class': 1.0,
                              'iou': 2.0}):
        super(PPSimYHead, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_align_head = use_align_head
        self.use_varifocal_loss = use_varifocal_loss

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

        # head
        self.simT_cls = nn.LayerList()
        self.simT_reg = nn.LayerList()
        self.simT_obj = nn.LayerList()
        for in_c in self.in_channels:
            self.simT_cls.append(nn.Conv2D(in_c, self.num_classes, 1))
            self.simT_reg.append(nn.Conv2D(in_c, 4, 1))
            self.simT_obj.append(nn.Conv2D(in_c, 1, 1))

        if self.use_align_head:
            self.reg_align = nn.LayerList()
            for in_c in self.in_channels:
                self.reg_align.append(nn.Conv2D(in_c, 4 * 2, 3, padding=1))

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_reg = math.log(self.grid_cell_scale / 2)
        for cls_, reg_, obj_ in zip(self.simT_cls, self.simT_reg,
                                    self.simT_obj):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, bias_reg)
            constant_(obj_.weight)
            constant_(obj_.bias, bias_cls)

        if self.use_align_head:
            for reg_ in self.reg_align:
                constant_(reg_.weight)

    def _reg_grid_sample(self, feat, offset, anchor_points):
        b, _, h, w = get_static_shape(feat)
        feat = paddle.reshape(feat, [-1, 1, h, w])
        offset = paddle.reshape(offset, [-1, 2, h, w]).transpose([0, 2, 3, 1])
        grid_shape = paddle.concat([w, h]).astype('float32')
        grid = (offset + anchor_points) / grid_shape
        grid = 2 * grid.clip(0., 1.) - 1
        feat = F.grid_sample(feat, grid)
        feat = paddle.reshape(feat, [b, -1, h, w])
        return feat

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        anchors, num_anchors_list, stride_tensor_list = generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale,
            self.grid_cell_offset)

        obj_logit_list, cls_logit_list = [], []
        bbox_pred_list = []
        for i, (
                feat, stem_cls, stem_reg, simT_cls, simT_reg, simT_obj, anchor,
                stride
        ) in enumerate(
                zip(feats, self.stem_cls, self.stem_reg, self.simT_cls,
                    self.simT_reg, self.simT_obj, anchors, self.fpn_strides)):
            b, _, h, w = get_static_shape(feat)
            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = simT_cls(stem_cls(feat, avg_feat))
            reg_feat = stem_reg(feat, avg_feat)
            reg_dist = simT_reg(reg_feat).exp()
            obj_logit = simT_obj(feat)
            # obj + cls
            obj_logit_list.append(obj_logit.flatten(2).transpose([0, 2, 1]))
            cls_logit_list.append(cls_logit.flatten(2).transpose([0, 2, 1]))

            # reg prediction and alignment
            reg_dist = reg_dist.flatten(2).transpose([0, 2, 1])
            anchor_centers = bbox_center(anchor).unsqueeze(0) / stride
            reg_bbox = batch_distance2bbox(anchor_centers, reg_dist)
            if self.use_align_head:
                reg_offset = self.reg_align[i](feat)
                reg_bbox = reg_bbox.transpose([0, 2, 1]).reshape([b, 4, h, w])
                anchor_centers = anchor_centers.reshape([1, h, w, 2])
                bbox_pred = self._reg_grid_sample(reg_bbox, reg_offset,
                                                  anchor_centers)
                bbox_pred = bbox_pred.flatten(2).transpose([0, 2, 1])
            else:
                bbox_pred = reg_bbox

            if not self.training:
                bbox_pred *= stride
            bbox_pred_list.append(bbox_pred)
        obj_logit_list = paddle.concat(obj_logit_list, axis=1)
        bbox_pred_list = paddle.concat(bbox_pred_list, axis=1)
        cls_logit_list = paddle.concat(cls_logit_list, axis=1)

        anchors = paddle.concat(anchors)
        anchors.stop_gradient = True
        stride_tensor_list = paddle.concat(stride_tensor_list)
        stride_tensor_list.stop_gradient = True

        if self.training:
            return self.get_loss([
                obj_logit_list, bbox_pred_list, cls_logit_list, anchors,
                num_anchors_list, stride_tensor_list
            ], targets)
        else:
            return obj_logit_list, bbox_pred_list, cls_logit_list

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
        obj_logits, pred_bboxes, cls_logits,\
        anchors, num_anchors_list, stride_tensor_list = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                gt_scores=gt_scores,
                pred_bboxes=pred_bboxes.detach() * stride_tensor_list)
            alpha_l = 0.25
        else:
            pred_scores = F.sigmoid(obj_logits) * F.sigmoid(cls_logits)
            pred_scores = (pred_scores + eps).sqrt()
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor_list,
                bbox_center(anchors),
                num_anchors_list,
                stride_tensor_list,
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                gt_scores=gt_scores)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor_list
        # obj loss
        one_hot_label = F.one_hot(assigned_labels, self.num_classes)
        if self.use_varifocal_loss:
            loss_obj = self._varifocal_loss(
                obj_logits,
                assigned_scores.sum(-1, keepdim=True),
                one_hot_label.sum(-1, keepdim=True),
                alpha=0.75 * self.num_classes)
        else:
            loss_obj = self._focal_loss(
                obj_logits,
                assigned_scores.sum(-1, keepdim=True),
                alpha=alpha_l)

        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        assigned_scores_sum = assigned_scores.sum()
        if core.is_compiled_with_dist(
        ) and parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        # pos/neg loss
        if num_pos > 0:
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            # multi-cls
            cls_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.num_classes])
            pred_cls_pos = paddle.masked_select(
                cls_logits, cls_mask).reshape([-1, self.num_classes])
            assigned_cls_pos = paddle.masked_select(
                one_hot_label, cls_mask).reshape([-1, self.num_classes])
            loss_cls = F.binary_cross_entropy_with_logits(
                pred_cls_pos,
                assigned_cls_pos,
                weight=bbox_weight,
                reduction='sum')
            loss_cls /= assigned_scores_sum
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum
        else:
            loss_cls = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_l1 = paddle.zeros([1])

        loss_obj /= assigned_scores_sum
        loss = self.loss_weight['obj'] * loss_obj + \
               self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou
        out_dict = {
            'loss': loss,
            'loss_obj': loss_obj,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1
        }
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_obj, pred_bboxes, pred_cls = head_outs
        pred_scores = F.sigmoid(pred_obj) * F.sigmoid(pred_cls)
        pred_scores = (pred_scores + eps).sqrt()
        pred_scores = pred_scores.transpose([0, 2, 1])

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
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num


# @register
# class PPRefineTHead(nn.Layer):
#     __shared__ = ['num_classes']
#     __inject__ = ['static_assigner', 'assigner', 'nms']
#
#     def __init__(self,
#                  in_channels=[1024, 512, 256],
#                  num_classes=80,
#                  fpn_strides=(32, 16, 8),
#                  grid_cell_scale=5.0,
#                  grid_cell_offset=0.5,
#                  static_assigner_epoch=4,
#                  use_varifocal_loss=True,
#                  static_assigner='ATSSAssigner',
#                  assigner='TaskAlignedAssigner',
#                  nms='MultiClassNMS',
#                  eval_input_size=[640, 640],
#                  loss_weight={
#                      'class': 1.0,
#                      'iou': 1.0,
#                      'iou_r': 2.0,
#                  }):
#         super(PPRefineTHead, self).__init__()
#         assert len(in_channels) > 0, "in_channels length should > 0"
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.fpn_strides = fpn_strides
#         self.grid_cell_scale = grid_cell_scale
#         self.grid_cell_offset = grid_cell_offset
#         self.iou_loss = GIoULoss()
#         self.loss_weight = loss_weight
#         self.use_varifocal_loss = use_varifocal_loss
#         self.eval_input_size = eval_input_size
#
#         self.static_assigner_epoch = static_assigner_epoch
#         self.static_assigner = static_assigner
#         self.assigner = assigner
#         self.nms = nms
#
#         self.stem_cls = nn.LayerList()
#         self.stem_reg = nn.LayerList()
#         for in_c in self.in_channels:
#             self.stem_cls.append(ESEAttn(in_c))
#             self.stem_reg.append(ESEAttn(in_c))
#
#         self.simT_cls = nn.LayerList()
#         self.simT_reg = nn.LayerList()
#         for in_c in self.in_channels:
#             self.simT_cls.append(
#                 nn.Conv2D(
#                     in_c, self.num_classes, 3, padding=1))
#             self.simT_reg.append(nn.Conv2D(in_c, 4, 3, padding=1))
#
#         self.cls_align = nn.LayerList()
#         self.reg_align = nn.LayerList()
#         for in_c in self.in_channels:
#             self.cls_align.append(nn.Conv2D(in_c, 1, 3, padding=1))
#             self.reg_align.append(nn.Conv2D(in_c, 4, 3, padding=1))
#
#         self._init_weights()
#
#     @classmethod
#     def from_config(cls, cfg, input_shape):
#         return {'in_channels': [i.channels for i in input_shape], }
#
#     def _init_weights(self):
#         bias_cls = bias_init_with_prob(0.01)
#         bias_reg = self.grid_cell_scale / 2
#         for cls_, reg_ in zip(self.simT_cls, self.simT_reg):
#             constant_(cls_.weight)
#             constant_(cls_.bias, bias_cls)
#             constant_(reg_.weight)
#             constant_(reg_.bias, bias_reg)
#
#         for cls_, reg_ in zip(self.cls_align, self.reg_align):
#             constant_(cls_.weight)
#             constant_(cls_.bias, bias_cls)
#             constant_(reg_.weight)
#
#         anchor_points, stride_tensor = self._generate_anchors()
#         self.register_buffer('anchor_points', anchor_points)
#         self.register_buffer('stride_tensor', stride_tensor)
#
#     def forward_train(self, feats, targets):
#         anchors, anchor_points, num_anchors_list, stride_tensor = \
#             generate_anchors_for_grid_cell(
#                 feats, self.fpn_strides, self.grid_cell_scale,
#                 self.grid_cell_offset)
#
#         cls_score_list = []
#         reg_dist_list, reg_refine_list = [], []
#         for i, feat in enumerate(feats):
#             # task decomposition
#             avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
#             cls_logit = self.simT_cls[i](self.stem_cls[i](feat, avg_feat))
#             reg_dist = self.simT_reg[i](self.stem_reg[i](feat, avg_feat))
#             reg_dist = F.relu(reg_dist)
#
#             # cls prediction and alignment
#             cls_prob = F.sigmoid(self.cls_align[i](feat))
#             cls_score = (F.sigmoid(cls_logit) * cls_prob + eps).sqrt()
#             cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
#
#             # reg prediction and alignment
#             reg_refine = self.reg_align[i](feat).exp() * reg_dist
#             reg_dist = reg_dist.flatten(2).transpose([0, 2, 1])
#             reg_dist_list.append(reg_dist)
#             reg_refine = reg_refine.flatten(2).transpose([0, 2, 1])
#             reg_refine_list.append(reg_refine)
#
#         cls_score_list = paddle.concat(cls_score_list, axis=1)
#         reg_dist_list = paddle.concat(reg_dist_list, axis=1)
#         reg_refine_list = paddle.concat(reg_refine_list, axis=1)
#
#         return self.get_loss([
#             cls_score_list, reg_dist_list, reg_refine_list, anchors,
#             anchor_points, num_anchors_list, stride_tensor
#         ], targets)
#
#     def _generate_anchors(self):
#         # just use in eval time
#         anchor_points = []
#         stride_tensor = []
#         for i, stride in enumerate(self.fpn_strides):
#             h = int(self.eval_input_size[0] / stride)
#             w = int(self.eval_input_size[1] / stride)
#             shift_x = paddle.arange(end=w) + self.grid_cell_offset
#             shift_y = paddle.arange(end=h) + self.grid_cell_offset
#             shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
#             anchor_point = paddle.cast(
#                 paddle.stack(
#                     [shift_x, shift_y], axis=-1), dtype='float32')
#             anchor_points.append(anchor_point.reshape([-1, 2]))
#             stride_tensor.append(
#                 paddle.full(
#                     [h * w, 1], stride, dtype='float32'))
#         anchor_points = paddle.concat(anchor_points)
#         stride_tensor = paddle.concat(stride_tensor)
#         return anchor_points, stride_tensor
#
#     def forward_eval(self, feats):
#         cls_score_list, reg_refine_list = [], []
#         for i, feat in enumerate(feats):
#             b, _, h, w = feat.shape
#             l = h * w
#             # task decomposition
#             avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
#             cls_logit = self.simT_cls[i](self.stem_cls[i](feat, avg_feat))
#             reg_dist = self.simT_reg[i](self.stem_reg[i](feat, avg_feat))
#             reg_dist = F.relu(reg_dist)
#
#             # cls prediction and alignment
#             cls_prob = F.sigmoid(self.cls_align[i](feat))
#             cls_score = (F.sigmoid(cls_logit) * cls_prob).sqrt()
#             cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
#
#             # reg prediction and alignment
#             reg_refine = self.reg_align[i](feat).exp() * reg_dist
#             reg_refine = reg_refine.reshape([b, 4, l])
#             reg_refine_list.append(reg_refine)
#
#         cls_score_list = paddle.concat(cls_score_list, axis=-1)
#         reg_refine_list = paddle.concat(reg_refine_list, axis=-1)
#
#         return cls_score_list, reg_refine_list
#
#     def forward(self, feats, targets=None):
#         assert len(feats) == len(self.fpn_strides), \
#             "The size of feats is not equal to size of fpn_strides"
#
#         if self.training:
#             return self.forward_train(feats, targets)
#         else:
#             return self.forward_eval(feats)
#
#     @staticmethod
#     def _focal_loss(score, label, alpha=0.25, gamma=2.0):
#         weight = (score - label).pow(gamma)
#         if alpha > 0:
#             alpha_t = alpha * label + (1 - alpha) * (1 - label)
#             weight *= alpha_t
#         loss = F.binary_cross_entropy(
#             score, label, weight=weight, reduction='sum')
#         return loss
#
#     @staticmethod
#     def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
#         weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
#         loss = F.binary_cross_entropy(
#             pred_score, gt_score, weight=weight, reduction='sum')
#         return loss
#
#     def _bbox_loss(self, pred_bboxes, assigned_labels, assigned_bboxes,
#                    assigned_scores, assigned_scores_sum):
#         # select positive samples mask
#         mask_positive = (assigned_labels != self.num_classes)
#         num_pos = mask_positive.sum()
#         # pos/neg loss
#         if num_pos > 0:
#             # l1 + iou
#             bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
#             pred_bboxes_pos = paddle.masked_select(pred_bboxes,
#                                                    bbox_mask).reshape([-1, 4])
#             assigned_bboxes_pos = paddle.masked_select(
#                 assigned_bboxes, bbox_mask).reshape([-1, 4])
#             bbox_weight = paddle.masked_select(
#                 assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
#
#             loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)
#
#             loss_iou = self.iou_loss(pred_bboxes_pos,
#                                      assigned_bboxes_pos) * bbox_weight
#             loss_iou = loss_iou.sum() / assigned_scores_sum
#         else:
#             loss_l1 = paddle.zeros([1])
#             loss_iou = paddle.zeros([1])
#         return loss_l1, loss_iou
#
#     def get_loss(self, head_outs, gt_meta):
#         pred_scores, pred_dist, pred_refine, anchors,\
#         anchor_points, num_anchors_list, stride_tensor = head_outs
#
#         pred_bboxes = batch_distance2bbox(anchor_points / stride_tensor,
#                                           pred_dist)
#         refine_bboxes = batch_distance2bbox(anchor_points / stride_tensor,
#                                             pred_refine)
#         gt_labels = gt_meta['gt_class']
#         gt_bboxes = gt_meta['gt_bbox']
#         pad_gt_mask = gt_meta['pad_gt_mask']
#         gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None
#         # label assignment
#         if gt_meta['epoch_id'] < self.static_assigner_epoch:
#             assigned_labels, assigned_bboxes, assigned_scores = \
#                 self.static_assigner(
#                 anchors,
#                 num_anchors_list,
#                 gt_labels,
#                 gt_bboxes,
#                 pad_gt_mask,
#                 bg_index=self.num_classes,
#                 gt_scores=gt_scores,
#                 pred_bboxes=pred_bboxes.detach() * stride_tensor)
#             assigned_labels_r, assigned_bboxes_r, assigned_scores_r = \
#                 self.static_assigner(
#                     anchors,
#                     num_anchors_list,
#                     gt_labels,
#                     gt_bboxes,
#                     pad_gt_mask,
#                     bg_index=self.num_classes,
#                     gt_scores=gt_scores,
#                     pred_bboxes=refine_bboxes.detach() * stride_tensor)
#             alpha_l = 0.25
#         else:
#             assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
#                 pred_scores.detach(),
#                 pred_bboxes.detach() * stride_tensor,
#                 anchor_points,
#                 num_anchors_list,
#                 stride_tensor,
#                 gt_labels,
#                 gt_bboxes,
#                 pad_gt_mask,
#                 bg_index=self.num_classes,
#                 gt_scores=gt_scores)
#             assigned_labels_r, assigned_bboxes_r, assigned_scores_r = \
#                 self.assigner(
#                 pred_scores.detach(),
#                 refine_bboxes.detach() * stride_tensor,
#                 anchor_points,
#                 num_anchors_list,
#                 stride_tensor,
#                 gt_labels,
#                 gt_bboxes,
#                 pad_gt_mask,
#                 bg_index=self.num_classes,
#                 gt_scores=gt_scores)
#             alpha_l = -1
#         # rescale bbox
#         assigned_bboxes /= stride_tensor
#         assigned_bboxes_r /= stride_tensor
#         # cls loss
#         if self.use_varifocal_loss:
#             one_hot_label = F.one_hot(assigned_labels_r, self.num_classes)
#             loss_cls = self._varifocal_loss(pred_scores, assigned_scores_r,
#                                             one_hot_label)
#         else:
#             loss_cls = self._focal_loss(
#                 pred_scores, assigned_scores_r, alpha=alpha_l)
#
#         assigned_scores_sum = assigned_scores.sum()
#         assigned_scores_r_sum = assigned_scores_r.sum()
#         if core.is_compiled_with_dist(
#         ) and parallel_helper._is_parallel_ctx_initialized():
#             paddle.distributed.all_reduce(assigned_scores_sum)
#             assigned_scores_sum = paddle.clip(
#                 assigned_scores_sum / paddle.distributed.get_world_size(),
#                 min=1)
#             paddle.distributed.all_reduce(assigned_scores_r_sum)
#             assigned_scores_r_sum = paddle.clip(
#                 assigned_scores_r_sum / paddle.distributed.get_world_size(),
#                 min=1)
#         loss_cls /= assigned_scores_r_sum
#
#         loss_l1, loss_iou = self._bbox_loss(pred_bboxes, assigned_labels,
#                                             assigned_bboxes, assigned_scores,
#                                             assigned_scores_sum)
#         loss_l1_r, loss_iou_r = self._bbox_loss(
#             refine_bboxes, assigned_labels_r, assigned_bboxes_r,
#             assigned_scores_r, assigned_scores_r_sum)
#         loss = self.loss_weight['class'] * loss_cls + \
#                self.loss_weight['iou'] * loss_iou + \
#                self.loss_weight['iou_r'] * loss_iou_r
#         out_dict = {
#             'loss': loss,
#             'loss_cls': loss_cls,
#             'loss_iou': loss_iou,
#             'loss_l1': loss_l1,
#             'loss_iou_r': loss_iou_r,
#             'loss_l1_r': loss_l1_r,
#         }
#         return out_dict
#
#     def post_process(self, head_outs, img_shape, scale_factor):
#         pred_scores, pred_dist = head_outs
#         pred_bboxes = batch_distance2bbox(self.anchor_points,
#                                           pred_dist.transpose([0, 2, 1]))
#         pred_bboxes *= self.stride_tensor
#         # scale bbox to origin
#         scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
#         pred_bboxes /= scale_factor
#         bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
#         return bbox_pred, bbox_num


@register
class PPVFTHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_input_size=[640, 640],
                 loss_weight={
                     'class': 1.0,
                     'iou': 1.0,
                     'iou_r': 2.0,
                 }):
        super(PPVFTHead, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_input_size = eval_input_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms

        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c))
            self.stem_reg.append(ESEAttn(in_c))

        self.simT_cls = nn.LayerList()
        self.simT_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.simT_cls.append(
                nn.Conv2D(
                    in_c, self.num_classes, 3, padding=1))
            self.simT_reg.append(nn.Conv2D(in_c, 4, 3, padding=1))

        self.cls_align = nn.LayerList()
        self.reg_align = nn.LayerList()
        for in_c in self.in_channels:
            self.cls_align.append(nn.Conv2D(in_c, 1, 3, padding=1))
            self.reg_align.append(nn.Conv2D(in_c, 4, 3, padding=1))

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_reg = self.grid_cell_scale / 2
        for cls_, reg_ in zip(self.simT_cls, self.simT_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, bias_reg)

        for cls_, reg_ in zip(self.cls_align, self.reg_align):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)

        anchor_points, stride_tensor = self._generate_anchors()
        self.register_buffer('anchor_points', anchor_points)
        self.register_buffer('stride_tensor', stride_tensor)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list = []
        reg_dist_list, reg_refine_list = [], []
        for i, feat in enumerate(feats):
            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.simT_cls[i](self.stem_cls[i](feat, avg_feat))
            reg_dist = self.simT_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = F.relu(reg_dist)

            # cls prediction and alignment
            cls_prob = F.sigmoid(self.cls_align[i](feat))
            cls_score = (F.sigmoid(cls_logit) * cls_prob + eps).sqrt()
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))

            # reg prediction and alignment
            reg_refine = self.reg_align[i](feat).exp() * reg_dist
            reg_dist = reg_dist.flatten(2).transpose([0, 2, 1])
            reg_dist_list.append(reg_dist)
            reg_refine = reg_refine.flatten(2).transpose([0, 2, 1])
            reg_refine_list.append(reg_refine)

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        reg_refine_list = paddle.concat(reg_refine_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_dist_list, reg_refine_list, anchors,
            anchor_points, num_anchors_list, stride_tensor
        ], targets)

    def _generate_anchors(self):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
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
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        cls_score_list, reg_refine_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.simT_cls[i](self.stem_cls[i](feat, avg_feat))
            reg_dist = self.simT_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = F.relu(reg_dist)

            # cls prediction and alignment
            cls_prob = F.sigmoid(self.cls_align[i](feat))
            cls_score = (F.sigmoid(cls_logit) * cls_prob).sqrt()
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))

            # reg prediction and alignment
            reg_refine = self.reg_align[i](feat).exp() * reg_dist
            reg_refine = reg_refine.reshape([b, 4, l])
            reg_refine_list.append(reg_refine)

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_refine_list = paddle.concat(reg_refine_list, axis=-1)

        return cls_score_list, reg_refine_list

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

    def _bbox_loss(self, pred_bboxes, assigned_labels, assigned_bboxes,
                   assigned_scores, assigned_scores_sum):
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
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
        return loss_l1, loss_iou

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_dist, pred_refine, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        pred_bboxes = batch_distance2bbox(anchor_points / stride_tensor,
                                          pred_dist)
        refine_bboxes = batch_distance2bbox(anchor_points / stride_tensor,
                                            pred_refine)
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels_r, assigned_bboxes_r, assigned_scores_r = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    gt_scores=gt_scores,
                    pred_bboxes=refine_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels_r, assigned_bboxes_r, assigned_scores_r = \
                self.assigner(
                pred_scores.detach(),
                refine_bboxes.detach() * stride_tensor,
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
        assigned_bboxes_r /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels_r, self.num_classes)
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores_r,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(
                pred_scores, assigned_scores_r, alpha=alpha_l)

        assigned_scores_r_sum = assigned_scores_r.sum()
        if core.is_compiled_with_dist(
        ) and parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.all_reduce(assigned_scores_r_sum)
            assigned_scores_r_sum = paddle.clip(
                assigned_scores_r_sum / paddle.distributed.get_world_size(),
                min=1)
        loss_cls /= assigned_scores_r_sum

        loss_l1, loss_iou = self._bbox_loss(
            pred_bboxes, assigned_labels_r, assigned_bboxes_r,
            assigned_scores_r, assigned_scores_r_sum)
        loss_l1_r, loss_iou_r = self._bbox_loss(
            refine_bboxes, assigned_labels_r, assigned_bboxes_r,
            assigned_scores_r, assigned_scores_r_sum)
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['iou_r'] * loss_iou_r
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1,
            'loss_iou_r': loss_iou_r,
            'loss_l1_r': loss_l1_r,
        }
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_dist = head_outs
        pred_bboxes = batch_distance2bbox(self.anchor_points,
                                          pred_dist.transpose([0, 2, 1]))
        pred_bboxes *= self.stride_tensor
        # scale bbox to origin
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num


@register
class PPVFT3Head(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 loss_weight={
                     'class': 1.0,
                     'iou': 1.0,
                     'iou_r': 1.5,
                     'iou_r3': 2.0,
                 }):
        super(PPVFT3Head, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms

        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c))
            self.stem_reg.append(ESEAttn(in_c))

        self.simT_cls = nn.LayerList()
        self.simT_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.simT_cls.append(
                nn.Conv2D(
                    in_c, self.num_classes, 3, padding=1))
            self.simT_reg.append(nn.Conv2D(in_c, 4, 3, padding=1))

        self.cls_align = nn.LayerList()
        self.reg_align = nn.LayerList()
        self.reg_refine3 = nn.LayerList()
        for in_c in self.in_channels:
            self.cls_align.append(nn.Conv2D(in_c, 1, 3, padding=1))
            self.reg_align.append(nn.Conv2D(in_c, 4, 3, padding=1))
            self.reg_refine3.append(nn.Conv2D(in_c, 4, 3, padding=1))

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_reg = self.grid_cell_scale / 2
        for cls_, reg_ in zip(self.simT_cls, self.simT_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, bias_reg)

        for cls_, reg_, reg_3 in zip(self.cls_align, self.reg_align,
                                     self.reg_refine3):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_3.weight)

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list = []
        reg_dist_list, reg_refine_list = [], []
        reg_refine3_list = []
        for i, feat in enumerate(feats):
            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.simT_cls[i](self.stem_cls[i](feat, avg_feat))
            reg_dist = F.relu(self.simT_reg[i](self.stem_reg[i](feat,
                                                                avg_feat)))

            # cls prediction and alignment
            cls_prob = F.sigmoid(self.cls_align[i](feat))
            cls_score = (F.sigmoid(cls_logit) * cls_prob + eps).sqrt()
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))

            # reg prediction and alignment
            reg_refine = self.reg_align[i](feat).exp() * reg_dist
            reg_refine3 = self.reg_refine3[i](feat).exp() * reg_refine
            reg_dist = reg_dist.flatten(2).transpose([0, 2, 1])
            reg_dist_list.append(reg_dist)
            reg_refine = reg_refine.flatten(2).transpose([0, 2, 1])
            reg_refine_list.append(reg_refine)
            reg_refine3 = reg_refine3.flatten(2).transpose([0, 2, 1])
            reg_refine3_list.append(reg_refine3)

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        reg_refine_list = paddle.concat(reg_refine_list, axis=1)
        reg_refine3_list = paddle.concat(reg_refine3_list, axis=1)

        if self.training:
            return self.get_loss([
                cls_score_list, reg_dist_list, reg_refine_list,
                reg_refine3_list, anchors, anchor_points, num_anchors_list,
                stride_tensor
            ], targets)
        else:
            return cls_score_list, reg_refine3_list, anchor_points, stride_tensor

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

    def _bbox_loss(self, pred_bboxes, assigned_labels, assigned_bboxes,
                   assigned_scores, assigned_scores_sum):
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
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
        return loss_l1, loss_iou

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_dist, pred_refine, pred_refine3, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        pred_bboxes = batch_distance2bbox(anchor_points / stride_tensor,
                                          pred_dist)
        refine_bboxes = batch_distance2bbox(anchor_points / stride_tensor,
                                            pred_refine)
        refine_bboxes3 = batch_distance2bbox(anchor_points / stride_tensor,
                                             pred_refine3)
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
            assigned_labels_r, assigned_bboxes_r, assigned_scores_r = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    gt_scores=gt_scores,
                    pred_bboxes=refine_bboxes.detach() * stride_tensor)
            assigned_labels_r3, assigned_bboxes_r3, assigned_scores_r3 = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    gt_scores=gt_scores,
                    pred_bboxes=refine_bboxes3.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
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
            assigned_labels_r, assigned_bboxes_r, assigned_scores_r = \
                self.assigner(
                pred_scores.detach(),
                refine_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                stride_tensor,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                gt_scores=gt_scores)
            assigned_labels_r3, assigned_bboxes_r3, assigned_scores_r3 = \
                self.assigner(
                    pred_scores.detach(),
                    refine_bboxes3.detach() * stride_tensor,
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
        assigned_bboxes_r /= stride_tensor
        assigned_bboxes_r3 /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels_r3, self.num_classes)
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores_r3,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(
                pred_scores, assigned_scores_r, alpha=alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        assigned_scores_r_sum = assigned_scores_r.sum()
        assigned_scores_r3_sum = assigned_scores_r3.sum()
        if core.is_compiled_with_dist(
        ) and parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
            paddle.distributed.all_reduce(assigned_scores_r_sum)
            assigned_scores_r_sum = paddle.clip(
                assigned_scores_r_sum / paddle.distributed.get_world_size(),
                min=1)
            paddle.distributed.all_reduce(assigned_scores_r3_sum)
            assigned_scores_r3_sum = paddle.clip(
                assigned_scores_r3_sum / paddle.distributed.get_world_size(),
                min=1)
        loss_cls /= assigned_scores_r3_sum

        loss_l1, loss_iou = self._bbox_loss(pred_bboxes, assigned_labels,
                                            assigned_bboxes, assigned_scores,
                                            assigned_scores_sum)
        loss_l1_r, loss_iou_r = self._bbox_loss(
            refine_bboxes, assigned_labels_r, assigned_bboxes_r,
            assigned_scores_r, assigned_scores_r_sum)
        loss_l1_r3, loss_iou_r3 = self._bbox_loss(
            refine_bboxes3, assigned_labels_r3, assigned_bboxes_r3,
            assigned_scores_r3, assigned_scores_r3_sum)
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['iou_r'] * loss_iou_r + \
               self.loss_weight['iou_r3'] * loss_iou_r3
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1,
            'loss_iou_r': loss_iou_r,
            'loss_l1_r': loss_l1_r,
            'loss_iou_r3': loss_iou_r3,
            'loss_l1_r3': loss_l1_r3,
        }
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_scores = pred_scores.transpose([0, 2, 1])
        pred_bboxes = batch_distance2bbox(anchor_points,
                                          pred_dist * stride_tensor)

        # scale bbox to origin
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num

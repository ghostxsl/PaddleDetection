import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register

from ..bbox_utils import batch_distance2bbox, batch_bbox_overlaps
from ..losses import GIoULoss
from ..initializer import bias_init_with_prob, constant_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.ops import paddle_distributed_is_initialized


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
class PPYOLOCHead(nn.Layer):
    __shared__ = ['num_classes', 'exclude_nms']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 static_assigner_epoch=4,
                 use_varifocal_loss=False,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_input_size=[],
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.0,
                 },
                 exclude_nms=False):
        super(PPYOLOCHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
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
        self.exclude_nms = exclude_nms
        # pred head
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        for in_channel in self.in_channels:
            self.cls_convs.append(
                nn.Conv2D(
                    in_channels=in_channel,
                    out_channels=self.num_classes,
                    kernel_size=3,
                    padding=1))
            self.reg_convs.append(
                nn.Conv2D(
                    in_channels=in_channel,
                    out_channels=4,
                    kernel_size=3,
                    padding=1))

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_reg = self.grid_cell_scale / 2
        for conv in self.cls_convs:
            constant_(conv.weight)
            constant_(conv.bias, bias_cls)
        for conv in self.reg_convs:
            constant_(conv.weight)
            constant_(conv.bias, bias_reg)

        if self.eval_input_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.register_buffer('anchor_points', anchor_points)
            self.register_buffer('stride_tensor', stride_tensor)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        pred_scores, pred_dist = [], []
        for feat, conv_cls, conv_reg in zip(feats, self.cls_convs,
                                            self.reg_convs):
            cls_logit = conv_cls(feat)
            reg_dist = conv_reg(feat)
            pred_scores.append(cls_logit.flatten(2).transpose([0, 2, 1]))
            pred_dist.append(reg_dist.flatten(2).transpose([0, 2, 1]))
        pred_scores = F.sigmoid(paddle.concat(pred_scores, 1))
        pred_dist = F.relu(paddle.concat(pred_dist, 1))

        return self.get_loss([
            pred_scores, pred_dist, anchors, anchor_points, num_anchors_list,
            stride_tensor
        ], targets)

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
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
        if self.eval_input_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)

        pred_scores, pred_dist = [], []
        for feat, conv_cls, conv_reg in zip(feats, self.cls_convs,
                                            self.reg_convs):
            cls_logit = conv_cls(feat)
            reg_dist = conv_reg(feat)
            pred_scores.append(cls_logit.flatten(2))
            pred_dist.append(reg_dist.flatten(2).transpose([0, 2, 1]))
        pred_scores = F.sigmoid(paddle.concat(pred_scores, axis=-1))
        pred_dist = F.relu(paddle.concat(pred_dist, axis=1))

        return pred_scores, pred_dist, anchor_points, stride_tensor

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
        pred_scores, pred_dist, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points / stride_tensor,
                                          pred_dist)
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
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
                bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels, self.num_classes)
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if paddle_distributed_is_initialized():
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou = \
            self._bbox_loss(pred_bboxes, assigned_labels, assigned_bboxes, assigned_scores, assigned_scores_sum)
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1,
        }
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor
        # scale bbox to origin
        scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
        scale_factor = paddle.concat(
            [scale_x, scale_y, scale_x, scale_y], axis=-1).reshape([-1, 1, 4])
        pred_bboxes /= scale_factor
        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_bboxes.sum(), pred_scores.sum()
        else:
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num


@register
class PPYOLOSHead(nn.Layer):
    __shared__ = ['num_classes', 'exclude_nms']
    __inject__ = ['assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 assigner='SimOTAAssigner',
                 nms='MultiClassNMS',
                 eval_input_size=[],
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.0,
                 },
                 exclude_nms=False):
        super(PPYOLOSHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.eval_input_size = eval_input_size

        self.assigner = assigner
        self.nms = nms
        self.exclude_nms = exclude_nms
        # pred head
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        for in_channel in self.in_channels:
            self.cls_convs.append(
                nn.Conv2D(
                    in_channels=in_channel,
                    out_channels=self.num_classes,
                    kernel_size=3,
                    padding=1))
            self.reg_convs.append(
                nn.Conv2D(
                    in_channels=in_channel,
                    out_channels=4,
                    kernel_size=3,
                    padding=1))

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_reg = self.grid_cell_scale / 2
        for conv in self.cls_convs:
            constant_(conv.weight)
            constant_(conv.bias, bias_cls)
        for conv in self.reg_convs:
            constant_(conv.weight)
            constant_(conv.bias, bias_reg)

        if self.eval_input_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.register_buffer('anchor_points', anchor_points)
            self.register_buffer('stride_tensor', stride_tensor)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        pred_scores, pred_dist = [], []
        for feat, conv_cls, conv_reg in zip(feats, self.cls_convs,
                                            self.reg_convs):
            cls_logit = conv_cls(feat)
            reg_dist = conv_reg(feat)
            pred_scores.append(cls_logit.flatten(2).transpose([0, 2, 1]))
            pred_dist.append(reg_dist.flatten(2).transpose([0, 2, 1]))
        pred_scores = F.sigmoid(paddle.concat(pred_scores, 1))
        pred_dist = F.relu(paddle.concat(pred_dist, 1))

        return self.get_loss([
            pred_scores, pred_dist, anchors, anchor_points, num_anchors_list,
            stride_tensor
        ], targets)

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
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
        if self.eval_input_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)

        pred_scores, pred_dist = [], []
        for feat, conv_cls, conv_reg in zip(feats, self.cls_convs,
                                            self.reg_convs):
            cls_logit = conv_cls(feat)
            reg_dist = conv_reg(feat)
            pred_scores.append(cls_logit.flatten(2))
            pred_dist.append(reg_dist.flatten(2).transpose([0, 2, 1]))
        pred_scores = F.sigmoid(paddle.concat(pred_scores, axis=-1))
        pred_dist = F.relu(paddle.concat(pred_dist, axis=1))

        return pred_scores, pred_dist, anchor_points, stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=-1, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss

    def _bbox_loss(self, pred_bboxes, assigned_labels, assigned_bboxes,
                   assigned_scores_sum):
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

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos, assigned_bboxes_pos)
            loss_iou = loss_iou.sum() / assigned_scores_sum
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
        return loss_l1, loss_iou

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_dist, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points / stride_tensor,
                                          pred_dist)
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        # label assignment
        center_and_strides = paddle.concat(
            [anchor_points, stride_tensor, stride_tensor], axis=-1)
        num_pos, assigned_labels, assigned_bboxes = [], [], []
        for pred_score, pred_bbox, gt_box, gt_label in zip(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor, gt_bboxes, gt_labels):
            pos_num, label, _, bbox_target = self.assigner(
                pred_score, center_and_strides, pred_bbox, gt_box, gt_label)
            num_pos.append(pos_num)
            assigned_labels.append(label)
            assigned_bboxes.append(bbox_target)
        num_pos = paddle.to_tensor(sum(num_pos)).clip(min=1)
        assigned_labels = paddle.to_tensor(np.stack(assigned_labels, axis=0))
        assigned_bboxes = paddle.to_tensor(np.stack(assigned_bboxes, axis=0))
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        assigned_scores = batch_bbox_overlaps(
            pred_bboxes, assigned_bboxes, is_aligned=True)
        one_hot_label = F.one_hot(assigned_labels,
                                  self.num_classes + 1)[..., :-1]
        assigned_scores = assigned_scores.unsqueeze(-1) * one_hot_label
        loss_cls = self._focal_loss(pred_scores, assigned_scores) / num_pos

        loss_l1, loss_iou = \
            self._bbox_loss(pred_bboxes, assigned_labels,
                            assigned_bboxes, num_pos)
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1,
        }
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor
        # scale bbox to origin
        scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
        scale_factor = paddle.concat(
            [scale_x, scale_y, scale_x, scale_y], axis=-1).reshape([-1, 1, 4])
        pred_bboxes /= scale_factor
        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_bboxes.sum(), pred_scores.sum()
        else:
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num

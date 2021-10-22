import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register

from paddle.fluid import core
from paddle.fluid.dygraph import parallel_helper
from ..bbox_utils import bbox_center
from ..losses import GIoULoss
from ..ops import iou_similarity
from ..initializer import normal_, constant_, bias_init_with_prob, uniform_
from ..backbones.darknet import ConvBNLayer
from .tood_head import ScaleReg
from paddle import ParamAttr
import math
from ppdet.modeling.layers import ConvNormLayer
from .tood_head import TaskDecomposition
from paddle.vision.ops import deform_conv2d


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
                 static_assigner_epoch=60,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 loss_weight={'class': 1.0,
                              'bbox': 1.0,
                              'iou': 5.0},
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
        for conv in self.cls_convs:
            constant_(conv.weight)
            constant_(conv.bias, bias_cls)
        for conv in self.reg_convs:
            constant_(conv.weight)
            constant_(conv.bias, math.log(self.grid_cell_scale / 2))

    def _generate_anchors(self, feats):
        anchors, num_anchors_list = [], []
        stride_tensor_list = []
        for feat, stride in zip(feats, self.fpn_strides):
            feat_shape = paddle.shape(feat).detach()
            h, w = feat_shape[2:]
            cell_half_size = self.grid_cell_scale * stride * 0.5
            shift_x = (paddle.arange(end=w) + self.grid_cell_offset) * stride
            shift_y = (paddle.arange(end=h) + self.grid_cell_offset) * stride
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor = paddle.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1)
            anchors.append(anchor.reshape([-1, 4]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor_list.append(
                paddle.full([num_anchors_list[-1], 1], stride))
        return anchors, num_anchors_list, stride_tensor_list

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides)
        anchors, num_anchors_list, stride_tensor_list = self._generate_anchors(
            feats)

        pred_logit, pred_dist = [], []
        for feat, conv_cls, conv_reg in zip(feats, self.cls_convs,
                                            self.reg_convs):
            cls_logit = conv_cls(feat)
            reg_dist = conv_reg(feat)
            if self.data_format == 'NCHW':
                cls_logit = cls_logit.transpose([0, 2, 3, 1])
                reg_dist = reg_dist.transpose([0, 2, 3, 1])
            pred_logit.append(cls_logit.flatten(1, 2))
            pred_dist.append(reg_dist.flatten(1, 2))
        pred_logit = paddle.concat(pred_logit, 1)
        pred_dist = paddle.concat(pred_dist, 1)

        anchors = paddle.concat(anchors)
        anchors.stop_gradient = True
        stride_tensor_list = paddle.concat(stride_tensor_list).unsqueeze(0)
        stride_tensor_list.stop_gradient = True

        if self.training:
            return self.get_loss([
                pred_logit, pred_dist, anchors, num_anchors_list,
                stride_tensor_list
            ], targets)
        else:
            return pred_logit, pred_dist, anchors, num_anchors_list, stride_tensor_list

    @staticmethod
    def _batch_distance2bbox(points, distance, max_shapes=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): [B, l, 2]
            distance (Tensor): [B, l, 4]
            max_shapes (tuple): [B, 2], "h w" format, Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, :, 0] - distance[:, :, 0]
        y1 = points[:, :, 1] - distance[:, :, 1]
        x2 = points[:, :, 0] + distance[:, :, 2]
        y2 = points[:, :, 1] + distance[:, :, 3]
        bboxes = paddle.stack([x1, y1, x2, y2], -1)
        if max_shapes is not None:
            out_bboxes = []
            for bbox, max_shape in zip(bboxes, max_shapes):
                bbox[:, 0] = bbox[:, 0].clip(min=0, max=max_shape[1])
                bbox[:, 1] = bbox[:, 1].clip(min=0, max=max_shape[0])
                bbox[:, 2] = bbox[:, 2].clip(min=0, max=max_shape[1])
                bbox[:, 3] = bbox[:, 3].clip(min=0, max=max_shape[0])
                out_bboxes.append(bbox)
            out_bboxes = paddle.stack(out_bboxes)
            return out_bboxes
        return bboxes

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
        pred_logit, pred_dist, anchors, num_anchors_list, stride_tensor_list = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None

        pred_scores = F.sigmoid(pred_logit)
        # distance2bbox
        anchor_centers = bbox_center(anchors)
        pred_bboxes = self._batch_distance2bbox(
            anchor_centers.unsqueeze(0), pred_dist.exp() * stride_tensor_list)

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
                pred_bboxes.detach(),
                anchor_centers,
                num_anchors_list,
                stride_tensor_list.squeeze(0),
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                gt_scores=gt_scores)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor_list
        pred_bboxes /= stride_tensor_list
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
            'loss_l1': loss_l1,
            'loss_iou': loss_iou
        }
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_logit, pred_dist, anchors, num_anchors_list, stride_tensor_list = head_outs
        pred_scores = F.sigmoid(pred_logit).transpose([0, 2, 1])

        pred_bboxes = self._batch_distance2bbox(
            bbox_center(anchors).unsqueeze(0),
            pred_dist.exp() * stride_tensor_list, img_shape)

        # scale bbox to origin
        scale_factor = scale_factor.flip([1]).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num

    def _iou_loss(self, pbox, gbox, eps=1e-9):
        px1, py1, px2, py2 = pbox.split(4, axis=-1)
        gx1, gy1, gx2, gy2 = gbox.split(4, axis=-1)
        x1 = paddle.maximum(px1, gx1)
        y1 = paddle.maximum(py1, gy1)
        x2 = paddle.minimum(px2, gx2)
        y2 = paddle.minimum(py2, gy2)

        overlap = ((x2 - x1).clip(0)) * ((y2 - y1).clip(0))

        area1 = (px2 - px1) * (py2 - py1)
        area1 = area1.clip(0)

        area2 = (gx2 - gx1) * (gy2 - gy1)
        area2 = area2.clip(0)

        union = area1 + area2 - overlap + eps
        iou = overlap / union
        return 1 - iou.pow(2.0)


@register
class YOLOXHeadv3(nn.Layer):
    __shared__ = ['num_classes', 'data_format']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 feat_channels=256,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5,
                 grid_cell_offset=0.5,
                 ignore_obj=False,
                 static_assigner_epoch=60,
                 static_assigner='ATSSAssigner',
                 assigner='SimOTAv2Assigner',
                 act='mish',
                 nms='MultiClassNMS',
                 loss_weight={'obj': 1.0,
                              'class': 1.0,
                              'iou': 5.0},
                 data_format='NCHW'):
        super(YOLOXHeadv3, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.ignore_obj = ignore_obj
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        self.data_format = data_format

        self.stem_conv = nn.LayerList()
        self.cls_convs = nn.LayerList()

        self.reg_convs = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.obj_preds = nn.LayerList()
        for in_channel in self.in_channels:
            self.stem_conv.append(
                ConvBNLayer(
                    ch_in=in_channel,
                    ch_out=self.feat_channels,
                    filter_size=1,
                    act=act,
                    data_format=data_format))
            self.cls_convs.append(
                nn.Sequential(
                    ConvBNLayer(
                        ch_in=self.feat_channels,
                        ch_out=self.feat_channels,
                        filter_size=3,
                        padding=1,
                        act=act,
                        data_format=data_format),
                    ConvBNLayer(
                        ch_in=self.feat_channels,
                        ch_out=self.feat_channels,
                        filter_size=3,
                        padding=1,
                        act=act,
                        data_format=data_format),
                    nn.Conv2D(
                        self.feat_channels,
                        self.num_classes,
                        1,
                        data_format=self.data_format)))
            self.reg_convs.append(
                nn.Sequential(
                    ConvBNLayer(
                        ch_in=self.feat_channels,
                        ch_out=self.feat_channels,
                        filter_size=3,
                        padding=1,
                        act=act,
                        data_format=data_format),
                    ConvBNLayer(
                        ch_in=self.feat_channels,
                        ch_out=self.feat_channels,
                        filter_size=3,
                        padding=1,
                        act=act,
                        data_format=data_format)))

            self.reg_preds.append(
                nn.Conv2D(
                    self.feat_channels, 4, 1, data_format=self.data_format))
            self.obj_preds.append(
                nn.Conv2D(
                    self.feat_channels, 1, 1, data_format=self.data_format))

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_head, obj_head in zip(self.cls_convs, self.obj_preds):
            constant_(cls_head[-1].weight)
            constant_(cls_head[-1].bias, bias_cls)
            constant_(obj_head.weight)
            constant_(obj_head.bias, bias_cls)

    def _generate_anchors(self, feats):
        anchors, num_anchors_list = [], []
        stride_tensor_list = []
        for feat, stride in zip(feats, self.fpn_strides):
            feat_shape = paddle.shape(feat).detach()
            h, w = feat_shape[2:]
            cell_half_size = self.grid_cell_scale * stride * 0.5
            shift_x = (paddle.arange(end=w) + self.grid_cell_offset) * stride
            shift_y = (paddle.arange(end=h) + self.grid_cell_offset) * stride
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor = paddle.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1)
            anchors.append(anchor.reshape([-1, 4]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor_list.append(
                paddle.full([num_anchors_list[-1], 1], stride))
        return anchors, num_anchors_list, stride_tensor_list

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides)
        anchors, num_anchors_list, stride_tensor_list = self._generate_anchors(
            feats)
        cls_logits, dist_preds, obj_logits = [], [], []
        for feat, conv_stem, conv_cls, conv_reg, pred_reg, pred_obj in zip(
                feats, self.stem_conv, self.cls_convs, self.reg_convs,
                self.reg_preds, self.obj_preds):
            head_feat = conv_stem(feat)
            cls_logit = conv_cls(head_feat)
            reg_feat = conv_reg(head_feat)
            dist_pred = pred_reg(reg_feat)
            obj_logit = pred_obj(reg_feat)
            if self.data_format == 'NCHW':
                cls_logit = cls_logit.transpose([0, 2, 3, 1])
                dist_pred = dist_pred.transpose([0, 2, 3, 1])
                obj_logit = obj_logit.transpose([0, 2, 3, 1])
            cls_logits.append(cls_logit.flatten(1, 2))
            dist_preds.append(dist_pred.flatten(1, 2))
            obj_logits.append(obj_logit.flatten(1, 2))
        cls_logits = paddle.concat(cls_logits, 1)
        dist_preds = paddle.concat(dist_preds, 1)
        obj_logits = paddle.concat(obj_logits, 1)

        anchors = paddle.concat(anchors)
        anchors.stop_gradient = True
        stride_tensor_list = paddle.concat(stride_tensor_list).unsqueeze(0)
        stride_tensor_list.stop_gradient = True

        if self.training:
            return self.get_loss([
                cls_logits, dist_preds, obj_logits, anchors, num_anchors_list,
                stride_tensor_list
            ], targets)
        else:
            return cls_logits, dist_preds, obj_logits, anchors, num_anchors_list, stride_tensor_list

    @staticmethod
    def _batch_distance2bbox(points, distance, max_shapes=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): [B, l, 2]
            distance (Tensor): [B, l, 4]
            max_shapes (tuple): [B, 2], "h w" format, Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, :, 0] - distance[:, :, 0]
        y1 = points[:, :, 1] - distance[:, :, 1]
        x2 = points[:, :, 0] + distance[:, :, 2]
        y2 = points[:, :, 1] + distance[:, :, 3]
        bboxes = paddle.stack([x1, y1, x2, y2], -1)
        if max_shapes is not None:
            out_bboxes = []
            for bbox, max_shape in zip(bboxes, max_shapes):
                bbox[:, 0] = bbox[:, 0].clip(min=0, max=max_shape[1])
                bbox[:, 1] = bbox[:, 1].clip(min=0, max=max_shape[0])
                bbox[:, 2] = bbox[:, 2].clip(min=0, max=max_shape[1])
                bbox[:, 3] = bbox[:, 3].clip(min=0, max=max_shape[0])
                out_bboxes.append(bbox)
            out_bboxes = paddle.stack(out_bboxes)
            return out_bboxes
        return bboxes

    def _obj_loss(self, logit, ious, mask_positive, ignore_mask, gamma=2.0):
        # use gFocal loss
        logit_pos = paddle.masked_select(logit, mask_positive)
        weight_pos = (F.sigmoid(logit_pos) - ious).pow(gamma)
        loss_pos = F.binary_cross_entropy_with_logits(
            logit_pos, ious, weight=weight_pos, reduction='sum')
        if self.ignore_obj:
            mask_negative = mask_positive.astype(
                ignore_mask.dtype) + ignore_mask
        else:
            mask_negative = mask_positive.astype(ignore_mask.dtype)
        logit_neg = paddle.masked_select(logit, mask_negative == 0)
        weight_neg = F.sigmoid(logit_neg).pow(gamma)
        loss_neg = F.binary_cross_entropy_with_logits(
            logit_neg,
            paddle.zeros_like(logit_neg),
            weight=weight_neg,
            reduction='sum')
        return loss_pos + loss_neg

    def get_loss(self, head_outs, gt_meta):
        cls_logits, dist_preds, obj_logits, anchors, num_anchors_list, stride_tensor_list = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None

        # get scores
        pred_scores = (F.sigmoid(cls_logits) * F.sigmoid(obj_logits)).sqrt()
        # distance2bbox
        anchor_centers = bbox_center(anchors)
        pred_bboxes = self._batch_distance2bbox(
            anchor_centers.unsqueeze(0), dist_preds.exp() * stride_tensor_list)

        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, _, ignore_mask = self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                gt_scores=gt_scores)
        else:
            assigned_labels, assigned_bboxes, ignore_mask = self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach(),
                anchor_centers,
                stride_tensor_list.squeeze(0),
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes,
                gt_scores=gt_scores)
        # rescale bbox
        assigned_bboxes /= stride_tensor_list
        pred_bboxes /= stride_tensor_list

        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.astype(paddle.float32).sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            ious = iou_similarity(pred_bboxes_pos, assigned_bboxes_pos)
            ious = paddle.diag(ious.detach()).unsqueeze(-1)
            ious_sum = ious.sum()
            if core.is_compiled_with_dist(
            ) and parallel_helper._is_parallel_ctx_initialized():
                paddle.distributed.all_reduce(ious_sum)
                ious_sum = paddle.clip(
                    ious_sum / paddle.distributed.get_world_size(), min=1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * ious
            loss_iou = loss_iou.sum() / ious_sum
            # cls
            assigned_labels_one_hot = F.one_hot(assigned_labels,
                                                self.num_classes)
            cls_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.num_classes])
            pred_cls_pos = paddle.masked_select(cls_logits, cls_mask).reshape(
                [-1, self.num_classes])
            assigned_cls_pos = paddle.masked_select(assigned_labels_one_hot,
                                                    cls_mask).reshape(
                                                        [-1, self.num_classes])
            loss_cls = F.binary_cross_entropy_with_logits(
                pred_cls_pos, assigned_cls_pos, weight=ious, reduction='sum')
            loss_cls /= ious_sum

            # obj loss
            loss_obj = self._obj_loss(
                obj_logits.squeeze(-1),
                ious.squeeze(-1), mask_positive, ignore_mask)
            loss_obj /= ious_sum
        else:
            loss_cls = paddle.zeros([1])
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])

            # obj loss
            target = paddle.zeros_like(obj_logits)
            weight = (F.sigmoid(obj_logits) - target).pow(2)
            loss_obj = F.binary_cross_entropy_with_logits(
                obj_logits, target, weight=weight)

        loss = self.loss_weight['obj'] * loss_obj + \
               self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou
        out_dict = {
            'loss': loss,
            'loss_obj': loss_obj,
            'loss_cls': loss_cls,
            'loss_l1': loss_l1,
            'loss_iou': loss_iou
        }
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        cls_logits, pred_dist, obj_logits, anchors, num_anchors_list, stride_tensor_list = head_outs

        pred_scores = F.sigmoid(cls_logits) * F.sigmoid(obj_logits)
        pred_scores = pred_scores.transpose([0, 2, 1])

        pred_dist = pred_dist.exp() * stride_tensor_list
        anchor_centers = bbox_center(anchors)
        pred_bboxes = self._batch_distance2bbox(
            anchor_centers.unsqueeze(0), pred_dist, img_shape)

        # scale bbox to origin
        scale_factor = scale_factor.flip([1]).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num


@register
class PPYOLOHeadv2(nn.Layer):
    __shared__ = ['num_classes', 'data_format']
    __inject__ = ['assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 num_convs=2,
                 grid_sampling=True,
                 num_sampling_points=4,
                 feat_channels=256,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5,
                 grid_cell_offset=0.5,
                 ignore_obj=False,
                 static_assigner_epoch=60,
                 assigner='DTSSAssigner',
                 act='mish',
                 nms='MultiClassNMS',
                 loss_weight={'obj': 1.0,
                              'class': 1.0,
                              'iou': 5.0},
                 data_format='NCHW',
                 lr_mult=1.0):
        super(PPYOLOHeadv2, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_convs = num_convs
        self.grid_sampling = grid_sampling
        self.num_sampling_points = num_sampling_points
        self.feat_channels = feat_channels
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.ignore_obj = ignore_obj
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight

        self.static_assigner_epoch = static_assigner_epoch
        self.assigner = assigner
        self.nms = nms
        self.data_format = data_format

        self.conv_reductions = nn.LayerList()
        for in_channel in self.in_channels:
            self.conv_reductions.append(
                ConvBNLayer(
                    ch_in=in_channel,
                    ch_out=self.feat_channels,
                    filter_size=1,
                    act=act,
                    data_format=data_format))

        self.inter_convs = nn.Sequential(*[
            ConvBNLayer(
                ch_in=self.feat_channels,
                ch_out=self.feat_channels,
                filter_size=3,
                padding=1,
                act=act,
                data_format=data_format) for _ in range(self.num_convs)
        ])

        self.cls_preds = nn.Conv2D(self.feat_channels, self.num_classes, 1)
        self.reg_preds = nn.Conv2D(self.feat_channels, 4, 1)

        self.scales_reg = nn.LayerList([ScaleReg() for _ in self.in_channels])

        # grid_sample
        if self.grid_sampling:
            self.cls_offset_conv = nn.Conv2D(
                self.feat_channels,
                self.num_sampling_points * 2,
                3,
                padding=1,
                weight_attr=ParamAttr(learning_rate=lr_mult),
                bias_attr=ParamAttr(learning_rate=lr_mult))
            self.cls_attn = nn.Conv2D(
                self.feat_channels, self.num_sampling_points, 3, padding=1)
            self.reg_offset_conv = nn.Conv2D(
                self.feat_channels,
                self.num_sampling_points * 2,
                3,
                padding=1,
                weight_attr=ParamAttr(learning_rate=lr_mult),
                bias_attr=ParamAttr(learning_rate=lr_mult))
            self.reg_attn = nn.Conv2D(
                self.feat_channels, self.num_sampling_points, 3, padding=1)

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        constant_(self.cls_preds.weight)
        constant_(self.cls_preds.bias, bias_cls)
        normal_(self.reg_preds.bias, std=0.01)
        if self.grid_sampling:
            constant_(self.cls_offset_conv.weight)
            uniform_(self.cls_offset_conv.bias, -0.5, 0.5)
            constant_(self.reg_offset_conv.weight)
            uniform_(self.reg_offset_conv.bias, -0.5, 0.5)
            constant_(self.cls_attn.weight)
            constant_(self.reg_attn.weight)

    def _generate_anchors(self, feats):
        anchors, num_anchors_list = [], []
        stride_tensor_list = []
        for feat, stride in zip(feats, self.fpn_strides):
            feat_shape = paddle.shape(feat).detach()
            h, w = feat_shape[2:]
            cell_half_size = self.grid_cell_scale * stride * 0.5
            shift_x = (paddle.arange(end=w) + self.grid_cell_offset) * stride
            shift_y = (paddle.arange(end=h) + self.grid_cell_offset) * stride
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor = paddle.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1)
            anchors.append(anchor.reshape([-1, 4]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor_list.append(
                paddle.full([num_anchors_list[-1], 1], stride))
        return anchors, num_anchors_list, stride_tensor_list

    def _grid_sample_head(self, prior_feat, offset, attn_weights,
                          reference_points):
        b, _, h, w = prior_feat.shape
        normalize_shape = paddle.to_tensor(
            [w, h], dtype='float32').reshape([1, 1, 1, 2])
        reference_points = reference_points.reshape([1, h, w, 2])
        offset = offset.reshape(
            [b * self.num_sampling_points, 2, h, w]).transpose([0, 2, 3, 1])
        reg_coord = (reference_points + offset) / normalize_shape * 2 - 1
        post_feat = F.grid_sample(
            prior_feat.tile([self.num_sampling_points, 1, 1, 1]),
            reg_coord,
            align_corners=False)
        post_feat = post_feat.reshape([b, self.num_sampling_points, -1, h, w])
        attn_weights = F.softmax(attn_weights.reshape([b, -1, 1, h, w]), axis=1)
        post_feat = (post_feat * attn_weights).sum(axis=1)
        return post_feat

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides)
        anchors, num_anchors_list, stride_tensor_list = self._generate_anchors(
            feats)
        cls_probs_prior, cls_probs_post = [], []
        dist_preds_prior, dist_preds_post = [], []
        for feat, conv_reduction, scale_reg, anchor, stride_tensor in zip(
                feats, self.conv_reductions, self.scales_reg, anchors,
                stride_tensor_list):
            feat = self.inter_convs(conv_reduction(feat))
            cls_prior_prob = F.sigmoid(self.cls_preds(feat))
            dist_prior_pred = scale_reg(self.reg_preds(feat))
            if self.grid_sampling:
                reference_points = bbox_center(anchor) / stride_tensor
                cls_offset = self.cls_offset_conv(feat)
                cls_attn_weights = self.cls_attn(feat)
                cls_post_prob = self._grid_sample_head(
                    cls_prior_prob, cls_offset, cls_attn_weights,
                    reference_points)

                reg_offset = self.reg_offset_conv(feat)
                reg_attn_weights = self.reg_attn(feat)
                dist_post_pred = self._grid_sample_head(
                    dist_prior_pred, reg_offset, reg_attn_weights,
                    reference_points)
            else:
                cls_post_prob, dist_post_pred = cls_prior_prob, dist_prior_pred

            if self.data_format == 'NCHW':
                cls_prior_prob = cls_prior_prob.transpose([0, 2, 3, 1])
                dist_prior_pred = dist_prior_pred.transpose([0, 2, 3, 1])
                cls_post_prob = cls_post_prob.transpose([0, 2, 3, 1])
                dist_post_pred = dist_post_pred.transpose([0, 2, 3, 1])
            cls_probs_prior.append(cls_prior_prob.flatten(1, 2))
            cls_probs_post.append(cls_post_prob.flatten(1, 2))
            dist_preds_prior.append(dist_prior_pred.flatten(1, 2))
            dist_preds_post.append(dist_post_pred.flatten(1, 2))

        cls_probs_prior = paddle.concat(cls_probs_prior, 1)
        cls_probs_post = paddle.concat(cls_probs_post, 1)
        dist_preds_prior = paddle.concat(dist_preds_prior, 1)
        dist_preds_post = paddle.concat(dist_preds_post, 1)

        anchors = paddle.concat(anchors)
        anchors.stop_gradient = True
        stride_tensor_list = paddle.concat(stride_tensor_list).unsqueeze(0)
        stride_tensor_list.stop_gradient = True

        if self.training:
            return self.get_loss([
                cls_probs_prior, cls_probs_post, dist_preds_prior,
                dist_preds_post, anchors, num_anchors_list, stride_tensor_list
            ], targets)
        else:
            return [
                cls_probs_post, dist_preds_post, anchors, num_anchors_list,
                stride_tensor_list
            ]

    @staticmethod
    def _batch_distance2bbox(points, distance, max_shapes=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): [B, l, 2]
            distance (Tensor): [B, l, 4]
            max_shapes (tuple): [B, 2], "h w" format, Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, :, 0] - distance[:, :, 0]
        y1 = points[:, :, 1] - distance[:, :, 1]
        x2 = points[:, :, 0] + distance[:, :, 2]
        y2 = points[:, :, 1] + distance[:, :, 3]
        bboxes = paddle.stack([x1, y1, x2, y2], -1)
        if max_shapes is not None:
            out_bboxes = []
            for bbox, max_shape in zip(bboxes, max_shapes):
                bbox[:, 0] = bbox[:, 0].clip(min=0, max=max_shape[1])
                bbox[:, 1] = bbox[:, 1].clip(min=0, max=max_shape[0])
                bbox[:, 2] = bbox[:, 2].clip(min=0, max=max_shape[1])
                bbox[:, 3] = bbox[:, 3].clip(min=0, max=max_shape[0])
                out_bboxes.append(bbox)
            out_bboxes = paddle.stack(out_bboxes)
            return out_bboxes
        return bboxes

    @staticmethod
    def _batch_iou_similarity(box1, box2, eps=1e-9):
        """Calculate iou of box1 and box2

        Args:
            box1 (Tensor): box with the shape [N, M, 4]
            box2 (Tensor): box with the shape [N, M, 4]

        Return:
            iou (Tensor): iou between box1 and box2 with the shape [N, M]
        """
        for a, b in zip(box1.shape, box2.shape):
            assert a == b, f"iou_shape: {box1.shape} is not equal {box2.shape}!"
        px1y1, px2y2 = box1[..., 0:2], box1[..., 2:4]
        gx1y1, gx2y2 = box2[..., 0:2], box2[..., 2:4]
        x1y1 = paddle.maximum(px1y1, gx1y1)
        x2y2 = paddle.minimum(px2y2, gx2y2)
        overlap = (x2y2 - x1y1).clip(0).prod(-1)
        area1 = (px2y2 - px1y1).clip(0).prod(-1)
        area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
        union = area1 + area2 - overlap + eps
        return overlap / union

    def _gfl_loss(self, probs, labels, ignore_mask=None, gamma=2.0):
        # use gFocal loss
        weight = (probs - labels).pow(gamma)
        if ignore_mask is not None:
            weight *= (1 - ignore_mask)
        loss = F.binary_cross_entropy(
            probs, labels, weight=weight, reduction='sum')
        return loss

    def _loss_cls_reg(self,
                      pred_scores,
                      pred_bboxes,
                      assigned_labels,
                      assigned_bboxes,
                      assigned_scores=None,
                      ignore_mask=None):

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
            ious = self._batch_iou_similarity(
                pred_bboxes_pos, assigned_bboxes_pos).detach().unsqueeze(-1)
            ious_sum = ious.sum()
            if core.is_compiled_with_dist(
            ) and parallel_helper._is_parallel_ctx_initialized():
                paddle.distributed.all_reduce(ious_sum)
                ious_sum = paddle.clip(
                    ious_sum / paddle.distributed.get_world_size(), min=1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * ious
            loss_iou = loss_iou.sum() / ious_sum
            # cls
            assigned_labels_one_hot = F.one_hot(assigned_labels,
                                                self.num_classes)
            if assigned_scores is None:
                assigned_scores = self._batch_iou_similarity(
                    pred_bboxes, assigned_bboxes).detach().unsqueeze(-1)
                assigned_scores *= assigned_labels_one_hot
            loss_cls = self._gfl_loss(pred_scores, assigned_scores, ignore_mask)
            loss_cls /= ious_sum
        else:
            loss_cls = self._gfl_loss(pred_scores,
                                      paddle.zeros_like(pred_scores)).mean()
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
        return {'loss_cls': loss_cls, 'loss_l1': loss_l1, 'loss_iou': loss_iou}

    def get_loss(self, head_outs, gt_meta):
        cls_probs_prior, cls_probs_post, dist_preds_prior, \
        dist_preds_post, anchors, num_anchors_list, stride_tensor_list = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None

        # distance2bbox prior
        anchor_centers = bbox_center(anchors)
        pred_bboxes_prior = self._batch_distance2bbox(
            anchor_centers.unsqueeze(0) / stride_tensor_list,
            dist_preds_prior.exp())
        pred_bboxes_post = self._batch_distance2bbox(
            anchor_centers.unsqueeze(0),
            dist_preds_post.exp() * stride_tensor_list)
        # ATSS + dynamic
        static_assigned_list, dynamic_assigned_list = self.assigner(
            cls_probs_post.detach(),
            pred_bboxes_post.detach(),
            anchors,
            num_anchors_list,
            gt_labels,
            gt_bboxes,
            bg_index=self.num_classes,
            gt_scores=gt_scores)
        # prior loss
        assigned_labels, assigned_bboxes, _, _ = static_assigned_list
        assigned_bboxes /= stride_tensor_list
        loss_prior = self._loss_cls_reg(cls_probs_prior, pred_bboxes_prior,
                                        assigned_labels, assigned_bboxes)
        # total loss
        loss = self.loss_weight['class'] * loss_prior['loss_cls'] + \
               self.loss_weight['iou'] * loss_prior['loss_iou']
        out_dict = {
            'loss': loss,
            'loss_cls_prior': loss_prior['loss_cls'],
            'loss_iou_prior': loss_prior['loss_iou'],
            'loss_l1_prior': loss_prior['loss_l1'],
        }

        # post loss
        if self.grid_sampling:
            assigned_labels, assigned_bboxes, assigned_scores, ignore_mask = dynamic_assigned_list
            assigned_bboxes /= stride_tensor_list
            pred_bboxes_post /= stride_tensor_list
            loss_post = self._loss_cls_reg(cls_probs_post, pred_bboxes_post,
                                           assigned_labels, assigned_bboxes,
                                           assigned_scores,
                                           ignore_mask.unsqueeze(-1))
            loss += self.loss_weight['class'] * loss_post['loss_cls'] + \
               self.loss_weight['iou'] * loss_post['loss_iou']
            out_dict['loss'] = loss
            out_dict.update({
                'loss_cls_post': loss_post['loss_cls'],
                'loss_iou_post': loss_post['loss_iou'],
                'loss_l1_post': loss_post['loss_l1'],
            })
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_dist, anchors, num_anchors_list, stride_tensor_list = head_outs

        pred_scores = pred_scores.transpose([0, 2, 1])

        pred_dist = pred_dist.exp() * stride_tensor_list
        anchor_centers = bbox_center(anchors)
        pred_bboxes = self._batch_distance2bbox(
            anchor_centers.unsqueeze(0), pred_dist, img_shape)

        # scale bbox to origin
        scale_factor = scale_factor.flip([1]).tile([1, 2]).unsqueeze(1)
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
                 static_assigner_epoch=60,
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
            normal_(self.reg_offset_conv2.weight, std=0.001)
            constant_(self.reg_offset_conv2.bias)

    def _generate_anchors(self, feats):
        anchors, num_anchors_list = [], []
        stride_tensor_list = []
        for feat, stride in zip(feats, self.fpn_strides):
            feat_shape = paddle.shape(feat).detach()
            h, w = feat_shape[2:]
            cell_half_size = self.grid_cell_scale * stride * 0.5
            shift_x = (paddle.arange(end=w) + self.grid_cell_offset) * stride
            shift_y = (paddle.arange(end=h) + self.grid_cell_offset) * stride
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor = paddle.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1)
            anchors.append(anchor.reshape([-1, 4]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor_list.append(
                paddle.full([num_anchors_list[-1], 1], stride))
        return anchors, num_anchors_list, stride_tensor_list

    @staticmethod
    def _deform_sampling(feat, offset):
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

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        anchors, num_anchors_list, stride_tensor_list = self._generate_anchors(
            feats)
        cls_score_list, bbox_pred_list = [], []
        for feat, conv_reduction, scale_reg, anchor, stride in zip(
                feats, self.stem_conv, self.scales_regs, anchors,
                self.fpn_strides):
            feat = F.relu(conv_reduction(feat))
            b, _, h, w = feat.shape
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
            reg_bbox = self._batch_distance2bbox(anchor_centers, reg_dist)
            if self.use_align_head:
                reg_bbox = reg_bbox.reshape([b, h, w, 4]).transpose(
                    [0, 3, 1, 2])
                reg_offset = F.relu(self.reg_offset_conv1(feat))
                reg_offset = self.reg_offset_conv2(reg_offset)
                bbox_pred = self._deform_sampling(reg_bbox, reg_offset)
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
        stride_tensor_list = paddle.concat(stride_tensor_list).unsqueeze(0)
        stride_tensor_list.stop_gradient = True

        if self.training:
            return self.get_loss([
                cls_score_list, bbox_pred_list, anchors, num_anchors_list,
                stride_tensor_list
            ], targets)
        else:
            return cls_score_list, bbox_pred_list, anchors, num_anchors_list, stride_tensor_list

    @staticmethod
    def _batch_distance2bbox(points, distance, max_shapes=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): [B, l, 2]
            distance (Tensor): [B, l, 4]
            max_shapes (tuple): [B, 2], "h w" format, Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, :, 0] - distance[:, :, 0]
        y1 = points[:, :, 1] - distance[:, :, 1]
        x2 = points[:, :, 0] + distance[:, :, 2]
        y2 = points[:, :, 1] + distance[:, :, 3]
        bboxes = paddle.stack([x1, y1, x2, y2], -1)
        if max_shapes is not None:
            out_bboxes = []
            for bbox, max_shape in zip(bboxes, max_shapes):
                bbox[:, 0] = bbox[:, 0].clip(min=0, max=max_shape[1])
                bbox[:, 1] = bbox[:, 1].clip(min=0, max=max_shape[0])
                bbox[:, 2] = bbox[:, 2].clip(min=0, max=max_shape[1])
                bbox[:, 3] = bbox[:, 3].clip(min=0, max=max_shape[0])
                out_bboxes.append(bbox)
            out_bboxes = paddle.stack(out_bboxes)
            return out_bboxes
        return bboxes

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
                stride_tensor_list.squeeze(0),
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
            'loss_l1': loss_l1,
            'loss_iou': loss_iou
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
        scale_factor = scale_factor.flip([1]).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num

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

eps = 1e-9

class ESEAttn(nn.Layer):
    def __init__(self, feat_channels):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2D(feat_channels, feat_channels, 1)
        self.conv = ConvNormLayer(
            feat_channels, feat_channels, 1, 1, norm_type='gn')

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        out = self.conv(feat * weight)
        return mish(out)


@register
class PPRefineHeadS(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 grid_size=2,
                 grid_paddings=[0, 0, 1, 1],
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 loss_weight={'class': 1.0,
                              'iou': 2.0}):
        super(PPRefineHeadS, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.grid_size = grid_size
        self.grid_paddings = grid_paddings
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
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
        # pred head
        self.simT_cls = nn.LayerList()
        self.simT_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.simT_cls.append(
                nn.Conv2D(
                    in_c, self.num_classes, 3, padding=1))
            self.simT_reg.append(nn.Conv2D(in_c, 4, 3, padding=1))
        # refine head
        self.cls_align = nn.LayerList()
        self.reg_align = nn.LayerList()
        for in_c in self.in_channels:
            self.cls_align.append(nn.Conv2D(in_c, 1, 3, padding=1))
            self.reg_align.append(
                nn.Conv2D(
                    in_c, grid_size * grid_size * 4, 3, padding=1))

        self.grid_size = grid_size
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

        for cls_, reg_ in zip(self.cls_align, self.reg_align):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)

    def generate_anchors_for_grid_cell(self, feats, fpn_strides, grid_cell_offset=0.5):
        anchor_points = []
        stride_tensor = []
        for feat, stride in zip(feats, fpn_strides):
            _, _, h, w = feat.shape
            shift_x = (paddle.arange(end=w) + grid_cell_offset).reshape((w, ))
            shift_y = (paddle.arange(end=h) + grid_cell_offset).reshape((h, ))
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(paddle.stack([shift_x, shift_y], axis=-1), dtype=feat.dtype)
            anchor_points.append(paddle.expand(anchor_point, [1, h, w, 2]))
            stride_tensor.append(
                paddle.full([1, h * w, 1], stride, dtype=feat.dtype))
        stride_tensor = paddle.concat(stride_tensor, axis=1)
        stride_tensor.stop_gradient = True
        return anchor_points, stride_tensor
    
    def batch_distance2bbox(self, points, distance):
        lt, rb = paddle.split(distance, 2, -1)
        x1y1 = points - lt
        x2y2 = points + rb
        return paddle.concat([x1y1, x2y2], -1)

    def forward_train(self, feats, targets=None):
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
            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.simT_cls[i](self.stem_cls[i](feat, avg_feat))
            reg_dist = self.simT_reg[i](self.stem_reg[i](feat, avg_feat)).exp()

            # cls prediction and alignment
            cls_prob = F.sigmoid(self.cls_align[i](feat))
            cls_score = (F.sigmoid(cls_logit) * cls_prob + eps).sqrt()
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))

            # reg prediction and alignment
            reg_dist = reg_dist.flatten(2).transpose([0, 2, 1])
            reg_bbox = batch_distance2bbox(anchor_centers[i], reg_dist)
            reg_bbox = reg_bbox.transpose([0, 2, 1]).reshape([b, 4, h, w])
            reg_attn = self.reg_align[i](feat).reshape([b, 4, -1, h, w])
            reg_attn = F.softmax(reg_attn, axis=2)
            reg_bbox = F.unfold(
                reg_bbox, self.grid_size,
                paddings=self.grid_paddings).reshape([b, 4, -1, h, w])
            bbox_pred = (reg_attn * reg_bbox).sum(2)
            bbox_pred = bbox_pred.flatten(2).transpose([0, 2, 1])
            bbox_pred_list.append(bbox_pred)
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        bbox_pred_list = paddle.concat(bbox_pred_list, axis=1)

        return self.get_loss([cls_score_list, bbox_pred_list, anchors, anchor_points, num_anchors_list, stride_tensor], targets)

    def forward_eval(self, feats):
        anchor_centers, stride_tensor = self.generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_offset
        )

        cls_score_list, bbox_pred_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.simT_cls[i](self.stem_cls[i](feat, avg_feat))
            reg_dist = self.simT_reg[i](self.stem_reg[i](feat, avg_feat)).exp()

            # cls prediction and alignment
            cls_prob = F.sigmoid(self.cls_align[i](feat))
            cls_score = (F.sigmoid(cls_logit) * cls_prob + eps).sqrt()
            cls_score_list.append(cls_score.reshape([b, self.num_classes, h * w]))

            # reg prediction and alignment
            reg_bbox = self.batch_distance2bbox(anchor_centers[i], reg_dist.transpose([0, 2, 3, 1]))
            reg_attn = self.reg_align[i](feat).reshape([b, 4, self.grid_size * self.grid_size, h, w])
            reg_attn = F.softmax(reg_attn, axis=2)
            reg_bbox = F.unfold(
                reg_bbox, self.grid_size,
                paddings=self.grid_paddings).reshape([b, 4, self.grid_size * self.grid_size, h, w])
            bbox_pred = (reg_attn * reg_bbox).sum(2)
            bbox_pred = bbox_pred.reshape([b, 4, h * w])
            bbox_pred_list.append(bbox_pred)
        cls_score_list = paddle.concat(cls_score_list, axis=2)
        bbox_pred_list = paddle.concat(bbox_pred_list, axis=2)

        return cls_score_list, bbox_pred_list, stride_tensor

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

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_bboxes, anchors, anchor_points,\
        num_anchors_list, stride_tensor = head_outs
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
                pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
            if self.use_varifocal_loss:
                assigned_scores = assigned_ious
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
        pred_bboxes = pred_bboxes.transpose([0, 2, 1]) * stride_tensor
        # clip bbox to origin
        b = pred_bboxes.shape[0]
        img_shape = img_shape.flip(-1).tile([1, 2]).reshape([b, 1, 4])
        pred_bboxes = paddle.where(pred_bboxes < img_shape, pred_bboxes,
                                   img_shape)
        pred_bboxes = paddle.where(pred_bboxes > 0, pred_bboxes,
                                   paddle.zeros_like(pred_bboxes))
        # scale bbox to origin
        scale_factor = scale_factor.flip(-1).tile([1, 2]).reshape([b, 1, 4])
        pred_bboxes /= scale_factor
        return pred_bboxes, pred_scores
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num



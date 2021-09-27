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
from ..initializer import normal_, constant_, bias_init_with_prob
from ..backbones.darknet import ConvBNLayer


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
    __inject__ = ['assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5,
                 grid_cell_offset=0.5,
                 iou_aware=True,
                 iou_aware_factor=0.7,
                 assigner='ATSSAssigner',
                 nms='MultiClassNMS',
                 loss_weight={
                     'obj': 1.0,
                     'class': 1.0,
                     'bbox': 1.0,
                     'iou': 5.0,
                     'iou_aware': 1.0
                 },
                 data_format='NCHW'):
        """
        Head for YOLOv3 network

        Args:
            num_classes (int): number of foreground classes
            iou_aware (bool): whether to use iou_aware
            iou_aware_factor (float): iou aware factor
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLOHead, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight

        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor

        self.assigner = assigner
        self.nms = nms
        self.data_format = data_format

        self.yolo_heads = nn.LayerList()
        for in_channel in self.in_channels:
            if self.iou_aware:
                num_filters = self.num_classes + 6
            else:
                num_filters = self.num_classes + 5
            self.yolo_heads.append(
                nn.Conv2D(
                    in_channels=in_channel,
                    out_channels=num_filters,
                    kernel_size=1,
                    data_format=data_format))

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        for head in self.yolo_heads:
            normal_(head.weight, std=0.01)
            normal_(head.bias, std=0.01)

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
        yolo_outputs = []
        for feat, conv_head in zip(feats, self.yolo_heads):
            yolo_output = conv_head(feat)
            if self.data_format == 'NCHW':
                yolo_output = yolo_output.transpose([0, 2, 3, 1])
            yolo_outputs.append(yolo_output.flatten(1, 2))
        yolo_outputs = paddle.concat(yolo_outputs, 1)

        anchors = paddle.concat(anchors)
        anchors.stop_gradient = True
        stride_tensor_list = paddle.concat(stride_tensor_list).unsqueeze(0)
        stride_tensor_list.stop_gradient = True

        if self.training:
            return self.get_loss(
                [yolo_outputs, anchors, num_anchors_list, stride_tensor_list],
                targets)
        else:
            return yolo_outputs, anchors, num_anchors_list, stride_tensor_list

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

    def _obj_loss(self, pred_obj, target_scores):
        target = (target_scores > 0)
        weight = paddle.where(target, target_scores,
                              paddle.ones_like(target_scores))
        loss = F.binary_cross_entropy_with_logits(
            pred_obj,
            target.astype(pred_obj.dtype),
            weight=weight,
            reduction='sum')
        return loss

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

    def get_loss(self, head_outs, gt_meta):
        yolo_outputs, anchors, num_anchors_list, stride_tensor_list = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None

        if self.iou_aware:
            pred_scores, pred_dist, pred_obj, pred_iou_aware = yolo_outputs.split(
                [self.num_classes, 4, 1, 1], axis=-1)
        else:
            pred_scores, pred_dist, pred_obj = yolo_outputs.split(
                [self.num_classes, 4, 1], axis=-1)

        assigned_labels, assigned_bboxes, assigned_scores, _ = self.assigner(
            anchors,
            num_anchors_list,
            gt_labels,
            gt_bboxes,
            bg_index=self.num_classes,
            gt_scores=gt_scores)
        # rescale bbox
        assigned_bboxes /= stride_tensor_list

        # obj loss
        loss_obj = self._obj_loss(pred_obj, assigned_scores)

        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.astype(paddle.float32).sum()
        assigned_scores_sum = assigned_scores.sum()
        # pos/neg loss
        if num_pos > 0:
            # cls
            assigned_labels_one_hot = F.one_hot(assigned_labels,
                                                self.num_classes)
            loss_cls = F.binary_cross_entropy_with_logits(
                pred_scores,
                assigned_labels_one_hot,
                weight=assigned_scores,
                reduction='sum') / assigned_scores_sum
            # distance2bbox
            anchor_centers = bbox_center(anchors /
                                         stride_tensor_list.squeeze(0))
            pred_bboxes = self._batch_distance2bbox(
                anchor_centers.unsqueeze(0), pred_dist.exp())
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            iou_weight = paddle.masked_select(
                assigned_scores.squeeze(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(
                pred_bboxes_pos, assigned_bboxes_pos,
                reduction='none') * iou_weight
            loss_l1 = loss_l1.sum() / assigned_scores_sum

            loss_iou = self._iou_loss(pred_bboxes_pos,
                                      assigned_bboxes_pos) * iou_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            if self.iou_aware:
                ious = iou_similarity(pred_bboxes_pos, assigned_bboxes_pos)
                ious = paddle.diag(ious)
                pred_iou_aware_pos = paddle.masked_select(
                    pred_iou_aware.squeeze(-1), mask_positive)
                loss_iou_aware = F.binary_cross_entropy_with_logits(
                    pred_iou_aware_pos,
                    ious.detach(),
                    weight=iou_weight.squeeze(-1),
                    reduction='sum') / assigned_scores_sum
        else:
            loss_cls = paddle.zeros([1])
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            if self.iou_aware:
                loss_iou_aware = paddle.zeros([1])

        loss_obj /= assigned_scores_sum.clip(min=1)
        loss = self.loss_weight['obj'] * loss_obj + \
               self.loss_weight['class'] * loss_cls + \
               self.loss_weight['bbox'] * loss_l1 + \
               self.loss_weight['iou'] * loss_iou
        if self.iou_aware:
            loss += (self.loss_weight['iou_aware'] * loss_iou_aware)
        out_dict = {
            'loss': loss,
            'loss_obj': loss_obj,
            'loss_cls': loss_cls,
            'loss_l1': loss_l1,
            'loss_iou': loss_iou
        }
        if self.iou_aware:
            out_dict.update({'loss_iou_aware': loss_iou_aware})
        return out_dict

    def post_process(self, head_outs, img_shape, scale_factor):
        yolo_outputs, anchors, num_anchors_list, stride_tensor_list = head_outs
        if self.iou_aware:
            pred_scores, pred_dist, pred_obj, pred_iou_aware = yolo_outputs.split(
                [self.num_classes, 4, 1, 1], axis=-1)
            pred_obj = F.sigmoid(pred_obj)
            pred_iou_aware = F.sigmoid(pred_iou_aware)
            pred_obj = pred_obj.pow(1 -
                                    self.iou_aware_factor) * pred_iou_aware.pow(
                                        self.iou_aware_factor)
        else:
            pred_scores, pred_dist, pred_obj = yolo_outputs.split(
                [self.num_classes, 4, 1], axis=-1)
            pred_obj = F.sigmoid(pred_obj)

        pred_scores = F.sigmoid(pred_scores) * pred_obj
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
                 assigner='SimOTAAssigner',
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
                    act='relu',
                    data_format=data_format))
            self.cls_convs.append(
                nn.Sequential(
                    ConvBNLayer(
                        ch_in=self.feat_channels,
                        ch_out=self.feat_channels,
                        filter_size=3,
                        padding=1,
                        act='relu',
                        data_format=data_format),
                    ConvBNLayer(
                        ch_in=self.feat_channels,
                        ch_out=self.feat_channels,
                        filter_size=3,
                        padding=1,
                        act='relu',
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
                        act='relu',
                        data_format=data_format),
                    ConvBNLayer(
                        ch_in=self.feat_channels,
                        ch_out=self.feat_channels,
                        filter_size=3,
                        padding=1,
                        act='relu',
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

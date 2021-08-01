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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register

__all__ = ['FCOSMono3DLoss']


@register
class FCOSMono3DLoss(nn.Layer):
    """
    FCOSMono3DLoss
    Args:
        num_classes (int):
        loss_alpha (float): alpha in focal loss
        loss_gamma (float): gamma in focal loss
        reg_weights (dict): weight for location loss
    """
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=3,
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 reg_weights={
                     'offset': 1.0,
                     'depth': 0.2,
                     'size': 1.0,
                     'rotation': 1.0
                 }):
        super(FCOSMono3DLoss, self).__init__()
        self.num_classes = num_classes
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.reg_weights = reg_weights

    def _get_pred_gt_mask_select_tensor(self, tensor, mask):
        mask = paddle.tile(mask, [1, 1, tensor.shape[-1]])
        return paddle.masked_select(tensor, mask.astype(paddle.bool))

    def _get_loss_regression(self, preds, targets, mask_positive):

        loss_offset = F.smooth_l1_loss(
            self._get_pred_gt_mask_select_tensor(preds[:, :, :2],
                                                 mask_positive),
            self._get_pred_gt_mask_select_tensor(targets[:, :, :2],
                                                 mask_positive),
            reduction='sum') * self.reg_weights['offset']

        loss_depth = F.smooth_l1_loss(
            self._get_pred_gt_mask_select_tensor(preds[:, :, 2:3],
                                                 mask_positive),
            self._get_pred_gt_mask_select_tensor(targets[:, :, 2:3],
                                                 mask_positive),
            reduction='sum') * self.reg_weights['depth']

        loss_size = F.smooth_l1_loss(
            self._get_pred_gt_mask_select_tensor(preds[:, :, 3:6],
                                                 mask_positive),
            self._get_pred_gt_mask_select_tensor(targets[:, :, 3:6],
                                                 mask_positive),
            reduction='sum') * self.reg_weights['size']

        pred_rot = self._get_pred_gt_mask_select_tensor(preds[:, :, 6:],
                                                        mask_positive)
        target_rot = self._get_pred_gt_mask_select_tensor(targets[:, :, 6:],
                                                          mask_positive)
        pred_rot_encode = pred_rot.sin() * target_rot.cos()
        target_rot_encode = pred_rot.cos() * target_rot.sin()
        loss_rotation_encode = F.smooth_l1_loss(
            pred_rot_encode, target_rot_encode,
            reduction='sum') * self.reg_weights['rotation']

        loss = loss_offset + loss_depth + loss_size + loss_rotation_encode
        return loss

    @paddle.no_grad()
    def log_metric(self, loss_dict, mask_positive, cls_logits, bboxes_reg,
                   centerness, direction_logits, target_class,
                   target_regression, target_centerness, target_direction):
        if mask_positive.sum() <= 0:
            loss_dict['log_recall'] = paddle.to_tensor([0.])
            loss_dict['log_precision'] = paddle.to_tensor([0.])
            for i in range(7):
                loss_dict[f'log_{i}'] = paddle.to_tensor([0.])
            loss_dict['log_dir'] = paddle.to_tensor([0.])
            print(f'num pos: {mask_positive.sum().item()}')
            return loss_dict
        # cls
        cls_prob = F.sigmoid(cls_logits)
        centerness = F.sigmoid(centerness)
        score = cls_prob * centerness
        cls_prob_max = score.max(axis=-1)
        cls_prob_max_ind = score.argmax(axis=-1)
        pred_cls = self._get_pred_gt_mask_select_tensor(
            cls_prob_max_ind.unsqueeze(-1), mask_positive)
        target_cls = self._get_pred_gt_mask_select_tensor(
            target_class.unsqueeze(-1), mask_positive)
        num_correct = (pred_cls == target_cls).astype(paddle.float32).sum()
        loss_dict['log_recall'] = num_correct / target_cls.shape[0]
        num_pred = (cls_prob_max >= 0.05).astype(paddle.float32).sum()
        loss_dict['log_precision'] = num_correct / (num_pred + 1e-9)
        # reg
        pred_reg = self._get_pred_gt_mask_select_tensor(
            bboxes_reg, mask_positive).reshape([-1, bboxes_reg.shape[-1]])
        target_reg = self._get_pred_gt_mask_select_tensor(
            target_regression,
            mask_positive).reshape([-1, bboxes_reg.shape[-1]])
        abs_err = (pred_reg - target_reg).abs() / target_reg.abs()
        abs_err = abs_err.mean(axis=0)
        for i in range(len(abs_err)):
            loss_dict[f'log_{i}'] = abs_err[i]
        # dir
        pred_dir = self._get_pred_gt_mask_select_tensor(
            direction_logits, mask_positive).reshape([-1, 2]).argmax(axis=-1)
        target_dir = self._get_pred_gt_mask_select_tensor(
            target_direction.unsqueeze(-1), mask_positive)
        recall_dir = (pred_dir == target_dir
                      ).astype(paddle.float32).sum() / target_dir.shape[0]
        loss_dict['log_dir'] = recall_dir

        return loss_dict

    def forward(self, cls_logits, bboxes_reg, centerness, direction_logits,
                target_class, target_regression, target_centerness,
                target_direction):

        bg_index = cls_logits.shape[-1]
        mask_positive = (
            target_class != bg_index).astype(paddle.float32).unsqueeze(-1)
        num_positive = mask_positive.sum()

        # 1. cls_logits: sigmoid_focal_loss
        # expand onehot labels
        target_class_one_hot = F.one_hot(target_class, num_classes=bg_index + 1)
        target_class_one_hot = target_class_one_hot[:, :, :-1]
        # sigmoid_focal_loss
        cls_loss = F.sigmoid_focal_loss(
            cls_logits,
            target_class_one_hot,
            alpha=self.loss_alpha,
            gamma=self.loss_gamma) / num_positive.clip(min=1)

        if num_positive > 0:
            # 2. bboxes_reg: smooth_l1_loss
            reg_loss = self._get_loss_regression(
                bboxes_reg, target_regression,
                mask_positive) / num_positive.clip(min=1)

            # 3. centerness: binary_cross_entropy_with_logits
            pred_ctn = self._get_pred_gt_mask_select_tensor(centerness,
                                                            mask_positive)
            target_ctn = self._get_pred_gt_mask_select_tensor(target_centerness,
                                                              mask_positive)
            ctn_loss = F.binary_cross_entropy_with_logits(
                pred_ctn, target_ctn,
                reduction='sum') / num_positive.clip(min=1)

            # 4. direction_logits: cross_entropy
            pred_dir = self._get_pred_gt_mask_select_tensor(
                direction_logits, mask_positive).reshape([-1, 2])
            target_dir = self._get_pred_gt_mask_select_tensor(
                target_direction.unsqueeze(-1), mask_positive)
            dir_loss = F.cross_entropy(
                pred_dir, target_dir,
                reduction='sum') / num_positive.clip(min=1)
        else:
            reg_loss = paddle.to_tensor([0.])
            ctn_loss = paddle.to_tensor([0.])
            dir_loss = paddle.to_tensor([0.])

        loss_all = {
            "loss_cls": cls_loss,
            "loss_box": reg_loss,
            "loss_centerness": ctn_loss,
            "loss_direction": dir_loss
        }

        loss_all = self.log_metric(loss_all, mask_positive, cls_logits,
                                   bboxes_reg, centerness, direction_logits,
                                   target_class, target_regression,
                                   target_centerness, target_direction)

        return loss_all

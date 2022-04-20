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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['MIAOD']


@register
class MIAOD(BaseArch):
    """
    Generalized Focal Loss network, see https://arxiv.org/abs/2006.04388

    Args:
        backbone (object): backbone instance
        neck (object): 'FPN' instance
        head (object): 'GFLHead' instance
    """

    __category__ = 'architecture'

    def __init__(self, backbone, neck, head):
        super(MIAOD, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self._stage_id = 0

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "head": head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)
        if not self.training:
            # eval phase
            head_outs = self.head(fpn_feats)
            if self.inputs.get('eval_uncertainty', False):
                uncertainty = self.calculate_uncertainty(head_outs)
                out = {'uncertainty': uncertainty}
            else:
                im_shape = self.inputs['im_shape']
                scale_factor = self.inputs['scale_factor']
                out = self.head.post_process(head_outs, im_shape, scale_factor)
            return out
        else:
            # train phase
            if self.inputs['stage_id'] != self._stage_id:
                for tensor in fpn_feats:
                    if self.inputs['stage_id'] == 2:
                        tensor.stop_gradient = True
                    else:
                        tensor.stop_gradient = False
                self._stage_id = self.inputs['stage_id']
            head_outs = self.head(fpn_feats, self.inputs['stage_id'])
            return head_outs

    def get_loss(self):
        loss = {}
        head_outs = self._forward()
        loss_gfl = self.head.get_loss(head_outs, self.inputs)
        loss.update(loss_gfl)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        return self._forward()

    def calculate_uncertainty(self, head_outs):
        uncertainty = []
        y_head_f_1, y_head_f_2, _, y_head_cls = head_outs
        y_head_f_2 = [y2.flatten(2).transpose([0, 2, 1]) for y2 in y_head_f_2]
        y_head_f_1 = paddle.concat(y_head_f_1, axis=1)
        y_head_f_2 = paddle.concat(y_head_f_2, axis=1)  # [B, SUM(hw), C]

        y_head_f_2 = paddle.nn.functional.sigmoid(y_head_f_2)
        for y1, y2 in zip(y_head_f_1, y_head_f_2):  # [sum(wh), c]
            loss_l2_p = paddle.pow(y1 - y2, 2)
            uncertainty_all_N = paddle.mean(loss_l2_p, axis=1)  # (sum(wh))
            arg = paddle.argsort(uncertainty_all_N)
            uncertainty_single = paddle.mean(uncertainty_all_N[arg[-10000:]])
            uncertainty.append(uncertainty_single)
        return paddle.concat(uncertainty)

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
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['FCOSMono3D']


@register
class FCOSMono3D(BaseArch):
    """
    FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection,
    see https://arxiv.org/abs/2104.10956

    Args:
        backbone (nn.Layer): backbone instance
        neck (nn.Layer): `FPN` instance
        fcos_mono3d_head (nn.Layer): `FCOSMono3DHead` instance
        fcos_mono3d_post_process (object): `FCOSMono3DPostProcess` instance
    """

    __category__ = 'architecture'
    __inject__ = ['fcos_mono3d_post_process']

    def __init__(self,
                 backbone,
                 neck,
                 fcos_mono3d_head,
                 fcos_mono3d_post_process='FCOSMono3DPostProcess'):
        super(FCOSMono3D, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.fcos_mono3d_head = fcos_mono3d_head
        self.fcos_mono3d_post_process = fcos_mono3d_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        fcos_mono3d_head = create(cfg['fcos_mono3d_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "fcos_mono3d_head": fcos_mono3d_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)
        fcos_head_outs = self.fcos_mono3d_head(fpn_feats, self.training)
        if not self.training:
            return self.fcos_mono3d_post_process(
                fcos_head_outs, self.inputs['scale_factor'], self.inputs['P2'])
        else:
            return fcos_head_outs

    def get_loss(self):
        fcos_head_outs = self._forward()
        loss = self.fcos_mono3d_head.get_loss(
            fcos_head_outs, self.inputs['type'], self.inputs['bbox_2d'],
            self.inputs['center_2d'], self.inputs['depth'],
            self.inputs['dimensions'], self.inputs['rotation_y'])
        total_loss = paddle.add_n(
            [v for k, v in loss.items() if 'log' not in k])
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output

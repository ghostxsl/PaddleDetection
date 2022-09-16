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

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['ViTDet']


@register
class ViTDet(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['data_format']

    def __init__(self,
                 backbone='VisionTransformer',
                 neck=None,
                 head='PPYOLOEHead',
                 data_format='NCHW'):
        super(ViTDet, self).__init__(data_format=data_format)
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None

        # head
        if neck:
            kwargs = {'input_shape': neck.out_shape}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "head": head,
        }

    def _forward(self):
        feats = self.backbone(self.inputs)
        if self.neck is not None:
            feats = self.neck(feats)

        if self.training:
            return self.head(feats, self.inputs)
        else:
            head_outs = self.head(feats)
            bbox, bbox_num = self.head.post_process(head_outs,
                                                    self.inputs['scale_factor'])
            return {'bbox': bbox, 'bbox_num': bbox_num}

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

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

import copy
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register, serializable
from ..bbox_utils import bbox_overlaps

__all__ = [
    '_get_clones', 'bbox_overlaps', 'bbox_cxcywh_to_xyxy',
    'bbox_xyxy_to_cxcywh', 'constant_', 'normal_', 'uniform_',
    '_calculate_fan_in_and_fan_out', 'xavier_uniform_', 'linear_init_',
    'conv_init_', 'PositionEmbedding', 'sigmoid_focal_loss'
]


def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


def bbox_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)


def bbox_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return paddle.stack(b, axis=-1)


def constant_(tensor: paddle.Tensor, value=0.):
    with paddle.no_grad():
        tensor.set_value(paddle.full_like(tensor, value, dtype=tensor.dtype))


def normal_(tensor: paddle.Tensor, mean=0., std=1.):
    with paddle.no_grad():
        tensor.set_value(paddle.normal(mean=mean, std=std, shape=tensor.shape))


def uniform_(tensor: paddle.Tensor, min, max):
    with paddle.no_grad():
        tensor.set_value(
            paddle.uniform(
                shape=tensor.shape, dtype=tensor.dtype, min=min, max=max))


def _calculate_fan_in_and_fan_out(tensor: paddle.Tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0, 0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def xavier_uniform_(tensor: paddle.Tensor, gain=1.):
    # This is an implementation like pytorch
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def linear_init_(module):
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    uniform_(module.bias, -bound, bound)


def conv_init_(module):
    bound = 1 / math.sqrt(math.prod(module.weight.shape[1:]))
    uniform_(module.weight, -bound, bound)
    uniform_(module.bias, -bound, bound)


@register
@serializable
class PositionEmbedding(nn.Layer):
    def __init__(self,
                 num_pos_feats=128,
                 temperature=10000,
                 normalize=True,
                 scale=None,
                 embed_type='sine',
                 num_embeddings=50):
        super(PositionEmbedding, self).__init__()
        assert embed_type in ['sine', 'learned']

        self.embed_type = embed_type
        if self.embed_type == 'sine':
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            if scale is not None and normalize is False:
                raise ValueError("normalize should be True if scale is passed")
            if scale is None:
                scale = 2 * math.pi
            self.scale = scale
        elif self.embed_type == 'learned':
            self.row_embed = nn.Embedding(num_embeddings, num_pos_feats)
            self.col_embed = nn.Embedding(num_embeddings, num_pos_feats)
        else:
            raise ValueError(f"not supported {self.embed_type}")

    def forward(self, mask):
        """
        Args:
            mask (Tensor): [B, H, W]
        Returns:
            pos (Tensor): [B, C, H, W]
        """
        assert mask.dtype == paddle.bool
        if self.embed_type == 'sine':
            mask = mask.astype('float32')
            y_embed = mask.cumsum(1, dtype='float32')
            x_embed = mask.cumsum(2, dtype='float32')
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            dim_t = 2 * (paddle.arange(self.num_pos_feats) //
                         2).astype('float32')
            dim_t = self.temperature**(dim_t / self.num_pos_feats)

            pos_x = x_embed.unsqueeze(-1) / dim_t
            pos_y = y_embed.unsqueeze(-1) / dim_t
            pos_x = paddle.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
                axis=4).flatten(3)
            pos_y = paddle.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
                axis=4).flatten(3)
            pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])
            return pos
        elif self.embed_type == 'learned':
            h, w = mask.shape[-2:]
            i = paddle.arange(w)
            j = paddle.arange(h)
            x_emb = self.col_embed(i)
            y_emb = self.row_embed(j)
            pos = paddle.concat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                axis=-1).transpose([2, 0, 1]).unsqueeze(0).tile(mask.shape[0],
                                                                1, 1, 1)
            return pos
        else:
            raise ValueError(f"not supported {self.embed_type}")


def sigmoid_focal_loss(logit, label, normalizer=1.0, alpha=0.25, gamma=2.0):
    prob = F.sigmoid(logit)
    ce_loss = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    p_t = prob * label + (1 - prob) * (1 - label)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        loss = alpha_t * loss
    return loss.mean(1).sum() / normalizer if normalizer > 1. else loss.mean(
        1).sum()

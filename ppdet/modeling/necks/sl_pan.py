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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from ..backbones.cspresnet import ConvBNLayer

from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec

__all__ = ['SLPAN']


class CSPStage(nn.Layer):
    def __init__(self, block_fn, channels, n=3, act='swish'):
        super(CSPStage, self).__init__()

        ch_mid = channels // 2
        self.conv1 = ConvBNLayer(channels, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(channels, ch_mid, 1, act=act)
        self.convs = nn.Sequential()
        for i in range(n):
            self.convs.add_sublayer(
                str(i), eval(block_fn)(ch_mid, ch_mid, act=act, shortcut=False))
        self.conv3 = ConvBNLayer(ch_mid * 2, channels, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = paddle.concat([y1, y2], axis=1)
        y = self.conv3(y)
        return y


class CSPPAN(nn.Layer):
    def __init__(self,
                 channels=256,
                 num_levels=4,
                 norm_type='bn',
                 norm_groups=32,
                 act='swish'):
        super(CSPPAN, self).__init__()
        self.channels = channels
        self.num_levels = num_levels

        # up
        self.conv_up = nn.LayerList([
            CSPStage(
                self.channels,
                kernel_size=kernel_size,
                norm_type=norm_type,
                norm_groups=norm_groups,
                act=act) for _ in range(self.num_levels - 1)
        ])
        # down
        self.conv_down = nn.LayerList([
            CSPStage(
                self.channels,
                kernel_size=kernel_size,
                norm_type=norm_type,
                norm_groups=norm_groups,
                act=act) for _ in range(self.num_levels - 1)
        ])

    def _feature_fusion_cell(self,
                             conv_layer,
                             lateral_feat,
                             sampling_feat,
                             route_feat=None):
        if route_feat is not None:
            out_feat = lateral_feat + sampling_feat + route_feat
        else:
            out_feat = lateral_feat + sampling_feat

        out_feat = conv_layer(out_feat)
        return out_feat

    def forward(self, feats):
        # feats: [P3 - P6]
        lateral_feats = []
        # up
        up_feature = feats[-1]
        for i, feature in enumerate(feats[::-1]):
            if i == 0:
                lateral_feats.append(feature)
            else:
                shape = paddle.shape(feature)
                up_feature = F.interpolate(
                    up_feature, size=[shape[2], shape[3]])
                lateral_feature = self._feature_fusion_cell(self.conv_up[i - 1],
                                                            feature, up_feature)
                lateral_feats.append(lateral_feature)
                up_feature = lateral_feature

        out_feats = []
        # down
        down_feature = lateral_feats[-1]
        for i, (lateral_feature,
                route_feature) in enumerate(zip(lateral_feats[::-1], feats)):
            if i == 0:
                out_feats.append(lateral_feature)
            else:
                down_feature = F.max_pool2d(down_feature, 3, 2, 1)
                if i == len(feats) - 1:
                    route_feature = None
                    weights = self.down_weights[
                        i - 1][:2] if self.use_weighted_fusion else None
                else:
                    weights = self.down_weights[
                        i - 1] if self.use_weighted_fusion else None
                out_feature = self._feature_fusion_cell(
                    self.conv_down[i - 1], lateral_feature, down_feature,
                    route_feature)
                out_feats.append(out_feature)
                down_feature = out_feature

        return out_feats


@register
@serializable
class SLPAN(nn.Layer):
    def __init__(self,
                 in_channels=(256, 512, 1024),
                 out_channel=256,
                 num_extra_levels=1,
                 fpn_strides=(8, 16, 32),
                 norm_type='bn',
                 norm_groups=32,
                 act='swish'):
        super(SLPAN, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', None]
        assert act in ['swish', 'relu', None]
        assert num_extra_levels >= 0, \
            "The `num_extra_levels` must be non negative(>=0)."

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_extra_levels = num_extra_levels
        self.norm_type = norm_type
        self.norm_groups = norm_groups
        self.act = act
        self.num_levels = len(in_channels) + num_extra_levels
        if len(fpn_strides) != self.num_levels:
            for i in range(num_extra_levels):
                fpn_strides += [fpn_strides[-1] * 2]
        self.fpn_strides = fpn_strides

        self.lateral_convs = nn.LayerList()
        for in_c in in_channels:
            self.lateral_convs.append(
                ConvNormLayer(in_c, self.out_channel, 1, 1))
        if self.num_extra_levels > 0:
            self.extra_convs = nn.LayerList()
            for i in range(self.num_extra_levels):
                if i == 0:
                    self.extra_convs.append(
                        ConvNormLayer(self.in_channels[-1], self.out_channel, 3,
                                      2))
                else:
                    self.extra_convs.append(nn.MaxPool2D(3, 2, 1))

        self.pan = CSPPAN(
            self.out_channel,
            self.num_levels,
            norm_type=norm_type,
            norm_groups=norm_groups,
            act=act)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'fpn_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channel, stride=s) for s in self.fpn_strides
        ]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        fpn_feats = []
        for conv_layer, feature in zip(self.lateral_convs, feats):
            fpn_feats.append(conv_layer(feature))
        if self.num_extra_levels > 0:
            feat = feats[-1]
            for conv_layer in self.extra_convs:
                feat = conv_layer(feat)
                fpn_feats.append(feat)

        return self.pan(fpn_feats)

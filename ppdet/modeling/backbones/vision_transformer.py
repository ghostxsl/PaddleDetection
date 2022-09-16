# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Constant, TruncatedNormal

from ppdet.modeling.shape_spec import ShapeSpec
from ppdet.core.workspace import register, serializable

from .transformer_utils import zeros_, DropPath, Identity, add_decomposed_rel_pos, window_partition, window_unpartition
from ..initializer import linear_init_


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='nn.GELU',
                 drop=0.,
                 lr_factor=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
            weight_attr=ParamAttr(learning_rate=lr_factor),
            bias_attr=ParamAttr(learning_rate=lr_factor))
        self.act = eval(act_layer)()
        self.fc2 = nn.Linear(
            hidden_features,
            out_features,
            weight_attr=ParamAttr(learning_rate=lr_factor),
            bias_attr=ParamAttr(learning_rate=lr_factor))
        self.drop = nn.Dropout(drop)

        self._init_weights()

    def _init_weights(self):
        linear_init_(self.fc1)
        linear_init_(self.fc2)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 window_size=None,
                 lr_factor=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.use_rel_pos = False if window_size is None else use_rel_pos
        self.rel_pos_zero_init = rel_pos_zero_init
        self.window_size = window_size
        self.lr_factor = lr_factor

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            weight_attr=ParamAttr(learning_rate=lr_factor),
            bias_attr=ParamAttr(learning_rate=lr_factor) if qkv_bias else False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(
            dim,
            dim,
            weight_attr=ParamAttr(learning_rate=lr_factor),
            bias_attr=ParamAttr(learning_rate=lr_factor))
        self.proj_drop = nn.Dropout(proj_drop)

        self._init_weights()

    def _init_weights(self):
        linear_init_(self.qkv)
        linear_init_(self.proj)

        if self.use_rel_pos:
            self.rel_pos_h = self.create_parameter(
                [2 * self.window_size - 1, self.head_dim],
                attr=ParamAttr(learning_rate=self.lr_factor),
                default_initializer=Constant(value=0.))
            self.rel_pos_w = self.create_parameter(
                [2 * self.window_size - 1, self.head_dim],
                attr=ParamAttr(learning_rate=self.lr_factor),
                default_initializer=Constant(value=0.))

            if not self.rel_pos_zero_init:
                TruncatedNormal(self.rel_pos_h, std=0.02)
                TruncatedNormal(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, C = paddle.shape(x)

        qkv = self.qkv(x).reshape(
            [B, H * W, 3, self.num_heads, self.head_dim]).transpose(
                [2, 0, 3, 1, 4])
        q, k, v = paddle.unbind(
            qkv.reshape([3, B * self.num_heads, H * W, self.head_dim]))
        attn = (q.matmul(k.transpose((0, 2, 1)))) * self.scale

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                          self.rel_pos_w, (H, W), (H, W))

        attn = self.attn_drop(F.softmax(attn, axis=-1))

        x = attn.matmul(v).reshape(
            [B, self.num_heads, H * W, self.head_dim]).transpose(
                [0, 2, 1, 3]).reshape([B, H, W, C])
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 window_size=None,
                 gamma=None,
                 act_layer='nn.GELU',
                 norm_layer='nn.LayerNorm',
                 lr_factor=1.0):
        super().__init__()
        self.window_size = window_size

        self.norm1 = eval(norm_layer)(dim,
                                      weight_attr=ParamAttr(
                                          learning_rate=lr_factor,
                                          regularizer=L2Decay(0.0)),
                                      bias_attr=ParamAttr(
                                          learning_rate=lr_factor,
                                          regularizer=L2Decay(0.0)))
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            lr_factor=lr_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = eval(norm_layer)(dim,
                                      weight_attr=ParamAttr(
                                          learning_rate=lr_factor,
                                          regularizer=L2Decay(0.0)),
                                      bias_attr=ParamAttr(
                                          learning_rate=lr_factor,
                                          regularizer=L2Decay(0.0)))
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop,
                       lr_factor=lr_factor)

        self.gamma = gamma
        if self.gamma is not None:
            self.gamma_1 = self.create_parameter(
                shape=([1]),
                attr=ParamAttr(learning_rate=lr_factor),
                default_initializer=Constant(value=self.gamma))
            self.gamma_2 = self.create_parameter(
                shape=([1]),
                attr=ParamAttr(learning_rate=lr_factor),
                default_initializer=Constant(value=self.gamma))

    def forward(self, x):
        _, H, W, _ = paddle.shape(x)
        y = self.norm1(x)
        if self.window_size is not None:
            y, pad_hw = window_partition(y, self.window_size)

        y = self.attn(y) if self.gamma is None else self.gamma_1 * self.attn(y)
        if self.window_size is not None:
            y = window_unpartition(y, self.window_size, pad_hw, (H, W))
        x = x + self.drop_path(y)

        y = self.norm2(x)
        y = self.mlp(y) if self.gamma is None else self.gamma_2 * self.mlp(y)
        x = x + self.drop_path(y)
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 lr_factor=0.01,
                 data_format="NHWC"):
        super().__init__()

        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            weight_attr=ParamAttr(learning_rate=lr_factor),
            bias_attr=ParamAttr(learning_rate=lr_factor),
            data_format=data_format)

    def forward(self, x):
        return self.proj(x)


@register
@serializable
class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch input
    """

    def __init__(self,
                 img_size=(640, 640),
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer='nn.GELU',
                 norm_layer='nn.LayerNorm',
                 lr_decay_rate=1.0,
                 global_attn_indexes=(2, 5, 8, 11),
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 window_size=None,
                 gamma=None,
                 use_abs_pos_emb=False,
                 use_sincos_pos_emb=True,
                 out_indices=(11, ),
                 with_fpn=True,
                 fpn_dim=[192, 384, 768],
                 out_format="NHWC",
                 *args,
                 **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.lr_decay_rate = lr_decay_rate
        self.global_attn_indexes = global_attn_indexes
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_sincos_pos_emb = use_sincos_pos_emb
        self.with_fpn = with_fpn
        self.out_format = out_format
        self.data_format = "NHWC"

        self.patch_h = img_size[0] // patch_size
        self.patch_w = img_size[1] // patch_size
        self.num_patches = self.patch_h * self.patch_w

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            data_format=self.data_format)

        self.pos_embed = self.create_parameter(
            [1, self.patch_h, self.patch_w, embed_dim],
            attr=ParamAttr(learning_rate=0.01),
            default_initializer=TruncatedNormal(
                std=0.02)) if use_abs_pos_emb else None

        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=None
                if i in self.global_attn_indexes else window_size,
                gamma=gamma,
                act_layer=act_layer,
                lr_factor=self.get_vit_lr_decay_rate(i),
                norm_layer=norm_layer) for i in range(depth)
        ])

        assert len(out_indices) <= 4, 'out_indices out of bound'
        self.out_indices = out_indices

        if self.with_fpn:
            self.out_channels = fpn_dim
            self.out_strides = [8, 16, 32]
            self.init_fpn(embed_dim, fpn_dim, norm_layer)
        else:
            self.out_channels = [embed_dim for _ in range(len(out_indices))]
            self.out_strides = [16 for _ in range(len(out_indices))]

    def get_vit_lr_decay_rate(self, layer_id):
        return self.lr_decay_rate**(self.depth - layer_id)

    def init_fpn(self,
                 in_dim=768,
                 out_dim=[192, 384, 768],
                 norm_layer='nn.LayerNorm'):
        self.p3 = nn.Sequential(
            nn.Upsample(
                scale_factor=2.0, mode='bilinear',
                data_format=self.data_format),
            nn.Conv2D(
                in_dim,
                out_dim[0],
                1,
                bias_attr=False,
                data_format=self.data_format),
            Block(
                dim=out_dim[0],
                num_heads=self.num_heads,
                drop_path=0.1,
                window_size=None,
                norm_layer=norm_layer))

        self.p4 = nn.Sequential(
            nn.Conv2D(
                in_dim,
                out_dim[1],
                1,
                bias_attr=False,
                data_format=self.data_format),
            Block(
                dim=out_dim[1],
                num_heads=self.num_heads,
                drop_path=0.1,
                window_size=None,
                norm_layer=norm_layer))

        self.p5 = nn.Sequential(
            nn.Conv2D(
                in_dim,
                out_dim[2],
                3,
                stride=2,
                padding=1,
                bias_attr=False,
                data_format=self.data_format),
            Block(
                dim=out_dim[2],
                num_heads=self.num_heads,
                drop_path=0.1,
                window_size=None,
                norm_layer=norm_layer))

    def interpolate_pos_encoding(self, h, w):
        if h == self.patch_h and w == self.patch_w:
            return self.pos_embed

        return F.interpolate(
            self.pos_embed,
            size=(h, w),
            mode='bicubic',
            data_format=self.data_format)

    def get_2d_sincos_position_embedding(
            self,
            h,
            w,
            temperature=10000., ):
        grid_x = paddle.arange(w, dtype=paddle.float32)
        grid_y = paddle.arange(h, dtype=paddle.float32)
        grid_y, grid_x = paddle.meshgrid(grid_y, grid_x)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_x = grid_x.flatten()[..., None] @omega[None]
        out_y = grid_y.flatten()[..., None] @omega[None]

        pos_emb = paddle.concat(
            [
                paddle.sin(out_y), paddle.cos(out_y), paddle.sin(out_x),
                paddle.cos(out_x)
            ],
            axis=1)[None, :, :]

        return pos_emb.reshape([-1, h, w, self.embed_dim])

    def forward(self, inputs):
        # NHWC format
        x = inputs['image'].transpose([0, 2, 3, 1])

        x = self.patch_embed(x)
        B, Hp, Wp, C = paddle.shape(x)

        if self.use_abs_pos_emb:
            x = x + self.interpolate_pos_encoding(Hp, Wp)
        elif self.use_sincos_pos_emb:
            x = x + self.get_2d_sincos_position_embedding(Hp, Wp)

        feats = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)

            if idx in self.out_indices:
                feats.append(x)

        if self.with_fpn:
            fpns = [self.p3, self.p4, self.p5]

            if len(feats) == 1:
                if self.out_format == "NCHW":
                    outputs = [
                        m(feats[-1]).transpose([0, 3, 1, 2]) for m in fpns
                    ]
                else:
                    outputs = [m(feats[-1]) for m in fpns]
            else:
                assert len(feats) == len(fpns), '`feats` not equal to `fpns`'
                outputs = [m(feats[i]) for i, m in enumerate(fpns)]

            return outputs

        return feats

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=c, stride=s)
            for c, s in zip(self.out_channels, self.out_strides)
        ]

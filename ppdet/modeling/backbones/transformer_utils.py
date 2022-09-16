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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.nn.initializer import TruncatedNormal, Constant, Assign

# Common initializations
ones_ = Constant(value=1.)
zeros_ = Constant(value=0.)
trunc_normal_ = TruncatedNormal(std=.02)


# Common Layers
def drop_path(x, drop_prob=0., training=False):
    """
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


# common funcs


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return tuple([x] * 2)


def add_parameter(layer, datas, name=None):
    parameter = layer.create_parameter(
        shape=(datas.shape), default_initializer=Assign(datas))
    if name:
        layer.add_parameter(name, parameter)
    return parameter


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear", )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1,
                                                                            0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = paddle.arange(
        q_size, dtype='float32')[:, None] * max(k_size / q_size, 1.0)
    k_coords = paddle.arange(
        k_size, dtype='float32')[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size /
                                                                 k_size, 1.0)

    return rel_pos_resized[relative_coords.astype('int64')]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape([B, q_h, q_w, dim])
    rel_h = paddle.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = paddle.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.reshape([B, q_h, q_w, k_h, k_w]) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).reshape([B, q_h * q_w, k_h * k_w])

    return attn


def window_partition(x, window_size, data_format="NHWC"):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), data_format=data_format)
    Hp, Wp = H + pad_h, W + pad_w

    x = x.reshape(
        [B, Hp // window_size, window_size, Wp // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape(
        [-1, window_size, window_size, C])
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    C = paddle.shape(windows)[-1]
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        [B, Hp // window_size, Wp // window_size, window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([B, Hp, Wp, C])

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x

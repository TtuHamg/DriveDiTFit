# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

# from timm.models.vision_transformer import PatchEmbed, Mlp, Attention,

# Mlp
from functools import partial
from itertools import repeat
import collections.abc

# PatchEmbed
from typing import Callable, List, Optional, Tuple, Union
from enum import Enum

#################################################################################
#                         Modulation Layers for DiT                             #
#################################################################################


class WeightModulatorLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank, use_add, fc_bias):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.use_add = use_add
        self.fc_bias = fc_bias

        if self.rank is not None:
            self.w_mul_in_dim = nn.Parameter(
                torch.ones(self.in_dim, self.rank) / math.sqrt(self.rank)
            )  # [d_in, r]
            self.w_mul_out_dim = nn.Parameter(
                torch.ones(self.out_dim, self.rank) / math.sqrt(self.rank)
            )  # [d_out, r]
            if self.use_add:
                pass
                # self.w_add_in_dim = nn.Parameter(
                #     torch.ones(self.in_dim, self.rank) * 0.0001
                # )  # [d_in, r]
                # self.w_add_out_dim = nn.Parameter(
                #     torch.zeros(self.out_dim, self.rank)
                # )  # [d_out, r]
            if self.fc_bias:
                self.b_mul = nn.Parameter(torch.ones(self.out_dim))
                if self.use_add:
                    self.b_add = nn.Parameter(torch.zeros(self.out_dim))
        else:
            self.w_mul = nn.Parameter(torch.ones(self.out_dim, self.in_dim))
            if self.use_add:
                # self.w_add = nn.Parameter(torch.zeros(self.out_dim, self.in_dim))
                pass
            if self.fc_bias:
                self.b_mul = nn.Parameter(torch.ones(self.out_dim))
                if self.use_add:
                    self.b_add = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, w, b=None):
        if self.rank is not None:
            w_mul = self.w_mul_out_dim @ self.w_mul_in_dim.transpose(
                1, 0
            )  # [d_out, d_in]
            w_hat = w * w_mul
            if self.use_add:
                # w_add = self.w_add_out_dim @ self.w_add_in_dim.transpose(
                #     1, 0
                # )  # [d_out, d_in]
                # w_hat = w_hat + w_add
                pass
            if self.fc_bias and b is not None:
                b_hat = b * self.b_mul
                if self.use_add:
                    b_hat = b_hat + self.b_add
            else:
                b_hat = None
        else:
            w_hat = w * self.w_mul
            if self.use_add:
                # w_hat = w_hat + self.w_add
                pass
            if self.bias and b is not None:
                b_hat = b * self.b_mul
                if self.use_add:
                    b_hat = b_hat + self.b_add
            else:
                b_hat = None
        return w_hat, b_hat


class WeightModulatorConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        rank,
        activation=nn.ReLU(),
        use_add=True,
        conv_bias=True,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.rank = rank
        self.activation = activation
        self.use_add = use_add
        self.conv_bias = conv_bias

        # w: [C_out, C_in, k, k]
        # b: [C_out, k, k]

        if self.rank is not None:
            self.w_mul1_out_channel_wise = nn.Parameter(
                torch.ones(self.out_channel, self.rank) / math.pow(self.rank, 2 / 3)
            )  # [C_out, r]
            self.w_mul1_instance_wise = nn.Parameter(
                torch.ones(self.rank, self.rank, self.kernel_size**2)
                / math.pow(self.rank, 2 / 3)
            )  # [r, r, k^2]
            if self.use_add:
                # self.w_add1_out_channel_bias = nn.Parameter(
                #     torch.ones(self.out_channel, self.rank) * 0.001
                # )  # [C_out, r]
                # self.w_add1_instance_bias = nn.Parameter(
                #     torch.zeros(self.kernel_size**2, self.rank)
                # )  # [k^2, r]
                pass

            self.w_mul2_in_channel_wise = nn.Parameter(
                torch.ones(self.in_channel, self.rank) / math.pow(self.rank, 2 / 3)
            )  # [C_in, r]
            if self.use_add:
                # self.w_add2_in_channel_bias = nn.Parameter(
                #     torch.ones(self.in_channel, self.rank) * 0.001
                # )  # [C_in, r]
                # self.w_add2_instance_bias = nn.Parameter(
                #     torch.zeros(self.kernel_size**2, self.rank)
                # )  # [k^2, r]
                pass

            if self.conv_bias:
                self.b_mul = nn.Parameter(torch.ones(self.out_channel))
                if self.use_add:
                    self.b_add = nn.Parameter(torch.zeros(self.out_channel))

        else:
            self.w_mul = nn.Parameter(
                torch.ones(
                    self.out_channel,
                    self.in_channel,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
            if self.use_add:
                # self.w_add = nn.Parameter(
                #     torch.zeros(
                #         self.out_channel,
                #         self.in_channel,
                #         self.kernel_size,
                #         self.kernel_size,
                #     )
                # )
                pass
            if self.conv_bias:
                self.b_mul = nn.Parameter(torch.ones(self.out_channel))
                if self.use_add:
                    self.b_add = nn.Parameter(torch.zeros(self.out_channel))

    def forward(self, w, b=None):
        if self.rank is not None:
            mul1 = (
                self.w_mul1_out_channel_wise @ self.w_mul1_instance_wise
            )  # [r, C_out, k^2]
            if self.use_add:
                # add1 = (
                #     self.w_add1_out_channel_bias
                #     @ self.w_add1_instance_bias.transpose(1, 0)
                # )  # [C_out, k^2]
                # mul1 = mul1 + add1.unsqueeze(0).repeat(
                #     self.rank, 1, 1
                # )  # [r, C_out, k^2]
                pass

            # TODO: whether use activation
            mul1 = self.activation(mul1)

            mul2 = self.w_mul2_in_channel_wise @ mul1.transpose(
                1, 0
            )  # [C_out, C_in, k^2]
            w_hat = w * mul2.view(
                self.out_channel, self.in_channel, self.kernel_size, self.kernel_size
            )  # [C_out, C_in, k, k]
            if self.use_add:
                # add2 = (
                #     self.w_add2_in_channel_bias
                #     @ self.w_add2_instance_bias.transpose(1, 0)
                # )  # [C_in, k^2]
                # w_hat = w_hat + add2.unsqueeze(0).repeat(self.out_channel, 1, 1).view(
                #     self.out_channel,
                #     self.in_channel,
                #     self.kernel_size,
                #     self.kernel_size,
                # )  # [C_out, C_in, k, k]
                pass

            if self.conv_bias and b is not None:
                b_hat = b * self.b_mul
                if self.use_add:
                    b_hat = b_hat + self.b_add
            else:
                b_hat = None
        else:
            w_hat = w * self.w_mul
            if self.use_add:
                # w_hat = w_hat + self.w_add
                pass
            if self.conv_bias and b is not None:
                b_hat = b * self.b_mul
                if self.use_add:
                    b_hat = b_hat + self.b_add
            else:
                b_hat = None

        return w_hat, b_hat


class ModulatorLinear(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        rank,
        modulation,
        use_add,
        bias,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.modulation = modulation
        self.use_add = use_add
        self.bias = bias

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if self.bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(0))

        if self.modulation:
            self.weight_modulation = WeightModulatorLinear(
                in_dim=in_dim, out_dim=out_dim, rank=rank, use_add=use_add, fc_bias=bias
            )

    def forward(self, x):
        if self.modulation:
            weight, bias = self.weight_modulation(self.weight, self.bias)
        else:
            weight, bias = self.weight, self.bias

        out = F.linear(x, weight, bias)
        return out


class ModulatorConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        rank,
        modulation,
        use_add,
        stride=1,
        padding=0,
        conv_bias=True,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rank = rank
        self.modulation = modulation
        self.use_add = use_add
        self.conv_bias = conv_bias

        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        if self.conv_bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(0))
        else:
            self.bias = None

        if self.modulation:
            self.weight_modulation = WeightModulatorConv(
                in_channel=in_channel,
                out_channel=out_channel,
                kernel_size=kernel_size,
                rank=rank,
                use_add=use_add,
                conv_bias=conv_bias,
            )

    def forward(self, x):
        if self.modulation:
            weight, bias = self.weight_modulation(self.weight, self.bias)
        else:
            weight, bias = self.weight, self.bias
        out = F.conv2d(x, weight, bias, self.stride, self.padding)
        return out


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self, hidden_size, frequency_embedding_size=256, modulation=False, rank=2
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            ModulatorLinear(
                frequency_embedding_size,
                hidden_size,
                rank=rank,
                modulation=modulation,
                use_add=True,
                bias=True,
            ),
            # nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            ModulatorLinear(
                hidden_size,
                hidden_size,
                rank=rank,
                modulation=modulation,
                use_add=True,
                bias=True,
            ),
            # nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.modulation = modulation
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob, scenario_num=0):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.scenario_num = scenario_num

        if scenario_num > 0:
            self.scenario_embedding_table = nn.Embedding(
                scenario_num + use_cfg_embedding, hidden_size
            )

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.scenario_num, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        if self.scenario_num > 0:
            use_dropout = self.dropout_prob > 0
            if (train and use_dropout) or (force_drop_ids is not None):
                labels = self.token_drop(labels, force_drop_ids)
            embeddings = self.scenario_embedding_table(labels)
            return embeddings

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                               Attention Moudle                                #
#################################################################################
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        modulation: bool = False,
        rank: int = 2,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rope: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = ModulatorLinear(
            dim, dim * 3, rank=rank, modulation=modulation, use_add=True, bias=qkv_bias
        )
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj = ModulatorLinear(
            dim, dim, rank=rank, modulation=modulation, use_add=True, bias=True
        )
        self.proj_drop = nn.Dropout(proj_drop)

        self.dim = dim

        self.sin_table_expand = nn.Parameter(
            torch.zeros(1, 1, 256, self.dim), requires_grad=False
        )
        self.cos_table_expand = nn.Parameter(
            torch.zeros(1, 1, 256, self.dim), requires_grad=False
        )
        sin_table_expand, cos_table_expand = self.init_rope(16, self.dim)

        self.sin_table_expand.data.copy_(torch.from_numpy(sin_table_expand).float())
        self.cos_table_expand.data.copy_(torch.from_numpy(cos_table_expand).float())

        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads * self.head_dim)
            .permute(2, 0, 1, 3)
            .unbind(0)
        )
        if self.rope:
            q_rotate_half = self.rotate_half(q)
            k_rotate_half = self.rotate_half(k)

            q = q * self.cos_table_expand.squeeze(
                0
            ) + q_rotate_half * self.sin_table_expand.squeeze(0)
            k = k * self.cos_table_expand.squeeze(
                0
            ) + k_rotate_half * self.sin_table_expand.squeeze(0)

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def create_sin_cos_cache(self, max_num_tokens, head_size):
        theta = 10000 ** (-np.arange(0, head_size, 2) / head_size)
        theta = theta.reshape(-1, 1).repeat(2, axis=1).flatten()

        pos = np.arange(0, max_num_tokens)
        table = pos.reshape(-1, 1) @ theta.reshape(1, -1)  # [max_num_tokens, head_size]

        sin_cache = np.sin(table)
        sin_cache[:, ::2] = -sin_cache[:, ::2]

        cos_cache = np.cos(table)
        return sin_cache, cos_cache

    def rotate_half(self, vec):
        *batch, token_length, head_dim = vec.shape
        return vec.reshape(*batch, token_length, -1, 2)[..., [1, 0]].reshape(
            *batch, token_length, head_dim
        )

    def init_rope(self, grid, head_dim):
        sin_table, cos_table = self.create_sin_cos_cache(grid, head_dim // 2)

        row_cos_table_expand = np.repeat(cos_table, grid, axis=0)
        col_cos_table_expand = np.vstack([cos_table] * grid)

        row_sin_table_expand = np.repeat(sin_table, grid, axis=0)
        col_sin_table_expand = np.vstack([sin_table] * grid)

        cos_table_expand = np.concatenate(
            [row_cos_table_expand, col_cos_table_expand], axis=-1
        )
        sin_table_expand = np.concatenate(
            [row_sin_table_expand, col_sin_table_expand], axis=-1
        )

        cos_table_expand = cos_table_expand[None, None, ...]
        sin_table_expand = sin_table_expand[None, None, ...]
        return cos_table_expand, sin_table_expand


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        modulation=False,
        rank=2,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = (
            partial(nn.Conv2d, kernel_size=1) if use_conv else ModulatorLinear
        )

        self.fc1 = linear_layer(
            in_features,
            hidden_features,
            rank=rank,
            modulation=modulation,
            use_add=True,
            bias=bias[0],
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(
            hidden_features,
            out_features,
            rank=rank,
            modulation=modulation,
            use_add=True,
            bias=bias[1],
        )
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


#################################################################################
#                              PatchEmbed Module                                #
#################################################################################


class Format(str, Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"
    NCL = "NCL"
    NLC = "NLC"


def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


try:
    from torch import _assert
except ImportError:

    def _assert(condition: bool, message: str):
        assert condition, message


class PatchEmbed(nn.Module):
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
        modulation: bool = False,
        rank: int = 2,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple(
                [s // p for s, p in zip(self.img_size, self.patch_size)]
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        # self.proj = nn.Conv2d(
        #     in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        # )
        self.proj = ModulatorConv(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            rank=rank,
            modulation=modulation,
            use_add=True,
            stride=patch_size,
            conv_bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(
                    H == self.img_size[0],
                    f"Input height ({H}) doesn't match model ({self.img_size[0]}).",
                )
                _assert(
                    W == self.img_size[1],
                    f"Input width ({W}) doesn't match model ({self.img_size[1]}).",
                )
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]}).",
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]}).",
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        modulation=False,
        block_mlp_modulation=False,
        cond_mlp_modulation=False,
        rank=2,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            modulation=modulation,
            rank=rank,
            **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            modulation=block_mlp_modulation,
            rank=rank,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ModulatorLinear(
                hidden_size,
                6 * hidden_size,
                rank=rank,
                modulation=cond_mlp_modulation,
                use_add=True,
                bias=True,
            ),  # nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, modulation, rank):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.linear = nn.Linear(
        #     hidden_size, patch_size * patch_size * out_channels, bias=True
        # )
        self.linear = ModulatorLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            rank=rank,
            modulation=modulation,
            use_add=True,
            bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ModulatorLinear(
                hidden_size,
                2 * hidden_size,
                rank=rank,
                modulation=modulation,
                use_add=True,
                bias=True,
            ),  # nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        modulation=False,
        patch_modulation=False,
        block_mlp_modulation=False,
        cond_mlp_modulation=False,
        rank=2,
        scenario_num=0,
        rope=False,
        finetune_depth=28,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.modulation = modulation
        self.rank = rank
        self.scenario_num = scenario_num

        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            bias=True,
            modulation=patch_modulation,
            rank=rank,
        )
        self.t_embedder = TimestepEmbedder(
            hidden_size, modulation=modulation, rank=rank
        )
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob, scenario_num=scenario_num
        )
        num_patches = self.x_embedder.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    modulation=modulation,
                    block_mlp_modulation=block_mlp_modulation,
                    cond_mlp_modulation=cond_mlp_modulation,
                    rank=rank,
                    rope=rope if index < finetune_depth else False,
                )
                for index in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels, modulation, rank
        )
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}

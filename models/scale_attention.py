from pathlib import Path

dir_containing_this_file = Path(__file__).resolve().parent
import sys

sys.path.insert(0, dir_containing_this_file)
import torch
from timm.models.vision_transformer import VisionTransformer, LayerScale
from timm.layers import Mlp, DropPath
from timm.models.vision_transformer import init_weights_vit_timm, get_init_weights_vit
from timm.models._manipulate import named_apply
from timm.layers import trunc_normal_
import timm
from multiscale_attn import *
import numpy as np
from functools import partial
import torch.nn as nn
from typing import Callable, List, Optional, Sequence, Tuple, Union

"""Scale Attention blocks"""


class AttentionForScale(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        # self.scale = 2 * dim ** -0.5

    def forward(self, x):
        B, num_regions, num_scales, C = x.shape  # N, 49, 86, d
        qkv = (
            self.qkv(x)
            .reshape(B, num_regions, num_scales, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 1, 4, 2, 5)
        )  # [3,B,49,6,N(86),d/6] 20314
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,49,6,N(86),d/86]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,49,6,86,86]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, num_regions, num_scales, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionForScale(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ScaleFormer(nn.Module):
    """Only scale attention blocks was included in ScaleFormer."""

    def __init__(
        self,
        depth=12,
        scales=2,
        num_heads=6,
        embed_dim=384,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=None,
        act_layer=None,
        init_values=None,
    ):
        super().__init__()
        pos_drop_rate = 0.0
        patch_drop_rate = 0.0  # patch dropout in vision transformer's implementation. Here we do not use it for scale attention.
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act_layer = act_layer or nn.GELU
        self.embed_dim = embed_dim

        self.blocks = nn.Sequential(
            *[
                ScaleBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                )
                for i in range(depth)
            ]
        )

        self.cls_token1 = nn.Parameter(
            torch.randn(1, 1, 1, embed_dim)
        )  # 1,49,1,embed_dim not good as the default
        if scales == 2:
            self.fea_dim = 6  # 6
        elif scales == 3:
            self.fea_dim = 22
        elif scales == 4:
            self.fea_dim = 86  # 2,6,22,86

        self.pos_embed_for_scale = nn.Parameter(
            torch.randn(1, 1, self.fea_dim, embed_dim)
        )  # # 1,49,self.fea_dim,embed_dim,embed_dim not good as the default
        self.pos_drop_for_scale = nn.Dropout(p=pos_drop_rate)
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed_for_scale, std=0.036)  # 0.036,0.02
        nn.init.normal_(self.cls_token1, std=0.036)  # 0.036,1e-6
        named_apply(get_init_weights_vit(mode=""), self.blocks)

    def forward(self, x):
        x = torch.cat(
            (self.cls_token1.expand(x.shape[0], 49, -1, -1), x), dim=2
        )  #  N,49,1+85,d
        x = self.pos_drop_for_scale(x + self.pos_embed_for_scale)

        for i in range(0, len(self.blocks)):
            x = self.blocks[i](x)
        cls_token = x[:, :, 1, :]

        return cls_token


"""Patch Attention blocks"""


class AttentionForPatch(Attention):
    def __init__(self, dim=768, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(self, x, cls_token=None, pos_embed=None, pos_drop=None):
        B = x.shape[0]  # N, 49, 86, d
        if len(x.size()) > 3:
            scaled_x = x[
                :, :, 0, :
            ]  # after scale attns, use the first token after scale attention as patch token
        else:
            scaled_x = x

        if cls_token is not None:
            scaled_x = torch.cat((cls_token.squeeze(1), scaled_x), dim=1)  # N,CLS+49,d

        if pos_embed is not None:
            scaled_x = pos_drop(scaled_x + pos_embed)  # N,CLS+49,d

        qkv1 = (
            self.qkv(scaled_x)
            .reshape(B, 50, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]  #  N, N_H, 50, d/N_H
        q1, k1 = self.q_norm(q1), self.k_norm(k1)  # identity in default

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        scaled_x = (attn1 @ v1).transpose(1, 2).reshape(B, 50, -1)
        scaled_x = self.proj(scaled_x)
        scaled_x = self.proj_drop(scaled_x)  # N,50,d

        return scaled_x


class PatchBlock(nn.Module):
    """Remove all Layer Scale, MLP projection and residual connection here."""

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        proj_drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.attn = AttentionForPatch(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, cls_token=None, pos_embed=None, pos_drop=None):
        x = self.attn(x, cls_token=cls_token, pos_embed=pos_embed, pos_drop=pos_drop)
        return x


class MultiscaleFormer(nn.Module):
    """In MultiscaleFormer class we include scale then patch attention blocks.
    Not inheritage from vision transformer but from scratch ."""

    def __init__(
        self,
        depth: int = 12,
        scales: int = 2,
        num_heads: int = 12,
        embed_dim: int = 768,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        init_values: Optional[float] = None,
        num_classes: int = 100,
        num_patches: int = 49,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        block_fn: Callable = ScaleBlock,
        block_fn1: Callable = PatchBlock,
    ):
        super().__init__()
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act_layer = act_layer or nn.GELU
        self.embed_dim = embed_dim

        self.scaleBlocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                )
                for i in range(depth)
            ]
        )
        self.blocks = nn.Sequential(
            *[
                block_fn1(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for i in range(depth)
            ]
        )

        if scales == 2:
            self.fea_dim = 21  # 6
        elif scales == 3:
            self.fea_dim = 22  # 22,82,70,85
        elif scales == 4:
            self.fea_dim = 86  # 2,6,22,86

        embed_len = num_patches + 1  # with cls_token
        self.pos_embed_for_scale = nn.Parameter(
            torch.randn(1, 1, self.fea_dim, embed_dim)
        )  # # 1,49,self.fea_dim,embed_dim,embed_dim not good as the default
        self.pos_drop_for_scale = nn.Dropout(p=pos_drop_rate)

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )  # for patch attentions
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.fc_norm = self.norm_layer(embed_dim)  # classification head
        self.head_drop = nn.Dropout(0.0)
        self.head = nn.Linear(self.embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed_for_scale, std=0.036)  # 0.036, default: 0.02
        trunc_normal_(self.pos_embed, std=0.036)  # 0.036, default: 0.02
        nn.init.normal_(self.cls_token, std=0.036)  # 0.036, default: 1e-6
        named_apply(get_init_weights_vit(mode=""), self.blocks)
        named_apply(get_init_weights_vit(mode=""), self.scaleBlocks)

    def forward(self, x):
        x = self.pos_drop_for_scale(x + self.pos_embed_for_scale)
        for i in range(0, len(self.blocks)):  # scale attentions
            x = self.scaleBlocks[i](x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1, -1)
        for i in range(0, len(self.blocks)):  # patch attentions
            if i == 0:
                x = self.blocks[i](x, cls_token, self.pos_embed, self.pos_drop)
            else:
                x = self.blocks[i](x)
        cls_token = x[:, 0, :]
        x = self.fc_norm(cls_token)
        cls_token = self.head_drop(cls_token)
        return self.head(cls_token)

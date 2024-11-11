import torch
from timm.models.vision_transformer import VisionTransformer,LayerScale
from timm.layers import Mlp, DropPath
from timm.models.vision_transformer import init_weights_vit_timm, get_init_weights_vit
from timm.models._manipulate import named_apply
from timm.layers import trunc_normal_
import timm
#from .multiscale_attn import *
from multiscale_attn import *
import numpy as np
from functools import partial
import torch.nn as nn

'''Scale Attention blocks'''
class AttentionForScale(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0,):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        # self.scale = 2 * dim ** -0.5 

    def forward(self, x, return_attention=False):
        B, num_regions, num_scales, C = x.shape # N, 49, 86, d
        qkv = self.qkv(x).reshape(B, num_regions,num_scales, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5) # [3,B,49,6,N(86),d/6] 20314
        q, k, v = qkv[0], qkv[1], qkv[2]   # [B,49,6,N(86),d/86]

        attn = (q @ k.transpose(-2, -1)) * self.scale # [B,49,6,86,86]
        attn_before_softmax = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(2, 3).reshape(B, num_regions,num_scales, C) 
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return x, attn, attn_before_softmax
        else:
            return x 

class ScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)        
        self.attn = AttentionForScale(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,x,return_attention=False):
        if return_attention:
            x_after_scale_attn, attn, attn_beforeSoftmax = self.drop_path1(self.ls1(self.attn(self.norm1(x),return_attention=True)))
            x = x + x_after_scale_attn
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x, attn, attn_beforeSoftmax
        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x

class ScaleFormer(nn.Module):
    '''Only scale attention blocks was included in ScaleFormer.'''
    def __init__(self,depth=12,scales=2,num_heads=6, embed_dim=384, mlp_ratio=4.,qkv_bias=True,qk_norm=False,proj_drop_rate=0.,
        attn_drop_rate=0.,norm_layer=None, act_layer=None, init_values = None ): 
        super().__init__()
        pos_drop_rate = 0.
        patch_drop_rate = 0. # patch dropout in vision transformer's implementation. Here we do not use it for scale attention.
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act_layer = act_layer or nn.GELU
        self.embed_dim = embed_dim

        self.blocks = nn.Sequential(*[
            ScaleBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,qk_norm=qk_norm, init_values=init_values,proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate, norm_layer=self.norm_layer, act_layer=self.act_layer)
            for i in range(depth)])

        self.cls_token1 = nn.Parameter(torch.randn(1, 1, 1, embed_dim)) # 1,49,1,embed_dim not good as the default
        if scales == 2:
            self.fea_dim = 6 # 6
        elif scales == 3:
            self.fea_dim = 22
        elif scales == 4:
            self.fea_dim = 86 # 2,6,22,86

        self.pos_embed_for_scale = nn.Parameter(torch.randn(1, 1, self.fea_dim, embed_dim)) # # 1,49,self.fea_dim,embed_dim,embed_dim not good as the default
        self.pos_drop_for_scale = nn.Dropout(p=pos_drop_rate)
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed_for_scale, std=0.036) # 0.036,0.02
        nn.init.normal_(self.cls_token1, std=0.036) # 0.036,1e-6
        named_apply(get_init_weights_vit(mode=''), self.blocks)

    def forward(self,x):
        x = torch.cat((self.cls_token1.expand(x.shape[0], 49, -1, -1), x), dim=2) #  N,49,1+85,d     
        x = self.pos_drop_for_scale(x + self.pos_embed_for_scale) 

        for i in range(0, len(self.blocks)):
            x = self.blocks[i](x)
        cls_token = x[:,:,1,:]
        
        return cls_token



'''Patch Attention blocks'''
class AttentionForPatch(Attention):
    def __init__(self, dim=768, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0, scale_token='random'):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.scale_token=scale_token

    def forward(self, x,cls_token=None,pos_embed=None,pos_drop=None,return_attention=False):
        B = x.shape[0] # N, 49, 86, d
        if x.dim() > 3:
            if self.scale_token=='channel' or self.scale_token=='random':
                # print(self.scale_token)
                scaled_x = x[:,:,0,:] # after scale attns, use the first token after scale attention as patch token
            elif self.scale_token == 'first':
                # print('first')
                scaled_x = x[:,:,0,:] # no scale token, use the first one
            else:
                # print('avergae')
                scaled_x = torch.mean(x, dim=-2) # no scale token, use the avg 
        else:
            scaled_x = x # from the 2nd block of scaleformer and for hybrid vit case

        if cls_token is not None:
            scaled_x = torch.cat((cls_token.squeeze(1), scaled_x), dim=1) # N,CLS+49,d

        if pos_embed is not None:
            scaled_x = pos_drop(scaled_x + pos_embed) # N,CLS+49,d

        qkv1 = self.qkv(scaled_x).reshape(B, 50, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2] #  N, N_H, 50, d/N_H
        q1, k1 = self.q_norm(q1), self.k_norm(k1) # identity in default

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1) 
        attn1 = self.attn_drop(attn1)
        
        scaled_x = (attn1 @ v1).transpose(1, 2).reshape(B, 50, -1)
        scaled_x = self.proj(scaled_x)
        scaled_x = self.proj_drop(scaled_x) # N,50,d

        if return_attention:
            return scaled_x, attn1
        else:
            return scaled_x


class PatchBlock(nn.Module):
    '''Remove all Layer Scale, MLP projection and residual connection here.'''
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        proj_drop=0.,
        attn_drop=0.,
        scale_token='random'
    ):
        super().__init__()    
        self.attn = AttentionForPatch(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,scale_token=scale_token)

    def forward(self,x,cls_token=None,pos_embed=None,pos_drop=None,return_attention=False):
        return self.attn(x,cls_token=cls_token,pos_embed=pos_embed,pos_drop=pos_drop,return_attention=return_attention) 

class MultiscaleFormer(nn.Module):
    '''In MultiscaleFormer class we include scale then patch attention blocks. 
        Not inheritage from vision transformer but from scratch .'''

    def __init__(self,depth=12,scales=2,num_heads=12, embed_dim=786, mlp_ratio=4.,qkv_bias=True,qk_norm=False,proj_drop_rate=0.,
        attn_drop_rate=0.,norm_layer=None, act_layer=None, init_values = None, num_classes = 100, num_patches = 49, scale_token='random',patch_attn=True): 
        super().__init__()
        pos_drop_rate = 0.
        patch_drop_rate = 0. # patch dropout in vision transformer's implementation. 
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act_layer = act_layer or nn.GELU
        self.embed_dim = embed_dim
        self.scale_token = scale_token
        self.patch_attn=patch_attn
        embed_len = num_patches + 1 # with cls_token

        self.scaleBlocks = nn.Sequential(*[
            ScaleBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,qk_norm=qk_norm, init_values=init_values,proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate, norm_layer=self.norm_layer, act_layer=self.act_layer)
            for i in range(depth)])
        if self.patch_attn:
            self.blocks = nn.Sequential(*[
                PatchBlock(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,proj_drop=proj_drop_rate,attn_drop=attn_drop_rate, scale_token=self.scale_token)
                for i in range(depth)])
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # for patch attentions
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
            self.pos_drop = nn.Dropout(p=pos_drop_rate)
        else:
            self.fc = nn.Linear(num_patches,1) # for abalation on scale attn only

        if self.scale_token=='channel' or self.scale_token=='random':
            if scales == 2:
                self.fea_dim = 18 # 6
            elif scales == 3:
                self.fea_dim = 22
            elif scales == 4:
                self.fea_dim = 86 # 2,6,22,86
        else:
            if scales == 2:
                self.fea_dim = 17 # 6
            elif scales == 3:
                self.fea_dim = 21
            elif scales == 4:
                self.fea_dim = 85 # 2,6,22,86

        
        self.pos_embed_for_scale = nn.Parameter(torch.randn(1, 1, self.fea_dim, embed_dim)) # # 1,49,self.fea_dim,embed_dim,embed_dim not good as the default
        self.pos_drop_for_scale = nn.Dropout(p=pos_drop_rate)

        

        # self.fc_norm = norm_layer(embed_dim)  # classification head
        self.head_drop = nn.Dropout(0.)
        self.head = nn.Linear(self.embed_dim, num_classes) 
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed_for_scale, std=0.036) # 0.036, default: 0.02
        if self.patch_attn:
            trunc_normal_(self.pos_embed, std=0.036) # 0.036, default: 0.02
            nn.init.normal_(self.cls_token, std=0.036) # 0.036, default: 1e-6
            named_apply(get_init_weights_vit(mode=''), self.blocks)
        named_apply(get_init_weights_vit(mode=''), self.scaleBlocks)

    def forward(self,x,return_scale_attention=False):
        x = self.pos_drop_for_scale(x + self.pos_embed_for_scale) 
        for i in range(0, len(self.scaleBlocks)): # scale attentions
            if return_scale_attention:
                if i == len(self.scaleBlocks)-1: # return only the final block
                    x, attn_for_scale, attn_for_scale_beforeSoftmax = self.scaleBlocks[i](x,True)
                    # return x, attn_for_scale, attn_for_scale_beforeSoftmax
                else:
                    x = self.scaleBlocks[i](x)
            else:
                x = self.scaleBlocks[i](x)

        if self.patch_attn == True:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1, -1) 
            for i in range(0, len(self.blocks)): # patch attentions
                if i == 0:
                    x = self.blocks[i](x,cls_token,self.pos_embed,self.pos_drop)
                elif i == len(self.blocks)-1: # return only the final block
                    if return_scale_attention:
                        x, attn_for_patch = self.blocks[i](x,return_attention=True)
                    else:
                        x = self.blocks[i](x)
                else:
                    x = self.blocks[i](x)
                
            cls_token = x[:,0,:]
        else:
            scale_token = torch.mean(x,dim=-2) # x[:,:,0,:]
            cls_token = self.fc(scale_token.transpose(1,2)).squeeze(-1)
        # x = self.fc_norm(cls_token)
        cls_token = self.head_drop(cls_token)

        if return_scale_attention:
            return self.head(cls_token), attn_for_scale, attn_for_scale_beforeSoftmax, attn_for_patch
        else:
            return self.head(cls_token)


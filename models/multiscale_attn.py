from torch.nn import functional as F
import math
import torch
from timm.models.vision_transformer import Attention,Block,LayerScale
import numpy as np
from functools import partial
import torch.nn as nn
from timm.layers import PatchEmbed, Mlp, DropPath

# version with LayerScale
# class MultiScaleAttention(Attention):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
#         super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)
#         # self.num_heads = num_heads
#         # head_dim = dim // num_heads
#         # self.scale = head_dim ** -0.5 # sqrt(dk)/2
#         self.scale = dim ** -0.5 
#         self.qkv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop1 = nn.Dropout(attn_drop)
#         self.proj1 = nn.Linear(dim, dim)
#         self.proj_drop1 = nn.Dropout(proj_drop)

#     def forward_with_scale(self, x):
#         B, num_regions, num_scales, C = x.shape # N, 49, 86, d
#         qkv = self.qkv1(x).reshape(B, num_regions,num_scales, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5) # [3,B,49,6,N(86),d/6] 20314
#         q, k, v = qkv[0], qkv[1], qkv[2]   # [B,49,6,N(86),d/86]

#         attn = (q @ k.transpose(-2, -1)) * self.scale # [B,49,6,86,86]
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop1(attn)

#         x = (attn @ v).transpose(2, 3).reshape(B, num_regions,num_scales, C) 
#         x = self.proj1(x)
#         x = self.proj_drop1(x)

#         return x 

#     def forward_with_region(self, x,cls_token,pos_embed=None,pos_drop=None):
#         B, num_regions, num_scales, C = x.shape # N, 49, 86, d
#         scaled_x = x[:,:,0,:] # use the first token after scale attention as patch token
#         scaled_x = torch.cat((cls_token.squeeze(1), scaled_x), dim=1) # N,CLS+49,d
#         # Pos_emb from pretrained ViT
#         if pos_embed is not None:
#             scaled_x = pos_drop(scaled_x + pos_embed) # N,CLS+49,d

#         qkv1 = self.qkv(scaled_x).reshape(B, num_regions+1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2] #  N, N_H, 50, d/N_H

#         attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
#         attn1 = attn1.softmax(dim=-1) # all attn the same value?
#         attn1 = self.attn_drop(attn1)

#         scaled_x = (attn1 @ v1).transpose(1, 2).reshape(B, num_regions+1, -1)
#         scaled_x = self.proj(scaled_x)
#         scaled_x = self.proj_drop(scaled_x) # N,50,d
#         cls_token = scaled_x[:,:1,:].unsqueeze(1)
#         x = torch.cat((scaled_x[:,1:,:].unsqueeze(2), x[:,:,1:,:]), dim=2)

#         return x, cls_token # the CLS token should carry information of different regions

# class MultiscaleBlock(Block):
#     def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_norm=False,init_values=None,proj_drop=0, attn_drop=0, drop_path=0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,):
#         super().__init__(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,qk_norm=qk_norm,init_values=init_values, proj_drop=proj_drop, 
#                          attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
#         # self.norm1 = norm_layer(dim)
#         self.attn = MultiScaleAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
#         self.attnOri = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        
#         '''WI layer scale & layer norm, mlp, attn drop.'''
#         # self.norm1_for_region = norm_layer(dim)
#         # self.ls1_for_region = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         # self.drop_path1_for_region = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         # self.norm2_for_region = norm_layer(dim)
#         # self.mlp_for_region = Mlp(
#         #     in_features=dim,
#         #     hidden_features=int(dim * mlp_ratio),
#         #     act_layer=act_layer,
#         #     drop=proj_drop,
#         # )
#         # self.ls2_for_region = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         # self.drop_path2_for_region = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward_new_block1(self, x,cls_token,pos_embed,pos_drop):
#         '''WI layer scale & layer norm, mlp, attn drop.'''
#         # x=x + self.drop_path1(self.ls1(self.attn.forward_with_scale(self.norm1(x))))
#         # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
#         # x = self.norm1_for_region(x)
#         # x, cls_token = self.attn.forward_with_region(x,cls_token,pos_embed,pos_drop)
#         # x = x + self.drop_path1_for_region(self.ls1_for_region(x))
#         # cls_token = cls_token + self.drop_path1_for_region(self.ls1_for_region(cls_token))
#         # x = x + self.drop_path2_for_region(self.ls2_for_region(self.mlp_for_region(self.norm2_for_region(x))))
#         # cls_token = cls_token + self.drop_path2_for_region(self.ls2_for_region(self.mlp_for_region(self.norm2_for_region(cls_token))))

#         '''WO layer scale or layer norm, mlp, attn drop.'''
#         x = x + self.drop_path1(self.ls1(self.attn.forward_with_scale(self.norm1(x))))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         x, cls_token = self.attn.forward_with_region(x,cls_token,pos_embed,pos_drop)
#         # x = x + self.drop_path1(x)
#         return x, cls_token

#     def forward_new(self, x,cls_token):
#         '''WI layer scale & layer norm, mlp, attn drop.'''
#         # x=x + self.drop_path1(self.ls1(self.attn.forward_with_scale(self.norm1(x))))
#         # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

#         # x = self.norm1_for_region(x)
#         # x, cls_token = self.attn.forward_with_region(x,cls_token)

#         # x = x + self.drop_path1_for_region(self.ls1_for_region(x))
#         # cls_token = cls_token + self.drop_path1_for_region(self.ls1_for_region(cls_token))
#         # x = x + self.drop_path2_for_region(self.ls2_for_region(self.mlp_for_region(self.norm2_for_region(x))))
#         # cls_token = cls_token + self.drop_path2_for_region(self.ls2_for_region(self.mlp_for_region(self.norm2_for_region(cls_token))))
        
#         '''WO layer scale or layer norm, mlp, attn drop.'''
#         x = x + self.drop_path1(self.ls1(self.attn.forward_with_scale(self.norm1(x))))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         x, cls_token = self.attn.forward_with_region(x,cls_token)
#         return x, cls_token

#     def forward(self, x):
#         if len(x.shape) > 3:
#             x = x.view(x.shape[0],x.shape[1]*x.shape[2],-1)
#         x = x + self.drop_path1(self.ls1(self.attnOri(self.norm1(x))))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x

class MultiScaleAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        # self.num_heads = num_heads
        # head_dim = dim // num_heads
        # self.scale = head_dim ** -0.5 # default
        # self.scale = 2* self.scale # 
        self.scale = 2 * dim ** -0.5 
        # self.scale =  (head_dim ** -0.5 )/2  # sqrt(dk)/2,0.76
        self.qkv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop1 = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)

    def forward_with_scale(self, x):
        B, num_regions, num_scales, C = x.shape # N, 49, 86, d
        qkv = self.qkv1(x).reshape(B, num_regions,num_scales, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5) # [3,B,49,6,N(86),d/6] 20314
        q, k, v = qkv[0], qkv[1], qkv[2]   # [B,49,6,N(86),d/6]

        attn = (q @ k.transpose(-2, -1)) * self.scale # [B,49,6,86,86]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop1(attn)
        

        x = (attn @ v).transpose(2, 3).reshape(B, num_regions,num_scales, C) 
        x = self.proj1(x)
        x = self.proj_drop1(x)

        return x 

    # def forward_with_region(self, x,cls_token,pos_embed=None,pos_drop=None):
    #     B, num_regions, num_scales, C = x.shape # N, 49, 86, d
    #     scaled_x = x[:,:,0,:] # use the first token after scale attention as patch token
    #     scaled_x = torch.cat((cls_token.squeeze(1), scaled_x), dim=1) # N,CLS+49,d
    #     if pos_embed is not None:
    #         scaled_x = pos_drop(scaled_x + pos_embed) # N,CLS+49,d

    #     qkv1 = self.qkv(scaled_x).reshape(B, num_regions+1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    #     q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2] #  N, N_H, 50, d/N_H

    #     attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
    #     attn1 = attn1.softmax(dim=-1) # all attn the same value?
    #     attn1 = self.attn_drop(attn1)

    #     scaled_x = (attn1 @ v1).transpose(1, 2).reshape(B, num_regions+1, -1)
    #     scaled_x = self.proj(scaled_x)
    #     scaled_x = self.proj_drop(scaled_x) # N,50,d
    #     cls_token = scaled_x[:,:1,:].unsqueeze(1)
    #     x = torch.cat((scaled_x[:,1:,:].unsqueeze(2), x[:,:,1:,:]), dim=2)

    #     return x, cls_token # the CLS token should carry information of different regions
    
    def forward_with_region(self, x,cls_token=None,pos_embed=None,pos_drop=None):
        B = x.shape[0] # N, 49, 86, d
        if len(x.size()) > 3:
            scaled_x = x[:,:,0,:] # use the first token after scale attention as patch token
            # scaled_x = torch.mean(x, dim=2) 
        else:
            scaled_x = x
        if cls_token is not None:
            scaled_x = torch.cat((cls_token.squeeze(1), scaled_x), dim=1) # N,CLS+49,d
        # Pos_emb from pretrained ViT
        if pos_embed is not None:
            scaled_x = pos_drop(scaled_x + pos_embed) # N,CLS+49,d

        qkv1 = self.qkv(scaled_x).reshape(B, 50, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2] #  N, N_H, 50, d/N_H

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1) # all attn the same value?
        attn1 = self.attn_drop(attn1)

        scaled_x = (attn1 @ v1).transpose(1, 2).reshape(B, 50, -1)
        scaled_x = self.proj(scaled_x)
        scaled_x = self.proj_drop(scaled_x) # N,50,d
        # cls_token = scaled_x[:,:1,:].unsqueeze(1)
        # x = torch.cat((scaled_x[:,1:,:].unsqueeze(2), x[:,:,1:,:]), dim=2)
        return scaled_x
        
class MultiscaleBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_norm=False,init_values=None,proj_drop=0, attn_drop=0, drop_path=0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,):
        super().__init__(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,qk_norm=qk_norm,init_values=init_values, proj_drop=proj_drop, 
                         attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        # self.norm1 = norm_layer(dim)
        self.attn = MultiScaleAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        # self.attnOri = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

    '''for mixed order, i.e., [scale - region]->[scale - region->...->[scale - region]'''
    # def forward_new_block1(self, x,cls_token,pos_embed,pos_drop):
    #     x=x + self.drop_path1(self.ls1(self.attn.forward_with_scale(self.norm1(x))))
    #     x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    #     x, cls_token = self.attn.forward_with_region(x,cls_token,pos_embed,pos_drop)
    #     # x = x + self.drop_path1(x)

    #     return x, cls_token

    # def forward_new(self, x,cls_token):
    #     x=x + self.drop_path1(self.ls1(self.attn.forward_with_scale(self.norm1(x))))
    #     x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    #     x, cls_token = self.attn.forward_with_region(x,cls_token)
    #     # x = x + self.drop_path1(x)
    #     # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    #     # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    #     return x, cls_token
    
    '''for separate order, i.e., scale->...->scale(12 blocks) - region->...->region(12 blocks)]'''
    def forward_change_order_attn1(self,x):
        x=x + self.drop_path1(self.ls1(self.attn.forward_with_scale(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    def forward_change_order_attn2_block1(self,x,cls_token,pos_embed,pos_drop):
        x = self.attn.forward_with_region(x,cls_token,pos_embed,pos_drop)
        return x

    def forward_change_order_attn2(self,x):
        x = self.attn.forward_with_region(x)
        cls_token = x[:,0,:]
        # cls_token = torch.mean(x, dim=1)
        return cls_token

    '''for vanilla vit'''
    def forward(self, x):
        if len(x.shape) > 3:
            x = x.view(x.shape[0],x.shape[1]*x.shape[2],-1)
        x = x + self.drop_path1(self.ls1(self.attnOri(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
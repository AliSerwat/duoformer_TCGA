import torch
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import init_weights_vit_timm, get_init_weights_vit
from timm.models._manipulate import named_apply
from timm.layers import trunc_normal_
import timm
from .multiscale_attn import *
import numpy as np
from functools import partial
import torch.nn as nn

class MultiscaleTransformer(VisionTransformer):
    def __init__(self, pretrained=False,depth=12,scales=2,num_heads=6, patch_size=16, embed_dim=384, mlp_ratio=4.,qkv_bias=True,qk_norm=False,drop_rate=0., drop_path_rate=0.,
        attn_drop_rate=0.,norm_layer=None, act_layer=None, init_values = 1e-5,num_classes=1000,model_type = 'scaleformer'):
        super().__init__(depth=depth,patch_size=patch_size,num_classes=num_classes,embed_dim=embed_dim,num_heads=num_heads) # class_token=False,global_pool=''
        pos_drop_rate = 0.
        patch_drop_rate = 0. # patch dropout in vision transformer's implementation. Here we do not use it for scale attention.

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act_layer = act_layer or nn.GELU
        self.embed_dim = embed_dim
        self.model = model_type
        self.blocks = nn.Sequential(*[
            MultiscaleBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,qk_norm=qk_norm, init_values=init_values,proj_drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=self.dpr[i], norm_layer=self.norm_layer, act_layer=self.act_layer)
            for i in range(depth)])

        self.num_patches = self.patch_embed.num_patches
        # with patch token
        # self.cls_token1 = nn.Parameter(torch.randn(1, 1, 1, embed_dim)) 
        if scales == 2:
            self.fea_dim = 6 # 6
        elif scales == 3:
            self.fea_dim = 22
        elif scales == 4:
            self.fea_dim = 86 # 2,6,22,86
        # without patch token
        self.cls_token1 = None
        # self.fea_dim = 5 # 1,5,21,85
        self.pos_embed_for_scale = nn.Parameter(torch.randn(1, 1, self.fea_dim, embed_dim))
        self.pos_drop_for_scale = nn.Dropout(p=pos_drop_rate)
        self._init_weights()

        # if pretrained,load half of the attention weights
        # if pretrained:
            # vanilla_model = torchvision.models.vit_base_patch16_224_in21k()
            # https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/R50%2BViT-B_16.npz;tab=live_object
            # https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz
            # vanilla_model= timm.models.vision_transformer.vit_base_patch32_224(pretrained=True,num_classes = num_classes)
            # pretrained_state_dict = vanilla_model.state_dict()
        # for i in range(12):
        #     self.blocks[i].load_state_dict(vanilla_model.blocks[i].state_dict())

    def _init_weights(self):
        super()._init_weights(self.pos_embed_for_scale) # give a dummy input to the parent function in timm 
        # my additional modifications here
        trunc_normal_(self.pos_embed_for_scale, std=0.036) # 0.036,0.02
        if self.cls_token1 is not None:
            nn.init.normal_(self.cls_token1, std=0.036) # 0.036,1e-6
        named_apply(get_init_weights_vit(mode=''), self.blocks)

    def forward(self,x):
        # x = self.patch_embed(x) # [N, 14*14, 384]
        # if len(x) == 1:
        #     B,D,H,W = x['3'].shape
        #     if (H,W) == (7,7): # only layer 4 used
        #         x = x['3'].reshape(B, D,49).permute(0,2,1) # flatten 7x7 to 49 
        #         cls_token = self.cls_token.expand(B, -1, -1)
        #         x = torch.cat((cls_token,x),dim=1)
        #         x = self.pos_drop(x+self.pos_embed) # for a single layer of feature
        # elif len(x) == 2:
        #     B = x['2'].shape[0]
        #     if self.model == 'vit':
        #         x['2'] = x['2'].reshape(B,self.embed_dim ,-1)
        #         x['3'] = x['3'].reshape(B,self.embed_dim ,-1)
        #         x = torch.cat((x['3'],x['2']),dim=2).permute(0,2,1) 
        #         cls_token = self.cls_token.expand(B, -1, -1)
        #         x = torch.cat((cls_token,x),dim=1)
        #         x = self.pos_drop(x+self.pos_embed1)

        if self.model == 'scaleformer':    # features already processed, [B, 49, 1/5/21/85, d]
            B = x.shape[0]
            cls_token = self.cls_token.expand(B, -1, -1, -1)  # Pretrained CLS token: 1,1,d -> N,1,1,d
            # if with patch token
            # x = torch.cat((self.cls_token1.expand(B, self.num_patches, -1, -1), x), dim=2) #  N,49,1+85,d    
            x = self.pos_drop_for_scale(x + self.pos_embed_for_scale) # [N, num_patches+cls_token ,channels] N,196+1,d
            # if without patch token     
            # x = self.pos_drop(x + self.pos_embed_for_scale) # [N, 49 , 85, channels] 
            for i in range(0, len(self.blocks)):
                x = self.blocks[i].forward_change_order_attn1(x)
            # cls_token = x[:,:,0,:]
            # return cls_token
            for i in range(0, len(self.blocks)):
                if i == 0:
                    # x,cls_token = self.blocks[i].forward_new_block1(x,cls_token,self.pos_embed,self.pos_drop)
                    x = self.blocks[i].forward_change_order_attn2_block1(x,cls_token,self.pos_embed,self.pos_drop) 
                else:
                    cls_token = self.blocks[i].forward_change_order_attn2(x)
                    # x,cls_token = self.blocks[i].forward_new(x,cls_token)
            
            cls_token = self.norm(cls_token)

        # if self.model == 'vit':
        #     for i in range(0, len(self.blocks)):
        #         x = self.blocks[i](x)
        #     x = self.norm(x)
        #     cls_token = x[:, 0]

        return self.head(cls_token).squeeze()  # Assuming CLS carries information of patches infusing multiscale features.
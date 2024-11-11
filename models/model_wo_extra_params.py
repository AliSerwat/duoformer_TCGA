import torch
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.layers import Mlp, DropPath
from timm.models.resnetv2 import ResNetV2
from torch import nn
import torchvision.models as models
#from torchvision.models import ResNet50_Weights,ResNet18_Weights
# from .multi_vision_transformer import *
# from .multiscale_attn import *
# from .projection_head import *
# from .scale_attention import *
# from .backbone import *
from multi_vision_transformer import *
from multiscale_attn import *
from projection_head import *
from scale_attention import *
from backbone import *
from resnet50ssl import *

class MyModel_no_extra_params(nn.Module):
    def __init__(self, depth=None, embed_dim=768,num_heads=12,init_values = 1e-5,num_classes=2,num_layers = 4, num_patches = 49,mlp_ratio=4.,attn_drop_rate=0.,
                proj_drop_rate = 0., proj_dim = 768,freeze_backbone=True,backbone = 'r50',scale_token='random',patch_attn=True):
        super().__init__()
        self.num_layers = num_layers
        self.proj_dim = proj_dim
        self.backbone=backbone
        self.scale_token = scale_token
        self.patch_attn=patch_attn
        if backbone == 'r50':
            #self.resnet_projector = nn.Sequential(*list(models.resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-2])
            self.resnet_projector = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
            print("Resnet 50 pretrained weights loaded!")
        elif backbone == 'r18':
            #self.resnet_projector = nn.Sequential(*list(models.resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-2])
            self.resnet_projector = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
            print("Resnet 18 pretrained weights loaded!")
        elif backbone =='r50_Swav':
            self.resnet_projector = resnet50FeatureExtractor(pretrained=True, progress=False, key="SwAV") # MoCoV2, BT ,removed the fc 
            print(f'Weights of {backbone} pretrained on TCGA+TULIP loaded!')
        elif backbone =='r50_BT':
            self.resnet_projector = resnet50FeatureExtractor(pretrained=True, progress=False, key="BT") # MoCoV2, BT ,removed the fc 
            print(f'Weights of {backbone} pretrained on TCGA+TULIP loaded!')
        elif backbone =='r50_MoCoV2':
            self.resnet_projector = resnet50FeatureExtractor(pretrained=True, progress=False, key="MoCoV2") # MoCoV2, BT ,removed the fc 
            print(f'Weights of {backbone} pretrained on TCGA+TULIP loaded!')
        

        if freeze_backbone:        # freeze the backbone in training or not
            for param in self.resnet_projector.parameters():
                param.requires_grad = False
            print("Backbone freezed during training!")

        if self.scale_token == 'random':
            self.channel_token = torch.nn.Parameter(torch.randn(1, 1, 1, self.proj_dim))
            nn.init.normal_(self.channel_token, std=0.036) # 0.036,1e-6
            print('Scale token initialized from random.')
        elif self.scale_token == 'channel': # 0.036,1e-6
            self.chann_proj1 = Channel_Projector_layer1(backbone=backbone)
            self.chann_proj2 = Channel_Projector_layer2(backbone=backbone)
            self.chann_proj3 = Channel_Projector_layer3()
            self.chann_proj_all = Channel_Projector_All(backbone=backbone)
            print('Scale token learned from channel.')

        self.projection = Projection(num_layers=self.num_layers, proj_dim=self.proj_dim, backbone = backbone )

        self.vision_transformer = MultiscaleFormer(depth=depth,scales=self.num_layers,num_heads=num_heads, embed_dim=embed_dim, 
                        mlp_ratio=mlp_ratio, qkv_bias=True, qk_norm=False, proj_drop_rate=proj_drop_rate,
                        attn_drop_rate=attn_drop_rate, norm_layer=None, act_layer=None, init_values = None, 
                        num_classes = num_classes, num_patches = num_patches, scale_token = scale_token,patch_attn=patch_attn)
        print("Multiscaletransformer implemented from scratch!")

        

        # self.scale_former = ScaleFormer(depth=12,scales=self.num_layers,num_heads=12,embed_dim=self.proj_dim)   # only scale attentions, consistent with pretrained hybrid

        self.index = {}
        for i in range(4):
            self.index[f'{4-i-1}']=torch.empty([49,4**i],dtype=torch.int64)

        for r in range(7):
            for c in range(7):
                p = r*7+c
                self.index['3'][p,:] = p
                self.index['2'][p,:] = torch.IntTensor(
                    [2*r*14+2*c, (2*r+1)*14+2*c, 2*r*14+(2*c+1),(2*r+1)*14+(2*c+1)])

                self.index['1'][p,:] = torch.IntTensor([4*r*28+4*c, 4*r*28+4*c+1,4*r*28+4*c+2,4*r*28+4*c+3,
                (4*r+1)*28+4*c, (4*r+1)*28+4*c+1, (4*r+1)*28+4*c+2, (4*r+1)*28+4*c+3,
                (4*r+2)*28+4*c, (4*r+2)*28+4*c+1, (4*r+2)*28+4*c+2, (4*r+2)*28+4*c+3,
                (4*r+3)*28+4*c, (4*r+3)*28+4*c+1, (4*r+3)*28+4*c+2, (4*r+3)*28+4*c+3])

                self.index['0'][p, :] = torch.IntTensor(
                [8*r*56+8*c, 8*r*56+8*c+1, 8*r*56+8*c+2,8*r*56+8*c+3,8*r*56+8*c+4,8*r*56+8*c+5,
                8*r*56+8*c+6, 8*r*56+8*c+7,
                (8*r+1)*56+8*c, (8*r+1)*56+8*c+1, (8*r+1)*56+8*c+2,(8*r+1)*56+8*c+3,(8*r+1)*56+8*c+4,
                (8*r+1)*56+8*c+5, (8*r+1)*56+8*c+6, (8*r+1)*56+8*c+7,
                (8*r+2)*56+8*c, (8*r+2)*56+8*c+1, (8*r+2)*56+8*c+2,(8*r+2)*56+8*c+3,(8*r+2)*56+8*c+4,
                (8*r+2)*56+8*c+5, (8*r+2)*56+8*c+6, (8*r+2)*56+8*c+7,
                (8*r+3)*56+8*c, (8*r+3)*56+8*c+1, (8*r+3)*56+8*c+2,(8*r+3)*56+8*c+3,(8*r+3)*56+8*c+4,
                (8*r+3)*56+8*c+5, (8*r+3)*56+8*c+6, (8*r+3)*56+8*c+7,
                (8*r+4)*56+8*c, (8*r+4)*56+8*c+1, (8*r+4)*56+8*c+2,(8*r+4)*56+8*c+3,(8*r+4)*56+8*c+4,
                (8*r+4)*56+8*c+5, (8*r+4)*56+8*c+6, (8*r+4)*56+8*c+7,
                (8*r+5)*56+8*c, (8*r+5)*56+8*c+1, (8*r+5)*56+8*c+2,(8*r+5)*56+8*c+3,(8*r+5)*56+8*c+4,
                (8*r+5)*56+8*c+5, (8*r+5)*56+8*c+6, (8*r+5)*56+8*c+7,
                (8*r+6)*56+8*c, (8*r+6)*56+8*c+1, (8*r+6)*56+8*c+2,(8*r+6)*56+8*c+3,(8*r+6)*56+8*c+4,
                (8*r+6)*56+8*c+5, (8*r+6)*56+8*c+6, (8*r+6)*56+8*c+7,
                (8*r+7)*56+8*c, (8*r+7)*56+8*c+1, (8*r+7)*56+8*c+2,(8*r+7)*56+8*c+3,(8*r+7)*56+8*c+4,
                (8*r+7)*56+8*c+5, (8*r+7)*56+8*c+6, (8*r+7)*56+8*c+7])

    def get_features(self,x):
        layers = []
        for i in range(4): #self.num_layers
            layers.append(str(7-i))
        # layers = ['4','5'] # '5','4'
        features = {}
        for name, module in list(self.resnet_projector.named_children()):
            x = module(x)
            if name in layers:
                features[str(int(name)-4)] = x
        return features

    def forward(self, x, return_scale_attention=False):
        if self.backbone=='r50_Swav' or self.backbone=='r50_BT' or self.backbone=='r50_MoCoV2':
            #x = self.get_features(x)  # feature extraction 
            x = self.resnet_projector(x) # feature extraction for resnet 50 pretrained on TCGA, output is a list contains all scales
            x = {str(i): output for i, output in enumerate(x)}
        else:
            x = self.get_features(x)

        if self.scale_token == 'channel':
            channel_fuse = {}
            channel_fuse['0'] = self.chann_proj1(x['0'])
            channel_fuse['1'] = self.chann_proj2(x['1'])
            channel_fuse['2'] = self.chann_proj3(x['2'])
            channel_fuse['3'] = x['3']
            channel_fuse_all = torch.cat([channel_fuse[key] for key in sorted(channel_fuse.keys())], dim=1) # gather channel-wise information
            channel_token = self.chann_proj_all(channel_fuse_all).unsqueeze(-1).permute(0,2,3,1) #49,1,768
            B,_,_,_ = channel_token.shape
        else:
            B,_,_,_ = x['0'].shape
        
        C = self.proj_dim
        if self.num_layers == 2:
            x = self.projection({'2':x['2'],'3':x['3']})
            x['3'] = x['3'].reshape(B,C,-1)
            x['2'] = x['2'].reshape(B,C,-1)
            #print(self.index['3'])
            #print(x['3'])
            x['3']= x['3'][:,:,self.index['3']] # [64, 768, 7, 7] -> [64, 49, 1, 7, 7]
            x['2']= x['2'][:,:,self.index['2']] # [64, 768, 14, 14] -> [64, 49, 4, 14, 14]
            x = torch.cat((x['3'],x['2']), dim = -1).permute(0,2,3,1) # [64, 768, 49, 5] -> [64, 49, 5, 768]
        elif self.num_layers == 4:
            x = self.projection({'0':x['0'],'1':x['1'],'2':x['2'],'3':x['3']})
            x['3'] = x['3'].reshape(B,C,-1)
            x['2'] = x['2'].reshape(B,C,-1)
            x['3']= x['3'][:,:,self.index['3']] # [64, 768, 7, 7] -> [64, 49, 1, 7, 7]
            x['2']= x['2'][:,:,self.index['2']] # [64, 768, 14, 14] -> [64, 49, 4, 14, 14]
            x['1'] = x['1'].reshape(B,C,-1)
            x['0'] = x['0'].reshape(B,C,-1)
            x['1']= x['1'][:,:,self.index['1']]
            x['0']= x['0'][:,:,self.index['0']]
            x = torch.cat((x['3'],x['2'],x['1'],x['0']), dim = -1).permute(0,2,3,1) 
        elif self.num_layers == 3:
            x = self.projection({'1':x['1'],'2':x['2'],'3':x['3']})
            x['3'] = x['3'].reshape(B,C,-1)
            x['2'] = x['2'].reshape(B,C,-1)
            x['3']= x['3'][:,:,self.index['3']] # [64, 768, 7, 7] -> [64, 49, 1, 7, 7]
            x['2']= x['2'][:,:,self.index['2']] # [64, 768, 14, 14] -> [64, 49, 4, 14, 14]
            x['1'] = x['1'].reshape(B,C,-1)
            x['1']= x['1'][:,:,self.index['1']]
            x = torch.cat((x['3'],x['2'],x['1']), dim = -1).permute(0,2,3,1) 
        if self.scale_token == 'channel':
            x = torch.cat((channel_token, x), dim=2)
        elif self.scale_token == 'random':
            x = torch.cat((self.channel_token.expand(B, 49, -1, -1), x), dim=2)

        return self.vision_transformer(x,return_scale_attention=return_scale_attention) 
        
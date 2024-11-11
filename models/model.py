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

class MyModel(nn.Module):
    def __init__(self, depth=None,patch_size=49, embed_dim=256,num_heads=6,init_values = 1e-5,num_classes=2,num_layers = 4, proj_dim = 512,model_ver='originalViT', pretrained=True):
        super().__init__()
        self.name = model_ver
        self.num_layers = num_layers
        self.proj_dim = proj_dim
        if pretrained:
            # self.resnet_projector = nn.Sequential(*list(models.resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-2])
            #self.resnet_projector = nn.Sequential(*list(models.resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-2])
            #BZ:
            self.resnet_projector = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
            # self.vanilla_hybrid = timm.create_model('vit_small_r26_s32_224.augreg_in21k_ft_in1k',pretrained=True,
            # num_classes=num_classes)
            # self.projection = nn.Conv2d(1024, self.proj_dim, kernel_size=(1,1),stride=(1,1))
            # self.projection1 = nn.Conv2d(2048, 768, kernel_size=(1,1),stride=(1,1))
            # nn.init.kaiming_normal_(self.projection.weight)  
            # if self.projection.bias is not None:
            #     nn.init.normal_(self.projection.bias, std=.02)
            # nn.init.kaiming_normal_(self.projection1.weight)  
            # if self.projection1.bias is not None:
            #     nn.init.normal_(self.projection1.bias, std=.02) # 1e-6
            # self.channel_squeeze =  nn.Sequential(
            #     nn.Linear(self.proj_dim, 384, bias=True),
            #     nn.GELU(),
            #     nn.Linear(384, 384, bias=True),
            # )
            # for layer in self.channel_squeeze:
            #     if isinstance(layer, nn.Linear):
            #         nn.init.trunc_normal_(layer.weight, std=0.02)
            #         nn.init.zeros_(layer.bias)

        else:
            self.resnet_projector = nn.Sequential(*list(models.resnet50().children())[:-2])
            # self.resnet_projector = nn.Sequential(*list(models.resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-2])
            # print("resnet 18 pretrained weights loaded!")
            # self.resnet_projector = Backbone()
            # self.resnet_projector2 = Backbone2() 
        self.chann_proj1 = Channel_Projector_layer1()
        self.chann_proj2 = Channel_Projector_layer2()
        self.chann_proj3 = Channel_Projector_layer3()
        self.chann_proj_all = Channel_Projector_All()
        self.projection = Projection(num_layers=self.num_layers, proj_dim=self.proj_dim )
        if self.num_layers > 1:
            self.vision_transformer = MultiscaleTransformer(pretrained=pretrained,depth=depth, scales=num_layers, num_heads=num_heads,patch_size=patch_size,embed_dim=embed_dim, 
                                            init_values = None,num_classes=num_classes,model_type = self.name,attn_drop_rate=0.1,drop_rate=0.1)  
            print("multiscaletransformer!")
            # self.scale_former = ScaleFormer(depth=12,scales=self.num_layers,num_heads=12,embed_dim=self.proj_dim) # consistent with pretrained hybrid

        # self.vanilla_vit= timm.models.vision_transformer.vit_base_patch32_224(pretrained=True,num_classes = num_classes)
        
        # for param in self.resnet_projector.parameters():
        #     param.requires_grad = False

        self.index = {}
        for i in range(4):
            self.index[f'{4-i-1}']=torch.empty([49,4**i],dtype=torch.int32)

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

    def forward(self, x):
        x = self.get_features(x)
        # x0 = self.resnet_projector(x) # x1,x2,x3
        # x1 = self.resnet_projector2(x)# x1,x2
        ################ for pretrained hybrid model ###############
        # x = self.vanilla_hybrid.patch_embed.backbone.stem(x) # 64, 56, 56
        # x1 = self.vanilla_hybrid.patch_embed.backbone.stages[0](x) # 256, 56, 56
        # x2 = self.vanilla_hybrid.patch_embed.backbone.stages[1](x1) # 512, 28, 28
        # x3 = self.vanilla_hybrid.patch_embed.backbone.stages[2](x2) # 1024, 14, 14
        # x4 = self.vanilla_hybrid.patch_embed.backbone.stages[3](x3) # 2048, 7, 7
        # x4 = self.vanilla_hybrid.patch_embed.backbone.norm(x4)
        # x4 = self.vanilla_hybrid.patch_embed.backbone.head(x4, pre_logits=False) # All Identity by default
        # # output = self.vanilla_hybrid.patch_embed.proj(x4) # 384, 7, 7 # use pretrained patch emb
        # x4 = self.projection1(x4)
        # x3 = self.projection(x3) # this is from scratch
        # # x2 = self.projection1(x2) # this is from scratch

        # B,C,H,W = x4.shape
        # x4 = x4.reshape(B,C,-1) # B, 384, 49
        # x3 = x3.reshape(B,C,-1) # B, 384, 196
        # x4= x4[:,:,self.index['3']]
        # x3= x3[:,:,self.index['2']]
        # # # x2 = x2.reshape(B,C,-1) # B, 384, 784
        # # # x2 = x2[:,:,self.index['1']]
        # if self.num_layers == 2:
        #     x = torch.cat((x4,x3), dim = -1).permute(0,2,3,1) # [64, 384, 49, 5] -> [64, 49, 5, 384]
        # # if self.num_layers == 3:
        # #     x = torch.cat((x4,x3,x2), dim = -1).permute(0,2,3,1) # [64, 384, 49, 22] -> [64, 49, 22, 384]

        # output = self.scale_former(x) # scale attention from scratch # [64, 49, 384]
        # output = self.channel_squeeze(output)
        # # x4_ori = x4_ori.reshape(x4_ori.shape[0],384,-1).permute(0,2,1)
        # # output = output * x4_ori
        # # print(output.shape)
        # # output = output.reshape[output.shape[0],self.proj_dim,7,7]
        # # print(output.shape)
        # # output = self.vanilla_hybrid.patch_embed.proj(output)
        # # patch attention from pretrained
        # # cls_token = self.vanilla_hybrid.cls_token.expand(output.shape[0], -1, -1)    
        # # output = torch.cat((cls_token, output), dim=1)
        # # output = output + self.vanilla_hybrid.pos_embed
        # # output = self.vanilla_hybrid.pos_drop(output)
        # # output = output.reshape(output.shape[0],output.shape[1],-1).permute(0,2,1)
        # output = self.vanilla_hybrid._pos_embed(output)
        # output = self.vanilla_hybrid.norm_pre(output)
        # output = self.vanilla_hybrid.blocks(output)
        # output = self.vanilla_hybrid.norm(output)
        # output = self.vanilla_hybrid.forward_head(output)
        # output = self.vanilla_hybrid(x)
        ################ end for pretrained hybrid model ###############
        # fea = {'3':x0[-1],'2':x1[-1]} # using features from different encoder
        # fea['3'] = self.projection.proj_heads3(x[-1])
        # fea['2'] = self.projection.proj_heads2(x[-2])
        # channel
        channel_fuse = {}
        channel_fuse['0'] = self.chann_proj1(x['0'])
        channel_fuse['1'] = self.chann_proj2(x['1'])
        channel_fuse['2'] = self.chann_proj3(x['2'])
        channel_fuse['3'] = x['3']
        channel_fuse_all = torch.cat([channel_fuse[key] for key in sorted(channel_fuse.keys())], dim=1)
        channel_token = self.chann_proj_all(channel_fuse_all).unsqueeze(-1).permute(0,2,3,1) #49,1,768

        x = self.projection({'2':x['2'],'3':x['3']})
        if self.name == 'scaleformer':
            # B,C,H,W = x['0'].shape
            # x['1'] = x['1'].reshape(B,C,-1)
            # x['0'] = x['0'].reshape(B,C,-1)
            # x['1']= x['1'][:,:,self.index['1']]
            # x['0']= x['0'][:,:,self.index['0']]
            # x = torch.cat((x['1'],x['0']), dim = -1).permute(0,2,3,1) 
            # x = x['0'].permute(0,2,3,1) 
            B,C,H,W = x['3'].shape
            x['3'] = x['3'].reshape(B,C,-1)
            x['2'] = x['2'].reshape(B,C,-1)
            x['3']= x['3'][:,:,self.index['3']] # [64, 768, 7, 7] -> [64, 49, 1, 7, 7]
            x['2']= x['2'][:,:,self.index['2']] # [64, 768, 14, 14] -> [64, 49, 4, 14, 14]
            if self.num_layers == 2:
                x = torch.cat((x['3'],x['2']), dim = -1).permute(0,2,3,1) # [64, 768, 49, 5] -> [64, 49, 5, 768]
            elif self.num_layers == 4:
                x['1'] = x['1'].reshape(B,C,-1)
                x['0'] = x['0'].reshape(B,C,-1)
                x['1']= x['1'][:,:,self.index['1']]
                x['0']= x['0'][:,:,self.index['0']]
                x = torch.cat((x['3'],x['2'],x['1'],x['0']), dim = -1).permute(0,2,3,1) 
            elif self.num_layers == 3:
                x['1'] = x['1'].reshape(B,C,-1)
                x['1']= x['1'][:,:,self.index['1']]
                x = torch.cat((x['3'],x['2'],x['1']), dim = -1).permute(0,2,3,1) 
        x = torch.cat((channel_token, x), dim=2) 
        output = self.vision_transformer(x)
        # cls_token = self.vanilla_vit.cls_token.expand(output.shape[0], -1, -1)    
        # output = torch.cat((cls_token, output), dim=1)
        # output = output + self.vanilla_vit.pos_embed
        # output = self.vanilla_vit.pos_drop(output)
        # output = self.vanilla_vit.norm_pre(output)
        # output = self.vanilla_vit.blocks(output)
        # output = self.vanilla_vit.norm(output)
        # output = self.vanilla_vit.forward_head(output)
        # x = x['3'].permute(0,2,1) 
        # x = self.vanilla_vit._pos_embed(x)
        # x = self.vanilla_vit.patch_drop(x)
        # x = self.vanilla_vit.norm_pre(x)
        # x = self.vanilla_vit.blocks(x)
        # x = self.vanilla_vit.norm(x)
        # x = self.vanilla_vit.forward_head(x)
        # return x
        # x1,x2,x3,output = self.resnet_projector(x)
        return output
        

class HybridModel(nn.Module):
    def __init__(self, num_classes=100,num_blocks=12,proj_dim=768,backbone='r50', freeze_backbone=False, num_heads=6):
        super().__init__()
        self.num_blocks = num_blocks
        self.proj_dim = proj_dim
        #self.resnet_projector = nn.Sequential(*list(models.resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-2])
        #BZ:
        #self.resnet_projector = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])

        




        self.backbone=backbone
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
            print("Weights of Resnet 50 pretrained on TCGA+TULIP loaded!")
        elif backbone =='r50_BT':
            self.resnet_projector = resnet50FeatureExtractor(pretrained=True, progress=False, key="BT") # MoCoV2, BT ,removed the fc 
            print("Weights of Resnet 50 pretrained on TCGA+TULIP loaded!")
        elif backbone =='r50_MoCoV2':
            self.resnet_projector = resnet50FeatureExtractor(pretrained=True, progress=False, key="MoCoV2") # MoCoV2, BT ,removed the fc 
            print("Weights of Resnet 50 pretrained on TCGA+TULIP loaded!")

        if freeze_backbone:        # freeze the backbone in training or not
            for param in self.resnet_projector.parameters():
                param.requires_grad = False
            print("Backbone freezed during training!")





        
        # for param in self.resnet_projector.parameters():
        #     param.requires_grad = False
        ### for baseline 2

        #BZ: num_layers is number of scales
        self.projection = Projection_for_hybrid( proj_dim=self.proj_dim, backbone = self.backbone )
        # print('num_blocks:')
        # print(self.num_blocks)
        # print('num_heads:')
        # print(num_heads)
        self.vision_transformer = VisionTransformer(patch_size=32, depth=self.num_blocks,num_classes=num_classes, num_heads=num_heads)  # n_patches:49
        ### for mixed feature experiment, we only need the vanilla ViT without patch embedding
        # self.vision_transformer = VisionTransformer(num_classes=num_classes,depth=self.num_blocks,embed_dim=proj_dim)  # emb_dim: 512, n_patches:196, attn_dim:196
        ### for mixed feature experiment wi two attn, emb_dim: 512, n_patches:196, attn_dim:512
        # self.projection1 = Projection(num_layers=1, proj_dim=784*4 )
        # self.projection2 = nn.Conv2d(784, proj_dim, kernel_size=(1,1),stride=(1,1))
        # nn.Conv1d(in_channels=512, out_channels=proj_dim, kernel_size=1)
        # nn.init.kaiming_normal_(self.projection2.weight)
        # if self.projection2.bias is not None:
                # nn.init.normal_(self.projection2.bias, std=1e-6)
        # self.vision_transformer2 = VisionTransformer(num_classes=num_classes,depth=self.num_blocks,patch_size=8,embed_dim=196,num_heads=num_heads,class_token=False,global_pool = '') # 224/8 x 224/8 =28x28=784
        # self.test =  nn.Linear(784,num_classes)

    def forward(self, x):
        if self.backbone=='r50_Swav':
            x =self.resnet_projector(x)[-1]
        elif self.backbone=='r50_BT':
            x =self.resnet_projector(x)[-1]
        elif self.backbone=='r50_MoCoV2':
            x =self.resnet_projector(x)[-1]
        else:
            x = self.resnet_projector(x) # 2048,7,7

        x = self.projection(x) # 784*4,7,7
        ###  this is for baseline 2: a pretrained r50 + vit from scratch(replacing patch emb by a single layer projection)
        x = x.flatten(2).transpose(1, 2)
        # remove patch emb layer from ViT
        x = self.vision_transformer._pos_embed(x)
        x = self.vision_transformer.patch_drop(x)
        x = self.vision_transformer.norm_pre(x)
        x = self.vision_transformer.blocks(x)
        x = self.vision_transformer.norm(x)
        x = self.vision_transformer.forward_head(x)
        ###    end of baseline 2

        ### this is for mixed feature experiment
        # B,C,H,W = x.shape
        # x = x.reshape(B,784,-1) # .transpose(1, 2)

        # ### for mixed feature with two attns
        # B,C,H,W = x.shape  # 784*4,7,7
        # x = x.reshape(B,784,-1) # .permute(0,2, 1) # 784,196
        # x = self.vision_transformer2._pos_embed(x)
        # x = self.vision_transformer2.patch_drop(x)
        # x = self.vision_transformer2.norm_pre(x)
        # x = self.vision_transformer2.blocks(x)
        # x = self.vision_transformer2.norm(x)
        # # x = x[:,:,0]
        # # x = self.test(x)
        # x = x.reshape(B,784,2*H,2*W) # 784,14,14
        # x = self.projection2(x) # 768,14,14
        # x = x.reshape(B,self.proj_dim,-1).permute(0,2, 1) # 196,768

        # remove patch emb layer from ViT
        # x = self.vision_transformer._pos_embed(x)
        # x = self.vision_transformer.patch_drop(x)
        # x = self.vision_transformer.norm_pre(x)
        # x = self.vision_transformer.blocks(x)
        # x = self.vision_transformer.norm(x)
        # # # # x = x[:,0,:]
        # # # # x = self.test(x)
        # x = self.vision_transformer.forward_head(x)
        return x

class ViTBase16(nn.Module):
    def __init__(self,n_classes=100, model_type = 'R50ViT'):
        super().__init__()
        if model_type == 'ViT':
            self.model = VisionTransformer(num_classes=n_classes)
            
        elif model_type == 'ViTPretrained':
            self.model = timm.create_model(
            'vit_base_r50_s16_224_in21k',pretrained=True,
            num_classes=n_classes)

        elif model_type == 'R50ViTPretrained':
            self.model = timm.create_model(
            'vit_base_r50_s16_224_in21k',pretrained=True,
            num_classes=n_classes)
            print(model_type, 'is created.')

        elif model_type == 'R50ViT':
            # self.model = timm.create_model(
            # 'vit_base_r50_s16_224_in21k',pretrained=False,
            # num_classes=n_classes)
            self.model = timm.create_model(
            'vit_small_r26_s32_224.augreg_in21k_ft_in1k',pretrained=True,
            num_classes=n_classes)
            print('pretrained hybrid loaded!')
        # self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        self.name = model_type

    def forward(self,x):
        return self.model(x)
    
def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params/1000000, total_params/1000000


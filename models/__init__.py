from .model import *
from .model_wo_extra_params import *
from .resnet50ssl import *

def build_model(depth=12,patch_size=49, embed_dim=256,num_heads=6, init_values = 1e-5,num_classes=100,num_layers = 4, 
                proj_dim = 384,model_ver='scaleformer',pretrained=True,freeze=True):
    return MyModel(depth=depth,patch_size=patch_size, embed_dim=embed_dim,num_heads=num_heads,num_classes=num_classes,init_values = init_values,
                num_layers = num_layers, proj_dim = proj_dim,model_ver = model_ver,pretrained=pretrained,freeze=freeze)
                
def build_model_no_extra_params(depth=12, 
                embed_dim=256,
                num_heads=6, 
                num_classes=100,
                num_layers = 4, 
                num_patches = 49,
                proj_dim = 384, 
                mlp_ratio=4., 
                attn_drop_rate=0., 
                proj_drop_rate = 0.,
                freeze_backbone=True,
                backbone = 'r50',
                pretrained=True):

    return MyModel_no_extra_params(depth=depth, 
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_classes=num_classes,
                num_layers = num_layers, 
                num_patches = num_patches,
                proj_dim = proj_dim,
                mlp_ratio = mlp_ratio, 
                attn_drop_rate = attn_drop_rate,  
                proj_drop_rate = proj_drop_rate, 
                freeze_backbone=freeze_backbone,  
                backbone = backbone,
                pretrained=pretrained)

def build_hybrid(num_classes=100,num_blocks=12,proj_dim=768,num_heads=12): # model_ver='R50ViT'
    # return HybridModel(num_classes=num_classes,num_blocks=num_blocks,proj_dim=proj_dim,num_heads=num_heads)
    return HybridModel(num_classes=num_classes,num_blocks=num_blocks,proj_dim=proj_dim)
    # return ViTBase16(n_classes=num_classes,model_type = 'R50ViT')

# def r50_pretrained_TCGA(key="SwAV"): 
#     return resnet50(pretrained=True, progress=False, key=key)

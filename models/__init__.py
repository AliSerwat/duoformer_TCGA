# from .model import *
# from .model_wo_extra_params import *
from model import *
from model_wo_extra_params import *
from resnet50ssl import *
from swin_transformer import *

# def build_model(depth=12,patch_size=49, embed_dim=256,num_heads=6, init_values = 1e-5,num_classes=100,num_layers = 4, 
#                 proj_dim = 384,model_ver='scaleformer',pretrained=True):
    # return MyModel(depth=depth,patch_size=patch_size, embed_dim=embed_dim,num_heads=num_heads,num_classes=num_classes,init_values = init_values,
    #             num_layers = num_layers, proj_dim = proj_dim,model_ver = model_ver,pretrained=pretrained)
def build_model(depth=12, 
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
                scale_token='random',
                patch_attn = True):

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
                scale_token=scale_token,
                patch_attn=patch_attn)

def build_hybrid(num_classes=100,num_blocks=12,proj_dim=768,backbone='r50', freeze_backbone=False, num_heads=12): # model_ver='R50ViT'
    # return HybridModel(num_classes=num_classes,num_blocks=num_blocks,proj_dim=proj_dim,num_heads=num_heads)
    return HybridModel(num_classes=num_classes,num_blocks=num_blocks,proj_dim=proj_dim,backbone=backbone, freeze_backbone=freeze_backbone,num_heads = num_heads)
    # return ViTBase16(n_classes=num_classes,model_type = 'R50ViT')

def build_r50_swin_transformer(num_classes=1000,backbone='r50',freeze_backbone=True): 
    config = Config(config_dict)
    return RESSwinTransformer( img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=num_classes,
                            embed_dim=128,
                            depths=config.MODEL.SWIN.DEPTHS,
                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            ape=False,
                            norm_layer=nn.LayerNorm,
                            patch_norm=True,
                            use_checkpoint=False,
                            fused_window_process=False,
                            backbone=backbone,
                            freeze_backbone = freeze_backbone)

class Config:
    def __init__(self, config_dict):
        self.config = config_dict

    def __getattr__(self, item):
        value = self.config.get(item)
        if isinstance(value, dict):
            return Config(value)
        else:
            return value  # Return the actual value, not wrapped in Config

# swin_base_patch4_window7_224.yaml
config_dict = {
    "MODEL": {
        "TYPE": "swin",
        "NAME": "swin_base_patch4_window7_224",
        "DROP_PATH_RATE": 0.5,
        "SWIN": {
            "EMBED_DIM": 128,
            "DEPTHS": [2, 2, 18, 2],
            "NUM_HEADS": [4, 8, 16, 32],
            "WINDOW_SIZE": 7
        }
    }
}
# swin_tiny_patch4_window7_224.yaml
# config_dict = {
#     "MODEL": {
#         "TYPE": "swin",
#         "NAME": "swin_tiny_patch4_window7_224",
#         "DROP_PATH_RATE": 0.2,
#         "SWIN": {
#             "EMBED_DIM": 96,
#             "DEPTHS": [2, 2, 6, 2],
#             "NUM_HEADS": [ 3, 6, 12, 24 ],
#             "WINDOW_SIZE": 7
#         }
#     }
# }
import torch
import torch.nn as nn
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models import load_checkpoint

# MobileViT imports (assumed to be defined elsewhere)
# from .mobilevit import MobileViTBlock, MobileViT


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'mobilevit_xxs': _cfg(),
    'mobilevit_xs': _cfg(),
    'mobilevit_s': _cfg(),
    'mobilevitv2_050': _cfg(),
    'mobilevitv2_075': _cfg(),
    'mobilevitv2_100': _cfg(),
    'mobilevitv2_125': _cfg(),
    'mobilevitv2_150': _cfg(),
    'mobilevitv2_175': _cfg(),
    'mobilevitv2_200': _cfg(),
}


class MobileViT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, model_type='v1'):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # Initialize MobileViT Block
        self.mobilevit_block = MobileViTBlock(
                img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, 
                num_heads=num_heads, mlp_ratio=mlp_ratio, model_type=model_type)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.mobilevit_block(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # MobileViT backbone
        for blk in self.mobilevit_block.blocks:
            x = blk(x)

        x = self.mobilevit_block.norm(x)
        return x[:, 1:]

    def forward(self, x):
        x = self.forward_features(x)
        return x


@register_model
def mobilevit_xxs(pretrained=False, **kwargs):
    model = MobileViT(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., model_type='v1', **kwargs)
    model.default_cfg = default_cfgs['mobilevit_xxs']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def mobilevit_xs(pretrained=False, **kwargs):
    model = MobileViT(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., model_type='v1', **kwargs)
    model.default_cfg = default_cfgs['mobilevit_xs']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def mobilevit_s(pretrained=False, **kwargs):
    model = MobileViT(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., model_type='v1', **kwargs)
    model.default_cfg = default_cfgs['mobilevit_s']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def mobilevitv2_050(pretrained=False, **kwargs):
    model = MobileViT(embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., model_type='v2', **kwargs)
    model.default_cfg = default_cfgs['mobilevitv2_050']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def mobilevitv2_075(pretrained=False, **kwargs):
    model = MobileViT(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., model_type='v2', **kwargs)
    model.default_cfg = default_cfgs['mobilevitv2_075']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def mobilevitv2_100(pretrained=False, **kwargs):
    model = MobileViT(embed_dim=512, depth=12, num_heads=8, mlp_ratio=4., model_type='v2', **kwargs)
    model.default_cfg = default_cfgs['mobilevitv2_100']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def mobilevitv2_125(pretrained=False, **kwargs):
    model = MobileViT(embed_dim=640, depth=12, num_heads=10, mlp_ratio=4., model_type='v2', **kwargs)
    model.default_cfg = default_cfgs['mobilevitv2_125']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def mobilevitv2_150(pretrained=False, **kwargs):
    model = MobileViT(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., model_type='v2', **kwargs)
    model.default_cfg = default_cfgs['mobilevitv2_150']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def mobilevitv2_175(pretrained=False, **kwargs):
    model = MobileViT(embed_dim=896, depth=12, num_heads=14, mlp_ratio=4., model_type='v2', **kwargs)
    model.default_cfg = default_cfgs['mobilevitv2_175']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def mobilevitv2_200(pretrained=False, **kwargs):
    model = MobileViT(embed_dim=1024, depth=12, num_heads=16, mlp_ratio=4., model_type='v2', **kwargs)
    model.default_cfg = default_cfgs['mobilevitv2_200']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

import torch
import torch.nn as nn
from functools import partial
import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model
import numpy as np
import sys
import torch.nn.functional as F
# from petl_factory import Linear


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 lora=False, lora_rank=8, lora_alpha=16., lora_dropout=0., lora_init='lora'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        if lora:
            self.fc1 = Linear(in_features, hidden_features, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_init=lora_init)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if lora:
            self.fc2 = Linear(hidden_features, out_features, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_init=lora_init)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 lora=False, lora_rank=8, lora_alpha=16., lora_dropout=0., lora_init='lora'):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        if lora:
            self.qkv = Linear(dim, dim * 3, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_init=lora_init)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if lora:
            self.proj = Linear(dim, dim, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_init=lora_init)
        else:
            self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # def easy_forward(self, x):
    #     B,N,C = x.shape
    #     cls_token = x[:,0:1,:]
    #     other_tokens = x
    #     tmp = (cls_token @ other_tokens.transpose(-2, -1)) * (self.dim ** -0.5)
    #     tmp = tmp.softmax(dim=-1)
    #     return tmp.view(B, N).detach()

    def forward(self, x):
        B, N, C = x.shape

        # qkv
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # batch x head x seq x embed_chunk

        # attention matrix
        attn = (q @ k.transpose(-2, -1)) * self.scale # batch x head x seq x seq'
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v) # batch x head x seq x embed_chunk
        x = x.transpose(1, 2) # batch x seq x head x embed_chunk
        x = x.reshape(B, N, C)

        # projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 lora=False, lora_rank=8, lora_alpha=16., lora_dropout=0., lora_init='lora'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            lora=lora, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_init=lora_init)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       lora=lora, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_init=lora_init)

        # # per-block token prune ratio
        # self.token_prune_ratio = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        B,N,C = x.shape
        # token_prune_ratio = self.token_prune_ratio.data.item()

        # # token selection
        # if token_prune_ratio > 0:
        #     K = int(N * (1 - token_prune_ratio))
        #     avg_cls_attn = self.attn.easy_forward(self.norm1(x)) # batch x seq
        #     avg_cls_attn[:,0] = 1e9 # ensure [CLS] is always kept at index 0
        #     if self.training:
        #         ids_shuffle = torch.multinomial(avg_cls_attn[:, 1:], N-1, replacement=False) + 1
        #         ids_shuffle = torch.cat((torch.zeros(B, 1).int().to(x.device), ids_shuffle), dim=1)
        #     else:
        #         _, ids_shuffle = torch.sort(avg_cls_attn, dim=1, descending=True)
        #     ids_restore = torch.argsort(ids_shuffle, dim=1)
        #     ids_keep = ids_shuffle[:, :K]
        #     ids_dump = ids_shuffle[:, K:]
        #     keep_tokens = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, C)) # batch x seq' x embed
        #     dump_tokens = torch.gather(x, 1, ids_dump.unsqueeze(-1).expand(-1, -1, C)) # batch x seq'' x embed
        #     fused = torch.mean(dump_tokens, dim=1, keepdim=True)
        #     x = torch.cat([keep_tokens, fused], dim=1)
        #     assert x.shape[1] == K+1

        # MHSA
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # # token recovery
        # if token_prune_ratio > 0:
        #     x_ = torch.cat([x[:,:K,:], dump_tokens+x[:,K:K+1,:].repeat(1,N-K,1)], dim=1)
        #     x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VIT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 lora=False, lora_rank=8, lora_alpha=16., lora_dropout=0., lora_init='lora'):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        print('Sequence length', num_patches)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                lora=lora, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_init=lora_init)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
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
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, clip_teacher=None):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class VisionTransformer(VIT):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, global_pool_wcls=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.global_pool_wcls = global_pool_wcls
        if self.global_pool or self.global_pool_wcls:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        elif self.global_pool_wcls:
            x = x.mean(dim=1)   # global pool with cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model




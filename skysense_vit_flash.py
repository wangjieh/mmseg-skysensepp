"""SkySense++ ViT-MSL backbone for MMSegmentation fine-tuning.

This module is a downstream segmentation version of the original
``VisionTransformerMSL`` backbone. The pre-training-only path that requires
annotation images is intentionally removed; ``forward`` accepts only image
inputs and returns NCHW feature maps for MMSegmentation decode heads.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from contextlib import nullcontext

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention as MMCVMultiheadAttention
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, kaiming_init, trunc_normal_
from mmengine.runner.checkpoint import load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.models.utils import PatchEmbed, resize
from mmseg.registry import MODELS


try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - for older PyTorch only
    SDPBackend = None
    sdpa_kernel = None


class SDPAMultiheadAttention(BaseModule):
    """Self-attention module that uses PyTorch SDPA/FlashAttention.

    This class is designed as a near drop-in replacement for MMCV's
    ``MultiheadAttention`` in this backbone:

    - it keeps an inner ``nn.MultiheadAttention`` named ``attn`` so existing
      checkpoints using keys like ``layers.0.attn.attn.in_proj_weight`` remain
      compatible;
    - for normal ViT self-attention with no masks, it calls
      ``torch.nn.functional.scaled_dot_product_attention`` directly;
    - if masks or non-self attention are provided, it falls back to
      ``nn.MultiheadAttention(..., need_weights=False)`` so PyTorch can still
      use the optimized SDPA path when supported.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 dropout_layer: Optional[dict] = None,
                 batch_first: bool = False,
                 bias: bool = True,
                 force_flash_attn: bool = False,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        unsupported = sorted(kwargs.keys())
        if unsupported:
            warnings.warn('Unused SDPAMultiheadAttention kwargs: '
                          f'{unsupported}')

        if embed_dims % num_heads != 0:
            raise ValueError(
                f'embed_dims ({embed_dims}) must be divisible by '
                f'num_heads ({num_heads}).')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.batch_first = batch_first
        self.force_flash_attn = force_flash_attn

        # Keep the same nested name as MMCV MultiheadAttention:
        # layers.{i}.attn.attn.in_proj_weight / out_proj.weight / ...
        self.attn = nn.MultiheadAttention(
            embed_dims,
            num_heads,
            dropout=attn_drop,
            bias=bias,
            batch_first=batch_first)
        self.proj_drop = nn.Dropout(proj_drop)

        # In this backbone no attention DropPath is passed by default. Keeping
        # an Identity here matches the existing config and avoids extra deps.
        if dropout_layer is not None:
            warnings.warn(
                'SDPAMultiheadAttention received dropout_layer, but this '
                'backbone keeps attention residual dropout as Identity. '
                'FFN DropPath is unchanged.')
        self.dropout_layer = nn.Identity()

    def _sdpa_context(self):
        if not self.force_flash_attn:
            return nullcontext()
        if sdpa_kernel is None or SDPBackend is None:
            raise RuntimeError(
                'force_flash_attn=True requires torch.nn.attention.sdpa_kernel '
                'and SDPBackend, which are unavailable in this PyTorch build.')
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)

    def _forward_self_attention_no_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path for standard ViT self-attention.

        Args:
            x: Tensor with shape [B, N, C] when batch_first=True, otherwise
               [N, B, C].

        Returns:
            Tensor with the same shape as ``x``.
        """
        original_batch_first = self.batch_first
        if not original_batch_first:
            x = x.transpose(0, 1)

        batch_size, num_tokens, channels = x.shape
        if channels != self.embed_dims:
            raise ValueError(
                f'Expected channels={self.embed_dims}, got {channels}.')

        qkv = F.linear(
            x,
            self.attn.in_proj_weight,
            self.attn.in_proj_bias)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads,
                          self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)

        # SDPA applies dropout according to dropout_p even in eval mode, so
        # explicitly disable it during evaluation.
        dropout_p = self.attn.dropout if self.training else 0.0

        with self._sdpa_context():
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False)

        out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        out = F.linear(out, self.attn.out_proj.weight,
                       self.attn.out_proj.bias)

        if not original_batch_first:
            out = out.transpose(0, 1)
        return out

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_pos: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if identity is None:
            identity = query

        # Preserve MMCV-style defaults.
        if key is None:
            key = query
        if value is None:
            value = key

        is_self_attention = key is query and value is key
        has_pos_embed = query_pos is not None or key_pos is not None

        if query_pos is not None:
            query = query + query_pos
        if key_pos is None and query_pos is not None and is_self_attention:
            key_pos = query_pos
        if key_pos is not None:
            key = key + key_pos

        # Standard ViT path: self-attention without masks/extra attention
        # position arguments. This directly calls SDPA, which can dispatch to
        # FlashAttention.
        if (is_self_attention and not has_pos_embed and attn_mask is None
                and key_padding_mask is None):
            out = self._forward_self_attention_no_mask(query)
        else:
            # Fallback path: still request need_weights=False so PyTorch can use
            # optimized SDPA kernels where possible.
            with self._sdpa_context():
                out = self.attn(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False)[0]

        return identity + self.dropout_layer(self.proj_drop(out))



from ..utils import load_checkpoint_state_dict, prepare_skysense_state_dict


class TransformerEncoderLayer(BaseModule):
    """A ViT encoder layer with OpenMMLab naming compatible with SkySense++."""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 batch_first: bool = True,
                 attn_cfg: Optional[dict] = None,
                 ffn_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 use_flash_attn: bool = True,
                 force_flash_attn: bool = False,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        attn_cfg = dict(attn_cfg or {})
        ffn_cfg = dict(ffn_cfg or {})

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))
        if use_flash_attn:
            self.attn = SDPAMultiheadAttention(
                **attn_cfg, force_flash_attn=force_flash_attn)
        else:
            self.attn = MMCVMultiheadAttention(**attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.ffn = FFN(**ffn_cfg)
        self.with_cp = with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        def _inner_forward(x):
            x = self.attn(self.norm1(x), identity=x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class SkySenseVisionTransformer(BaseModule):
    """SkySense++ ViT-MSL backbone for semantic segmentation.

    The official S1/S2 release weights have the following important traits:

    - ``pos_embed`` contains patch tokens only, no class-position token;
    - ``cls_token`` exists as a parameter but is not used in downstream
      segmentation;
    - ``mask_token`` / ``vocabulary_token`` / ``vocabulary_weight`` are kept so
      release weights can be loaded without dropping SkySense++ specific
      tensors.

    Args mirror MMSegmentation's ViT where possible, but ``with_cls_token`` is
    effectively disabled for the SkySense++ release checkpoints.
    """

    def __init__(self,
                 img_size: Union[int, Tuple[int, int]] = 256,
                 patch_size: int = 4,
                 patch_pad: Union[str, int] = 'corner',
                 in_channels: int = 10,
                 embed_dims: int = 1024,
                 num_layers: int = 24,
                 num_heads: int = 16,
                 mlp_ratio: int = 4,
                 out_indices: Union[int, Sequence[int]] = (5, 11, 17, 23),
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 with_cls_token: bool = False,
                 output_cls_token: bool = False,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 act_cfg: dict = dict(type='GELU'),
                 patch_norm: bool = False,
                 patch_bias: bool = True,
                 pre_norm: bool = False,
                 final_norm: bool = False,
                 interpolate_mode: str = 'bicubic',
                 num_fcs: int = 2,
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 use_flash_attn: bool = True,
                 force_flash_attn: bool = False,
                 frozen_stages: int = -1,
                 vocabulary_size: int = 64,
                 input_adapt_mode: str = 'auto',
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__(init_cfg=None)

        # Accept legacy config fields from MMSeg / original fine-tuning configs.
        self.downscale_indices = kwargs.pop('downscale_indices', None)
        self.out_origin = kwargs.pop('out_origin', False)
        self.frozen_exclude = kwargs.pop('frozen_exclude', ['all'])
        if kwargs:
            warnings.warn('Unused SkySenseVisionTransformer kwargs: '
                          f'{sorted(kwargs.keys())}')

        if pretrained is not None:
            warnings.warn('`pretrained` is deprecated. Use '
                          '`init_cfg=dict(type="Pretrained", checkpoint=...)`.',
                          DeprecationWarning)
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.skysense_init_cfg = init_cfg

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2
        else:
            raise TypeError('img_size must be int or tuple[int, int].')

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.use_flash_attn = use_flash_attn
        self.force_flash_attn = force_flash_attn
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.frozen_stages = frozen_stages
        self.with_cls_token = False
        self.output_cls_token = output_cls_token
        if with_cls_token:
            warnings.warn('SkySense++ release ViT weights use patch-only '
                          'pos_embed; class token is kept as an unused '
                          'parameter for checkpoint compatibility.')

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding=patch_pad,
            bias=patch_bias,
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.vocabulary_size = vocabulary_size + 1
        self.vocabulary_token = nn.Parameter(
            torch.zeros(self.vocabulary_size, embed_dims))
        self.vocabulary_weight = nn.Parameter(
            torch.zeros(1, patch_size * patch_size))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            out_indices = [out_indices]
        self.out_indices = tuple(
            i if i >= 0 else num_layers + i for i in out_indices)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    use_flash_attn=use_flash_attn,
                    force_flash_attn=force_flash_attn,
                    batch_first=True))

        self.pre_norm = pre_norm
        if pre_norm:
            self.pre_norm_name, pre_norm_layer = build_norm_layer(
                norm_cfg, embed_dims, postfix='pre')
            self.add_module(self.pre_norm_name, pre_norm_layer)

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.input_adapt_mode = input_adapt_mode

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self) -> None:
        if (isinstance(self.skysense_init_cfg, dict)
                and self.skysense_init_cfg.get('type') == 'Pretrained'):
            checkpoint = self.skysense_init_cfg['checkpoint']
            state_dict = load_checkpoint_state_dict(checkpoint)
            target_grid = (self.img_size[0] // self.patch_size,
                           self.img_size[1] // self.patch_size)
            state_dict = prepare_skysense_state_dict(
                state_dict,
                self.state_dict(),
                pos_embed_key='pos_embed',
                patch_embed_key='patch_embed.projection.weight',
                pos_embed_grid=target_grid,
                interpolate_mode=self.interpolate_mode,
                input_adapt_mode=self.input_adapt_mode,
                drop_relative_position_buffers=False)
            load_state_dict(self, state_dict, strict=False)
            return

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.vocabulary_token, std=.02)
        nn.init.zeros_(self.vocabulary_weight)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    if 'ffn' in name:
                        nn.init.normal_(module.bias, mean=0., std=1e-6)
                    else:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                kaiming_init(module, mode='fan_in', bias=0.)
            elif isinstance(module, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                constant_init(module, val=1.0, bias=0.)

    def _resize_pos_embed(self, pos_embed: torch.Tensor,
                          hw_shape: Tuple[int, int]) -> torch.Tensor:
        if pos_embed.shape[1] == hw_shape[0] * hw_shape[1]:
            src_h = self.img_size[0] // self.patch_size
            src_w = self.img_size[1] // self.patch_size
        else:
            src_h = src_w = int(pos_embed.shape[1]**0.5)
            if src_h * src_w != pos_embed.shape[1]:
                raise ValueError(
                    f'Unexpected pos_embed shape {tuple(pos_embed.shape)}.')
        pos = pos_embed.reshape(1, src_h, src_w,
                                pos_embed.shape[-1]).permute(0, 3, 1, 2)
        pos = resize(
            pos,
            size=hw_shape,
            mode=self.interpolate_mode,
            align_corners=False)
        pos = pos.flatten(2).transpose(1, 2).contiguous()
        return pos

    def _pos_embedding(self, x: torch.Tensor,
                       hw_shape: Tuple[int, int]) -> torch.Tensor:
        pos_embed = self.pos_embed
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = self._resize_pos_embed(pos_embed, hw_shape)
        return self.drop_after_pos(x + pos_embed)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x, hw_shape = self.patch_embed(inputs)
        x = self._pos_embedding(x, hw_shape)
        if self.pre_norm:
            x = getattr(self, self.pre_norm_name)(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)
            if i in self.out_indices:
                batch_size, _, channels = x.shape
                out = x.reshape(batch_size, hw_shape[0], hw_shape[1],
                                channels).permute(0, 3, 1,
                                                  2).contiguous()
                outs.append(out)
        return tuple(outs)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages + 1):
            if i < len(self.layers):
                self.layers[i].eval()
                for param in self.layers[i].parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    module.eval()

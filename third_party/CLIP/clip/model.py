from collections import OrderedDict
from typing import Tuple, Union
from scipy import interpolate
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import math
import fvcore.nn.weight_init as weight_init
from detectron2.layers import get_norm, Conv2d


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.grid_size = spacial_dim
        self.in_features = embed_dim
        self.dense_clip = True

    def forward(self, x, mask=None):
        b, c, gh, gw = x.shape
        # remove irrelated feature
        if mask is not None:
            mask = F.interpolate(mask[:, None, ...], size=(gh, gw)).squeeze(
                1
            )  # [N,H,W] -> [N,grid,grid]
            mask = (mask > 0.5).reshape(mask.shape[0], -1)
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
            if x.size()[0] == 1:
                x = x.expand(mask.shape[0], c, gh, gw)

        x = x.reshape(x.shape[0], c, gh * gw).permute(2, 0, 1)  # NCHW -> (HW)NC

        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        positional_embedding = self.positional_embedding
        if not (self.positional_embedding.shape[0] == x.shape[0]):
            cls_pos = positional_embedding[0:1, :]
            per_pos_embedding = (
                F.interpolate(
                    positional_embedding[1:, :]
                    .permute(1, 0)
                    .view(1, -1, self.grid_size, self.grid_size),
                    size=(gh, gw),
                    mode="bicubic",
                    align_corners=False,
                )
                .reshape(-1, gh * gw)
                .permute(1, 0)
            )
            positional_embedding = torch.cat([cls_pos, per_pos_embedding])

        x = x + positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        if self.dense_clip:
            x = self.v_proj(x)
            x = self.c_proj(x)
            return x
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
            key_padding_mask=mask,
        )
        return x


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, use_fpn=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.use_fpn = use_fpn

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        # fpn layers
        self.fpn = FPN(
            in_features=["p2", "p3", "p4", "p5"],
            out_channels=512 # the same dimension with text features
        ) if self.use_fpn else None

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        ) 

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, mask: torch.Tensor = None, return_cls=True):
        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3),
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        outputs = {}
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)  # 1/4,1/4
        x = self.layer1(x)  # 1/4,1/4, 256
        outputs["p2"] = x
        x = self.layer2(x)  # 1/8,1/8, 512
        outputs["p3"] = x
        x = self.layer3(x)  # 1/16,1/16, 1024
        outputs["p4"] = x
        x = self.layer4(x)  # 1/32,1/32, 2048
        b, c, gh, gw = x.shape
        x = self.attnpool(x, mask) # (HW+1)NC
        if return_cls:
            return x[0]
        x = x[1:].permute(1, 2, 0).reshape(b, self.output_dim, gh, gw)
        outputs["p5"] = x

        if self.use_fpn:
            x = self.fpn(outputs)  # 1/4,1/4, 512
        else:
            x = outputs["p5"]

        return x.permute(0, 2, 3, 1)


class FPN(nn.Module):
    def __init__(
        self, in_features, out_channels, norm=""
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
        """
        super(FPN, self).__init__()

        self.strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}
        in_channels_per_feature = {"p2": 256, "p3": 512, "p4": 1024, "p5": 512}

        lateral_convs = []
        use_bias = norm == ""
        for k, in_channels in in_channels_per_feature.items():
            lateral_norm = get_norm(norm, out_channels)
            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            weight_init.c2_xavier_fill(lateral_conv)
            lateral_convs.append(lateral_conv)

        output_norm = get_norm(norm, out_channels)
        self.output_conv = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
        )
        weight_init.c2_xavier_fill(self.output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.in_features = tuple(in_features)

    def forward(self, features):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        out_feat = features[self.in_features[0]]
        out_feat = self.lateral_convs[0](out_feat)
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, lateral_conv in enumerate(self.lateral_convs):
            if idx > 0:
                feat_name = self.in_features[idx]
                feat = features[feat_name]
                feat = lateral_conv(feat)
                scale_factor = self.strides[feat_name] / self.strides[self.in_features[0]]
                feat = F.interpolate(
                    feat, scale_factor=scale_factor, mode="bilinear", align_corners=False
                )
                out_feat = out_feat + feat
        del features
        return self.output_conv(out_feat)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, **kwargs):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask, **kwargs
        )[0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = x + self.attention(self.ln_1(x), **kwargs)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, **kwargs):
        for block in self.resblocks:
            x = block(x, **kwargs)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.grid_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        inter_method="bicubic",
        return_cls=True,
    ):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        b, _, gh, gw = x.size()
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        positional_embedding = self.positional_embedding
        if not (self.positional_embedding.shape[0] == x.shape[0]):
            cls_pos = positional_embedding[0:1, :]
            if inter_method in ["bicubic", "bilinear"]:
                per_pos_embedding = (
                    F.interpolate(
                        positional_embedding[1:, :]
                        .permute(1, 0)
                        .view(1, -1, self.grid_size, self.grid_size),
                        size=(gh, gw),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .reshape(-1, gh * gw)
                    .permute(1, 0)
                )
            else:

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, self.grid_size // 2)
                    if gp > gh // 2:
                        right = q
                    else:
                        left = q
                dis = []
                cur = 1
                for i in range(self.grid_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)
                r_ids = [-_ for _ in reversed(dis)]
                if self.grid_size % 2 == 0:
                    y = r_ids + dis
                else:
                    y = r_ids + [0] + dis
                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, self.grid_size // 2)
                    if gp > gw // 2:
                        right = q
                    else:
                        left = q
                dis = []
                cur = 1
                for i in range(self.grid_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)
                r_ids = [-_ for _ in reversed(dis)]
                if self.grid_size % 2 == 0:
                    x = r_ids + [0] + dis[:-1]
                else:
                    x = r_ids + [0] + dis
                dx = np.arange(-gw // 2, gw / 2, 1.0)
                dy = np.arange(-gh // 2, gh / 2, 1.0)
                all_rel_pos_bias = []

                for i in range(positional_embedding.shape[-1]):
                    z = (
                        positional_embedding[1:, i]
                        .view(self.grid_size, self.grid_size)
                        .float()
                        .numpy()
                    )
                    f_cubic = interpolate.interp2d(x, y, z, kind="cubic")
                    all_rel_pos_bias.append(
                        torch.Tensor(f_cubic(dx, dy))
                        .contiguous()
                        .view(-1, 1)
                        .to(positional_embedding.device)
                    )
                per_pos_embedding = torch.cat(all_rel_pos_bias, dim=-1)

            positional_embedding = torch.cat([cls_pos, per_pos_embedding])
        x = x + positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # remove irrelated feature
        if mask is not None:
            mask = F.interpolate(mask[:, None, ...], size=(gh, gw)).squeeze(
                1
            )  # [N,H,W] -> [N,grid,grid]
            mask = (mask > 0.5).reshape(mask.shape[0], -1)
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
            if x.size()[1] == 1:
                x = x.expand(x.size()[0], mask.shape[0], x.size()[2])

        x = self.transformer(x, key_padding_mask=mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if return_cls:
            x = self.ln_post(x[:, 0, :])
        else:
            x = self.ln_post(x[:, 1:, :])

        if self.proj is not None:
            x = x @ self.proj
        if not return_cls:
            x = x.reshape(b, gh, gw, -1)
        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        vision_use_fpn: bool,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                use_fpn=vision_use_fpn,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, **kwargs):
        return self.visual(image.type(self.dtype), **kwargs)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, vision_use_fpn: bool = False):
    # assert False
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width ** 2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        vision_use_fpn,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, False) #not vision_use_fpn)
    return model.eval()

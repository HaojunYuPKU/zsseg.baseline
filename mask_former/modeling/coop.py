import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import interpolate
import clip
from detectron2.utils.comm import get_local_rank, synchronize
from detectron2.modeling.backbone.resnet import BasicBlock
from .clip_utils.clip_helper import CLIP


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.COOP.MODEL_NAME
    url = clip.clip._MODELS[backbone_name]
    rank = get_local_rank()
    if rank == 0:
        # download on rank 0 only
        model_path = clip.clip._download(url)
    synchronize()
    if rank != 0:
        model_path = clip.clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.clip.build_model(state_dict or model.state_dict())
    for param in model.parameters():
        param.requires_grad = False
    model.float()
    return model


class TextPromptLearner(nn.Module):
    def __init__(self, clip_model, n_prefix=1, n_suffix=0, init_prompt=None):
        super().__init__()
        self.n_prefix = n_prefix
        self.n_suffix = n_suffix
        self.dtype = clip_model.dtype
        self.init_prompt = init_prompt
        if init_prompt is not None and '{}' in init_prompt:
            prompts, lengths = clip.tokenize(
                init_prompt.split("{}"), return_length=True
            )
            self.n_prefix = lengths[0] - 2
            self.n_suffix = lengths[1] - 2
            if init_prompt[-1] == ".":
                self.n_suffix -= 1

            with torch.no_grad():
                embedding = clip_model.token_embedding(prompts).type(
                    self.dtype
                )  # 2,77,512
            if self.n_prefix == 0:
                self.prefix_prompt = None
            else:
                self.prefix_prompt = nn.Parameter(
                    embedding[0, 1 : 1 + self.n_prefix, :]
                )
            if self.n_suffix == 0:
                self.suffix_prompt = None
            else:
                self.suffix_prompt = nn.Parameter(
                    embedding[1, 1 : 1 + self.n_suffix, :]
                )
        else:
            prompt_dim = clip_model.text_projection.data.shape[1]
            if self.n_prefix == 0:
                self.prefix_prompt = None
            else:
                vec = torch.empty(self.n_prefix, prompt_dim)
                nn.init.normal_(vec, std=0.02)
                self.prefix_prompt = nn.Parameter(vec)
            if self.n_suffix == 0:
                self.suffix_prompt = None
            else:
                vec = torch.empty(self.n_suffix, prompt_dim)
                nn.init.normal_(vec, std=0.02)
                self.suffix_prompt = nn.Parameter(vec)
            print(f"Prefix {self.n_prefix} and suffix {self.n_suffix}")
        sentence = "X."
        prompt = clip.tokenize(sentence)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(self.dtype)  # 2,77,512
        self.register_buffer("start_signal", embedding[0, :1, :])  # 1,512
        self.register_buffer("dot_signal", embedding[0, 2:3, :])  # 1,512
        self.register_buffer("end_signal", embedding[0, 3:4, :])  # 1,512
        self.register_buffer("pad_signal", embedding[0, 3:4, :])  # 1,512
        self.cat_bucket = {}

        self.token_embedding = clip_model.token_embedding
        self.text_projection = clip_model.text_projection
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

    def forward(self, class_names):
        prefix = [self.start_signal]
        if self.prefix_prompt is not None:
            prefix.append(self.prefix_prompt)
        prefix = torch.cat(prefix)
        suffix = [self.dot_signal, self.end_signal]
        if self.suffix_prompt is not None:
            suffix.insert(0, self.suffix_prompt)
        suffix = torch.cat(suffix)
        # only process those which are not in bucket
        left_class_names = [
            cls_name for cls_name in class_names if cls_name not in self.cat_bucket
        ]
        if len(left_class_names) > 0:
            with torch.no_grad():
                cat_prompts, name_lengths = clip.tokenize(
                    left_class_names, return_length=True
                )
                name_lengths = [
                    n - 2 for n in name_lengths
                ]  # remove start end end prompt
                cat_embeddings = self.token_embedding(
                    cat_prompts.to(prefix.device)
                ).type(self.dtype)
                cat_embeddings = [
                    embedding[1 : 1 + length]
                    for embedding, length in zip(cat_embeddings, name_lengths)
                ]
            self.cat_bucket.update(
                {
                    name: embedding
                    for name, embedding in zip(left_class_names, cat_embeddings)
                }
            )

        lengths = [
            len(prefix) + len(suffix) + len(self.cat_bucket[name])
            for name in class_names
        ]
        embeddings = torch.stack(
            [
                torch.cat(
                    [prefix, self.cat_bucket[name], suffix]
                    + [self.pad_signal.expand(77 - length, -1)]
                )
                for name, length in zip(class_names, lengths)
            ]
        )  # cls,77,512
        # import pdb
        # pdb.set_trace()
        indices = torch.Tensor(lengths).long().to(embeddings.device) - 1
        # align
        # for debug
        # text = clip.tokenize([self.init_prompt.format(class_names[i]) for i in range(80)]).to(prefix.device) #k,77
        # if (torch.abs(text.argmax(dim=-1).to(prefix.device)-indices).max()>0) or (torch.abs(clip_model.token_embedding(text)-embeddings).max()>0):
        #     import pdb
        #     pdb.set_trace()
        #
        return self.get_text_feature(embeddings, indices)

    def get_text_feature(self, x, indices):
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]

        # select from dot embedding
        # if self.dot_feature:
        # indices = indices-1
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), indices] @ self.text_projection
        return x

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """

        repr = f"prefix_prompt:{self.n_prefix},suffix_prompt:{self.n_suffix}\n"
        if self.init_prompt is not None:
            repr = repr + f"[Prompt_Init{self.init_prompt}]"
        else:
            repr = repr + "[Normal_Init(mu=0,std=0.02)]"
        return repr


class SimpleRGBPromptLearner(nn.Module):
    def __init__(self, init_prompt, clip_model):
        super().__init__()
        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).reshape(1, 3, 1, 1)
        )
        self.mask_fill = init_prompt
        if self.mask_fill == "learn_rgb":
            self.rgb_prompt = nn.Parameter(torch.empty(1, 3, 1, 1))
            nn.init.normal_(self.rgb_prompt, mean=0.5)
        elif self.mask_fill == "learn_rgb_zero_init":
            self.rgb_prompt = nn.Parameter(torch.empty(1, 3, 1, 1))
            nn.init.normal_(self.rgb_prompt, std=0.02)
        elif self.mask_fill == "learn_rgb_patch":
            self.rgb_prompt = nn.Parameter(torch.empty(1, 3, 4, 4))
            nn.init.normal_(self.rgb_prompt, mean=0.5)
        elif self.mask_fill == "zero":
            self.register_buffer("rgb_prompt", torch.zeros(1, 3, 1, 1))
        elif self.mask_fill == "mean":
            self.register_buffer(
                "rgb_prompt", torch.Tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1)
            )
        elif self.mask_fill == "none":
            self.rgb_prompt = None
        else:
            raise NotImplementedError()
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype

    def forward(self, image, fg_mask=None):
        # normalize to 0~1
        if isinstance(image, (tuple,list)):
            image = [i / 255.0 for i in image]
        else:
            image = image / 255.0

        if (fg_mask is not None) and (self.rgb_prompt is not None):

            if not isinstance(image, (list,tuple)):
                if self.mask_fill == "learn_rgb_patch":
                    _, h, w = fg_mask.shape
                    ph, pw = self.rgb_prompt.shape[-2:]
                    rh = (h + ph) // ph
                    rw = (w + pw) // pw
                    fill_value = (
                        self.rgb_prompt[:, :, None, :, None, :]
                        .expand(1, -1, rh, ph, rw, pw)
                        .reshape(1, -1, ph * rh, pw * rw)[..., :h, :w]
                    )
                else:
                    fill_value = self.rgb_prompt
                image = (
                    image * fg_mask[:, None, :, :].float()
                    + (1 - fg_mask[:, None, :, :].float()) * fill_value
                )
            else:

                def _mask_fill(patch, patch_mask):
                    # not mask out
                    if self.mask_fill == "learn_rgb_patch":
                        pattern = self.rgb_prompt
                        _, ph, pw = pattern.shape
                        _, h, w = patch.shape
                        rh = (h + ph) // ph
                        rw = (w + pw) // pw
                        fill_value = (
                            pattern[:, None, :, None, :]
                            .expand(-1, rh, ph, rw, pw)
                            .reshape(-1, ph * rh, pw * rw)[:, :h, :w]
                        )
                    else:
                        fill_value = self.rgb_prompt
                    return (
                        patch * patch_mask.float()
                        + (1 - patch_mask.float()) * fill_value[0]
                    )

                image = [_mask_fill(x, xm) for x, xm in zip(image, fg_mask)]
                image = torch.cat([F.interpolate(x.unsqueeze(0), (224, 224)) for x in image])

        image = (image - self.pixel_mean) / self.pixel_std
        image_features = self.visual(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features


class FeaturePromptLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.background_embedding = nn.Parameter(
            torch.empty(1, self.conv1.weight.shape[0], 1, 1)
        )
        nn.init.normal_(self.background_embedding, std=0.02)
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

        self.grid_size = clip_model.visual.grid_size

        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).reshape(1, 3, 1, 1)
        )

        self.dtype = clip_model.dtype

    def forward(self, image, fg_mask=None):
        # normalize to 0~1
        image = image / 255.0
        image = (image - self.pixel_mean) / self.pixel_std

        image_features = self.forward_features(
            image.type(self.dtype), mask=fg_mask.type(self.dtype)
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward_features(self, x, mask=None, inter_method="bicubic"):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        b, _, gh, gw = x.size()
        # add mask embedding
        downsampled_mask = F.interpolate(
            mask[:, None, ...], size=(gh, gw), mode="bilinear"
        )
        downsampled_mask = (downsampled_mask>0.5).float()
        x = x * downsampled_mask + (1 - downsampled_mask) * self.background_embedding
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
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x

class LightNetworkPromptLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

        self.grid_size = clip_model.visual.grid_size

        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).reshape(1, 3, 1, 1)
        )

        self.dtype = clip_model.dtype
        self.build_transform_layer()
    
    def forward(self, image, fg_mask=None):
        # normalize to 0~1
        image = image / 255.0
        image = (image - self.pixel_mean) / self.pixel_std

        image_features = self.forward_features(
            image.type(self.dtype), mask=fg_mask.type(self.dtype)
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward_features(self, x, mask=None, inter_method="bicubic"):
        x = self.light_conv(torch.cat([x,mask.unsqueeze(1)],dim=1))
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
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x
    def build_transform_layer(self):
        self.light_conv = nn.Conv2d(4,3,1,bias=False)

class LightNetworkV2PromptLearner(LightNetworkPromptLearner):
    def build_transform_layer(self):
        self.light_conv = nn.Sequential(
        nn.Conv2d(4,16,1),
        nn.ReLU(),
        nn.Conv2d(16,3,1)  
        )
class LightNetworkV3PromptLearner(LightNetworkPromptLearner):
    def build_transform_layer(self):
        self.light_conv = nn.Sequential(
        nn.Conv2d(4,16,3,padding=1),
        BasicBlock(16,16),
        nn.Conv2d(16,3,3,padding=1),
        )
class CustomCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        clip_model = load_clip_to_cpu(cfg)
        self.prompt_learner = TextPromptLearner(
            clip_model,
            cfg.MODEL.COOP.N_PREFIX,
            cfg.MODEL.COOP.N_SUFFIX,
            init_prompt=cfg.MODEL.COOP.CTX_INIT,
        )
        self.logit_scale = clip_model.logit_scale
        # RGB Prompt
        if cfg.MODEL.COOP.IMAGE_PROMPT_METHOD == "rgb":
            self.rgb_prompt_learner = SimpleRGBPromptLearner(
                cfg.MODEL.COOP.INPUT_MASK_FILL, clip_model=clip_model
            )
        elif cfg.MODEL.COOP.IMAGE_PROMPT_METHOD == "patch_embedding":
            self.rgb_prompt_learner = FeaturePromptLearner(clip_model)
        elif cfg.MODEL.COOP.IMAGE_PROMPT_METHOD == "light_network":
            self.rgb_prompt_learner = LightNetworkPromptLearner(clip_model)
        elif cfg.MODEL.COOP.IMAGE_PROMPT_METHOD == "light_network_v2":
            self.rgb_prompt_learner = LightNetworkV2PromptLearner(clip_model)
        elif cfg.MODEL.COOP.IMAGE_PROMPT_METHOD == "light_network_v3":
            self.rgb_prompt_learner = LightNetworkV3PromptLearner(clip_model)
        else:
            raise NotImplementedError()
        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).reshape(1, 3, 1, 1)
        )
        self.text_features = {}
        self.fix_text_prompt = False
        if cfg.MODEL.COOP.FIX_TEXTPROMPT:
            self.prompt_learner.eval()
            for param in self.prompt_learner.parameters():
                param.requires_grad = False
            self.fix_text_prompt = True
        self.with_background_embedding = cfg.MODEL.CLIP.WITH_BACKGROUND
        self.embed_dim = clip_model.text_projection.shape[-1]
        if self.with_background_embedding:
            self.background_features = nn.Parameter(torch.empty(1, self.embed_dim))
            # TODO: how to normalize it?
            nn.init.normal_(
                self.background_features, std=clip_model.transformer.width ** -0.5
            )

    def forward(self, image, class_names, mask=None):
        text_features = self.get_text_features(class_names)

        image_features = self.get_visual_features(image, mask=mask)
        # Scale is proper ?
        return self.text_cosine_similarity(image_features, text_features, norm=False)

    def get_text_features(self, class_names):
        if (not self.prompt_learner.training) or self.fix_text_prompt:
            left_class_names = [
                name for name in class_names if name not in self.text_features
            ]
            if len(left_class_names) > 0:
                text_features = self.prompt_learner(left_class_names)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                for k, v in zip(left_class_names, text_features):
                    self.text_features[k] = v
            text_features = torch.stack(
                [self.text_features[name] for name in class_names]
            )
        else:
            # clean buffer
            self.text_features = {}
            text_features = self.prompt_learner(class_names)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_visual_features(self, image, mask=None):
        image_features = self.rgb_prompt_learner(image, mask)
        return image_features

    def text_cosine_similarity(self, features, text_features=None, norm=True, **kwargs):
        if norm:
            features = features / features.norm(dim=-1, keepdim=True)
        
        proj_kernel = text_features
        temperature = self.logit_scale.exp()
        if self.with_background_embedding:
            background_features = (
                self.background_features / self.background_features.norm(dim=-1)
            )
            proj_kernel = torch.cat([proj_kernel, background_features], dim=0)  # C+1,D
        return temperature * features @ proj_kernel.T

    @property
    def device(self):
        return self.pixel_mean.device

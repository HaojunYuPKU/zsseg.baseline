from typing import List, Union

import clip
import numpy as np
import torch
import logging
from torch import nn
from torch.nn import functional as F
from detectron2.utils.comm import get_local_rank, synchronize
from detectron2.utils.logger import log_first_n


class CLIP:
    PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
    PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)
    IMAGENET_TOKEN = [
        "a bad photo of a {}.",
        "a photo of many {}.",
        "a sculpture of a {}.",
        "a photo of the hard to see {}.",
        "a low resolution photo of the {}.",
        "a rendering of a {}.",
        "graffiti of a {}.",
        "a bad photo of the {}.",
        "a cropped photo of the {}.",
        "a tattoo of a {}.",
        "the embroidered {}.",
        "a photo of a hard to see {}.",
        "a bright photo of a {}.",
        "a photo of a clean {}.",
        "a photo of a dirty {}.",
        "a dark photo of the {}.",
        "a drawing of a {}.",
        "a photo of my {}.",
        "the plastic {}.",
        "a photo of the cool {}.",
        "a close-up photo of a {}.",
        "a black and white photo of the {}.",
        "a painting of the {}.",
        "a painting of a {}.",
        "a pixelated photo of the {}.",
        "a sculpture of the {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a plastic {}.",
        "a photo of the dirty {}.",
        "a jpeg corrupted photo of a {}.",
        "a blurry photo of the {}.",
        "a photo of the {}.",
        "a good photo of the {}.",
        "a rendering of the {}.",
        "a {} in a video game.",
        "a photo of one {}.",
        "a doodle of a {}.",
        "a close-up photo of the {}.",
        "a photo of a {}.",
        "the origami {}.",
        "the {} in a video game.",
        "a sketch of a {}.",
        "a doodle of the {}.",
        "a origami {}.",
        "a low resolution photo of a {}.",
        "the toy {}.",
        "a rendition of the {}.",
        "a photo of the clean {}.",
        "a photo of a large {}.",
        "a rendition of a {}.",
        "a photo of a nice {}.",
        "a photo of a weird {}.",
        "a blurry photo of a {}.",
        "a cartoon {}.",
        "art of a {}.",
        "a sketch of the {}.",
        "a embroidered {}.",
        "a pixelated photo of a {}.",
        "itap of the {}.",
        "a jpeg corrupted photo of the {}.",
        "a good photo of a {}.",
        "a plushie {}.",
        "a photo of the nice {}.",
        "a photo of the small {}.",
        "a photo of the weird {}.",
        "the cartoon {}.",
        "art of the {}.",
        "a drawing of the {}.",
        "a photo of the large {}.",
        "a black and white photo of a {}.",
        "the plushie {}.",
        "a dark photo of a {}.",
        "itap of a {}.",
        "graffiti of the {}.",
        "a toy {}.",
        "itap of my {}.",
        "a photo of a cool {}.",
        "a photo of a small {}.",
        "a tattoo of the {}.",
    ]


class PromptLeaner(nn.Module):
    def __init__(self, clip_model, n_prefix=1, n_suffix=0, init_prompt=None):
        super().__init__()
        self.n_prefix = n_prefix
        self.n_suffix = n_suffix
        self.dtype = clip_model.dtype
        self.init_prompt = init_prompt
        if init_prompt is not None:    
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
                self.prefix_prompt = nn.Parameter(embedding[0, 1 : 1 + self.n_prefix, :])
            if self.n_suffix == 0:
                self.suffix_prompt = None
            else:
                self.suffix_prompt = nn.Parameter(embedding[1, 1 : 1 + self.n_suffix, :])
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
            

        sentence = "X."
        prompt = clip.tokenize(sentence)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(self.dtype)  # 2,77,512
        self.register_buffer("start_signal", embedding[0, :1, :])  # 1,512
        self.register_buffer("dot_signal", embedding[0, 2:3, :])  # 1,512
        self.register_buffer("end_signal", embedding[0, 3:4, :])  # 1,512
        self.register_buffer("pad_signal", embedding[0, 3:4, :])  # 1,512
        self.cat_bucket = {}

    def forward(self, class_names,clip_model):
        prefix = [self.start_signal]
        if self.prefix_prompt is not None:
            prefix.append(self.prefix_prompt)
        prefix = torch.cat(prefix)
        suffix = [self.dot_signal,self.end_signal]
        if self.suffix_prompt is not None:
            suffix.insert(0,self.suffix_prompt)
        suffix = torch.cat(suffix)
        # only process those which are not in bucket
        left_class_names = [
            cls_name for cls_name in class_names if cls_name not in self.cat_bucket
        ]
        if len(left_class_names)>0:
            with torch.no_grad():
                cat_prompts, name_lengths = clip.tokenize(
                    left_class_names, return_length=True
                )
                name_lengths = [n - 2 for n in name_lengths]  # remove start end end prompt
                cat_embeddings = clip_model.token_embedding(cat_prompts.to(prefix.device)).type(clip_model.dtype)
                cat_embeddings = [
                    embedding[1 : 1 + length] for embedding, length in zip(cat_embeddings,name_lengths)
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
        #align
        # for debug
        # text = clip.tokenize([self.init_prompt.format(class_names[i]) for i in range(80)]).to(prefix.device) #k,77
        # if (torch.abs(text.argmax(dim=-1).to(prefix.device)-indices).max()>0) or (torch.abs(clip_model.token_embedding(text)-embeddings).max()>0):
        #     import pdb
        #     pdb.set_trace()
        #
        return self.get_text_feature(embeddings, indices, clip_model)

    @staticmethod
    def get_text_feature(x, indices, clip_model):
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        
        #select from dot embedding
        #if self.dot_feature:
        #indices = indices-1
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), indices] @ clip_model.text_projection
        return x

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """

        repr = f'prefix_prompt:{self.n_prefix},suffix_prompt:{self.n_suffix}\n'
        if self.init_prompt is not None:
            repr = repr + f"[Prompt_Init{self.init_prompt}]"
        else:
            repr = repr + "[Normal_Init(mu=0,std=0.02)]"
        return repr

class PromptTuningClipHelper(nn.Module):
    def __init__(
        self,
        model: str,
        temperature: int = 100,
        n_prefix=1,
        n_suffix=0,
        n_prompt=1,
        init_prompt=None,
        freeze=False,
        with_background_embedding: bool = False
    ):
        super().__init__()
        rank = get_local_rank()
        if rank == 0:
            # download on rank 0 only
            model, _ = clip.load(model, device="cpu")
        synchronize()
        if rank != 0:
            model, _ = clip.load(model, device="cpu")
        synchronize()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.temperature = temperature
        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).view(1, -1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).view(1, -1, 1, 1), False
        )
        if init_prompt is None:
            self.prompt_leaner = nn.ModuleList(
                [
                    PromptLeaner(self.model, n_prefix, n_suffix)
                    for _ in range(n_prompt)
                ]
            )
        elif isinstance(init_prompt, str):
            assert "{}" in init_prompt
            self.prompt_leaner = nn.ModuleList(
                [PromptLeaner(self.model, init_prompt=init_prompt)]
            )
        elif isinstance(init_prompt, (int, list, tuple)):
            if isinstance(init_prompt, int):
                init_prompt = CLIP.IMAGENET_TOKEN[:init_prompt]
            elif len(init_prompt) == 2:
                init_prompt = CLIP.IMAGENET_TOKEN[init_prompt[0] : init_prompt[1]]
            else:
                init_prompt = [CLIP.IMAGENET_TOKEN[i] for i in init_prompt]
            self.prompt_leaner = nn.ModuleList(
                [PromptLeaner(self.model, init_prompt=prompt) for prompt in init_prompt]
            )
        else:
            raise NotImplementedError()
        self.freeze = freeze
        if freeze:
            for param in self.prompt_leaner.parameters():
                param.requires_grad=False
            self.prompt_leaner.eval()
        self.with_background_embedding = with_background_embedding
        self.embed_dim = model.text_projection.shape[-1]
        if with_background_embedding:
            self.background_features = nn.Parameter(torch.empty(1, self.embed_dim))
            # TODO: how to normalize it?
            nn.init.normal_(
                self.background_features, std=self.model.transformer.width ** -0.5
            )
    @torch.no_grad()
    def get_frozen_text_features(
        self,
        noun_list,
        text_tokens: Union[str, List[str]] = CLIP.IMAGENET_TOKEN,
        post_norm=True,
    ):
        if not isinstance(text_tokens, list):
            # ensemble
            text_tokens = [text_tokens]

        text_features_bucket = []
        for style in text_tokens:
            noun_tokens = [clip.tokenize(style.format(c)) for c in noun_list]
            text_inputs = torch.cat(noun_tokens).to(self.device)
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_bucket.append(text_features)
        del text_inputs
        # ensemble by averaging
        text_features = torch.stack(text_features_bucket).mean(dim=0)
        if post_norm:
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)  # Cls,C

        return text_features

    def get_text_features(
        self,
        noun_list,
        post_norm=True,
    ):

        text_features = torch.stack(
            [leaner(noun_list, self.model) for leaner in self.prompt_leaner]
        )
        text_features =text_features/ text_features.norm(dim=-1, keepdim=True)  # k,Cls,C
        text_features = text_features.mean(dim=0)
        if post_norm:
            text_features =text_features/ text_features.norm(dim=-1, keepdim=True)  # Cls,C
        self.text_features = text_features

        return self.text_features

    def text_cosine_similarity(self, features, text_features=None, norm=True, **kwargs):
        if norm:
            features = features / features.norm(dim=-1, keepdim=True)
        proj_kernel = text_features
        if proj_kernel is None:
            proj_kernel = self.text_features
        if self.with_background_embedding:
            background_features = (
                self.background_features / self.background_features.norm(dim=-1)
            )
            proj_kernel = torch.cat([proj_kernel, background_features], dim=0)  # C+1,D
        temperature = kwargs.get("temperature", self.temperature)
        return temperature * features @ proj_kernel.T[None, ...]

    def get_visual_features(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 100,
        norm=True,
    ):

        image_features = []
        num_batches = (len(images) + 0.5 * batch_size) // batch_size
        num_batches = max(num_batches, 1)
        torch.cuda.empty_cache()
        for i in range(int(num_batches)):
            start = i * batch_size
            end = (i + 1) * batch_size
            if i == num_batches - 1:
                end = len(images)

            if isinstance(images, torch.Tensor):
                image_features.append(self.model.encode_image(images[start:end]))
            else:
                image_features.append(
                    torch.cat(
                        [
                            self.model.encode_image(_img[None, ...])
                            for _img in images[start:end]
                        ]
                    )
                )
        image_features = torch.cat(image_features)
        if norm:
            image_features =image_features/image_features.norm(dim=-1, keepdim=True)

        return image_features
    @property
    def device(self):
        return self.pixel_mean.device
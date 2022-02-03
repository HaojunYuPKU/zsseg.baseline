from typing import List, Union

import clip
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.utils.comm import get_local_rank, synchronize
from .mask_utils import crop_with_mask

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


class ClipHelper(nn.Module):
    def __init__(
        self,
        model: str,
        fix_size_input: bool = True,
        background_fill = (int(x) for x in CLIP.PIXEL_MEAN),
        mask_expand_ratio: int = 1.0,
        thr: float = 0.5,
        temperature: int = 100,
        with_background: bool = True,
    ):
        super().__init__()
        rank = get_local_rank()
        if rank == 0:
            #download on rank 0 only
            model, _ = clip.load(model, device="cpu")
        synchronize()
        if rank != 0:
            model, _ = clip.load(model, device="cpu")
        synchronize()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.fix_size_input = fix_size_input
        if self.fix_size_input:
            self.input_size = (224,224)
        
        self.background_fill = background_fill
        self.mask_expand_ratio = mask_expand_ratio
        self.thr = thr
        self.temperature = temperature
        self.embed_dim = model.text_projection.shape[-1]
        self.with_background = with_background
        if self.with_background:
            self.background_features = nn.Parameter(torch.empty(1, self.embed_dim))
            # TODO: how to normalize it?
            nn.init.normal_(
                self.background_features, std=self.model.transformer.width ** -0.5
            )
        self.text_features = None
        self.register_buffer("pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).view(1,-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(CLIP.PIXEL_STD).view(1,-1, 1, 1), False)

    @torch.no_grad()
    def get_text_features(
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
            text_features /= text_features.norm(dim=-1, keepdim=True)  # Cls,C

        self.text_features = text_features
        return self.text_features

    def text_cosine_similarity(self, features, text_features=None,norm=True, **kwargs):
        if norm:
            features = features / features.norm(dim=-1, keepdim=True)
        proj_kernel = text_features
        if proj_kernel is None:
            proj_kernel = self.text_features

        if self.with_background:
            background_features = (
                self.background_features / self.background_features.norm(dim=-1)
            )
            proj_kernel = torch.cat([proj_kernel, background_features], dim=0)  # C+1,D
        temperature = kwargs.get("temperature", self.temperature)
        return temperature * features @ proj_kernel.T[None, ...]

    def get_visual_features(
        self, images: Union[torch.Tensor, List[torch.Tensor]], batch_size: int = 100,norm=True
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
            
            if isinstance(images,torch.Tensor):
                image_features.append(self.model.encode_image(images[start:end]))
            else:
                image_features.append(torch.cat([self.model.encode_image(_img[None,...]) for _img in images[start:end]]))
        image_features = torch.cat(image_features)
        if norm:
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features

    def crop_and_resize_image(self,images,masks):
        patches = [crop_with_mask(image,mask.squeeze(0),fill=(0,0,0),expand_ratio=1.0,thr=None) for image,mask in zip(images,masks)]
        #resize and preprocess
        if self.fix_size_input:
            patches = [F.interpolate(patch[None],size=self.input_size) for patch in patches]
            if len(patches)==0:
                return None
            patches = torch.cat(patches)
            # normalize
            patches = (patches/255. - self.pixel_mean) / self.pixel_std
        else:
            patches = [(p/255-self.pixel_mean[0])/self.pixel_std[0] for p in patches]
        
        return patches

    @property
    def device(self):
        return self.pixel_mean.device
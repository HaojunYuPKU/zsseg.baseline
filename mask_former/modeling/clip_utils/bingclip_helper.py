#
from typing import List, Union
from transformers import RobertaTokenizer
from clip import dense_clip
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import logging
from detectron2.utils.logger import log_first_n
from .clip_helper import CLIP
from .prompt_learner import RobertaPromptLearner, MultiPromptLearners


def get_model(weights, text, vision, eval_mode=True):
    checkpoint = torch.load(weights, map_location="cpu")
    args = checkpoint["args"]
    model = dense_clip.MMModel(args)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    log_first_n(logging.INFO, "Missing Keys in CLIP Model: {}".format(missing))
    log_first_n(logging.INFO, "Unexpected Keys in CLIP Model: {}".format(unexpected))
    if eval_mode:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    if not text:
        del model.sentence_model
        model.sentence_model = None
    if not vision:
        del model.visual_model
        model.visual_model = None
    else:
        model.visual_model.backbone.return_no_pool = True
    return model


class BingClipHelper(nn.Module):
    def __init__(
        self,
        weights,
        temperature: int = 100,
        text=True,
        vision=True,
        eval_mode = True,
        init_with_embedding=True,
        prompt_num = None, 
        left_prompt_length=4
    ):
        super().__init__()
        

        self.temperature = temperature
        self.text_features = None
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = get_model(weights, text, vision,eval_mode=eval_mode)
        prompt_num = len(CLIP.IMAGENET_TOKEN) if prompt_num is None else prompt_num
        if init_with_embedding:
            self.prompt_learner = MultiPromptLearners(
                [
                    RobertaPromptLearner(
                        model.sentence_model.backbone, init_prompt=template,tokenizer=tokenizer
                    )
                    for template in CLIP.IMAGENET_TOKEN[:prompt_num]
                ]
            )
        else:
            self.prompt_learner = MultiPromptLearners(
                [
                    RobertaPromptLearner(
                        model.sentence_model.backbone, prompt_dim=768,left_prompt_length=left_prompt_length,right_prompt_length=0,tokenizer=tokenizer
                    )
                    for i in range(prompt_num)
                ]
            )
        self.projector = model.sentence_model.projector

    def get_text_features(
        self,
        noun_list,
        post_norm=True,
        device="cpu",
        ensemble=True
    ):
        self.device = device
        #debug
        #! Revert to 21dc6071d2a03d1743d61b33f624f67d27473aec if any problem!!!
        text_features = self.prompt_learner(noun_list)
        text_features = self.projector(text_features)
        text_features =text_features / text_features.norm(dim=-1,keepdim=True) #K,Cls,C
        if ensemble:
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

        # if self.with_background:
        #     background_features = (
        #         self.background_features / self.background_features.norm(dim=-1)
        #     )
        #     proj_kernel = torch.cat([proj_kernel, background_features], dim=0)  # C+1,D
        temperature = kwargs.get("temperature", self.temperature)
        if proj_kernel.dim()==3:
            return [temperature * features@ proj_kernel[i].t() for i in range(len(proj_kernel))]
        else:
            return temperature * features @ proj_kernel.t()

    @torch.no_grad()
    def encode_image(self, image):
        image = image / 255.0
        image = (
            image - image.new_tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1)
        ) / image.new_tensor(CLIP.PIXEL_STD).reshape(1, 3, 1, 1)
        return self.model.encode_image(image)

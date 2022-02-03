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


def get_model(weights,text,vision,eval_mode=True):
    checkpoint = torch.load(weights,map_location="cpu")
    args =checkpoint["args"]
    model = dense_clip.MMModel(args)
    missing,unexpected = model.load_state_dict(checkpoint["model"],strict=False)
    log_first_n(logging.INFO,"Missing Keys in CLIP Model: {}".format(missing))
    log_first_n(logging.INFO,"Unexpected Keys in CLIP Model: {}".format(unexpected))
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
        text = True,
        vision = True
    ):
        super().__init__()
        self.model = get_model(weights,text,vision)
     
        self.temperature = temperature
        self.text_features = None
        
    @torch.no_grad()
    def get_text_features(
        self,
        noun_list,
        templates: Union[str, List[str]] = CLIP.IMAGENET_TOKEN,
        post_norm=True,
        device="cpu",
    ):
        self.device = device
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        with torch.no_grad():
            text_features = []
            for classname in noun_list:
                texts = [template.format(classname) for template in templates] #format with class
                texts = tokenizer(texts, return_tensors='pt', padding=True).to(self.device) #tokenize
                with torch.no_grad():
                    class_embeddings = self.model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                text_features.append(class_embedding)
            text_features = torch.stack(text_features, dim=0).to(self.device)
        if post_norm:
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)  # Cls,C
        self.text_features = text_features
        return self.text_features


    def text_cosine_similarity(self, features, text_features=None,norm=True, **kwargs):
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
        return temperature * features @ proj_kernel.t()

    @torch.no_grad()
    def encode_image(self,image):
        image =image/255.
        image = (image - image.new_tensor(CLIP.PIXEL_MEAN).reshape(1,3,1,1))/ image.new_tensor(CLIP.PIXEL_STD).reshape(1,3,1,1)
        return self.model.encode_image(image)
    @property
    def pixel_mean(self):
        return CLIP.PIXEL_MEAN
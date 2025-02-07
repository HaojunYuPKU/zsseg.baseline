# Copyright (c) Facebook, Inc. and its affiliates.
import os
import logging

import torch
from torch import nn
from torch.nn import functional as F


from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
# from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling import SemanticSegmentor
from ..modeling.clip_adapter import (
    PerPixelClipAdapter,
    PredefinedPromptExtractor,
    ImageNetPromptExtractor,
    VILDPromptExtractor,
    LearnablePromptExtractor,
    build_prompt_learner
)


@META_ARCH_REGISTRY.register()
class ClipOnlyPerPixelModel(SemanticSegmentor):
    @configurable
    def __init__(
        self,
        clip_adapter,
        use_gt_categories,
        common_stride,
        ignore_value=-1,
        loss_weight=1.0,
        **kwargs,
    ):
        super(ClipOnlyPerPixelModel, self).__init__(**kwargs)
        self.clip_adapter = clip_adapter
        self.use_gt_categories = use_gt_categories
        self.ignore_value = ignore_value
        self.loss_weight = loss_weight
        self.common_stride = common_stride
        self.size_divisibility = 32
        self.unfreeze_model()

    @classmethod
    def from_config(cls, cfg):
        # backbone = build_backbone(cfg)
        # sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER)
        clip_adapter = PerPixelClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
            vision_use_fpn=cfg.MODEL.CLIP_ADAPTER.USE_FPN,
        )
        return {
            "clip_adapter": clip_adapter,
            "use_gt_categories": cfg.MODEL.CLIP_ADAPTER.USE_GT_CATEGORIES,
            "common_stride": cfg.MODEL.CLIP_ADAPTER.COMMON_STRIDE,
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "backbone": None,
            "sem_seg_head": None,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def unfreeze_model(self):
        for param in self.clip_adapter.clip_model.visual.parameters():
            param.requires_grad = True

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0]
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.training:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.size_divisibility, self.ignore_value
            ).tensor
        else:
            targets = None
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        if self.training:
            text_features = self.clip_adapter.get_text_features(class_names)
            image_features = self.clip_adapter.get_image_features(images.tensor, per_pixel=True)
            targets = {
                "sem_seg_target": targets,
                "text_features": text_features,
                "cosine_sim_func": self.clip_adapter.get_sim_logits,
            }
            return self.losses(image_features, targets)

        processed_results = []
        for input_per_image, image, image_size in zip(
            batched_inputs, images.tensor, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            # image = input_per_image["image"]  # C,H,W
            clip_act_map = (
                self.clip_adapter(image[None], class_names, per_pixel=True)
                .softmax(dim=-1)
                .permute(0, 3, 1, 2)
                .squeeze(0)
            )  # cls,gh,gw
            if self.use_gt_categories:
                assert "sem_seg" in input_per_image
                target = input_per_image["sem_seg"]
                gt_categories = target.unique().tolist()
                gt_categories = list(
                    set(gt_categories) - set([self.ignore_value]))
                r = torch.zeros_like(clip_act_map)
                r[gt_categories] = clip_act_map[gt_categories]
                del clip_act_map
            else:
                r = clip_act_map

            pad_size = tuple(images.tensor.shape[-2:])
            r = self.sem_seg_postprocess(
                r, pad_size, image_size, height, width
            )
            processed_results.append({"sem_seg": r})
        return processed_results

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names

    def losses(self, predictions, targets):
        predictions = targets["cosine_sim_func"](
            targets["text_features"], predictions
        ).permute(0, 3, 1, 2)
        targets = targets["sem_seg_target"]
        predictions = (
            predictions.float()
        )  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses

    def sem_seg_postprocess(self, result, pad_size, img_size, output_height, output_width):
        result = F.interpolate(
            result[None], size=pad_size, mode="bilinear", align_corners=False
        )
        result = result[:, :, : img_size[0], : img_size[1]]
        result = F.interpolate(
            result, size=(output_height, output_width), mode="bilinear", align_corners=False
        )
        return result[0]

# Copyright (c) Facebook, Inc. and its affiliates.
import os
import logging
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from .modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    build_prompt_learner,
)
from .mask_former_model import MaskFormer
from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.embedding_criterion import EmbeddingCriterion
from .utils.post_process_utils import dense_crf_post_process
from .utils.graph_cut import graph_cut_post_process


@META_ARCH_REGISTRY.register()
class ZeroShotMaskFormerV3(MaskFormer):
    """
    Main class for zero shot mask classification semantic segmentation architectures.
    """
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        region_clip_adapter: nn.Module = None,
        criterion: nn.Module,
        embed_criterion: nn.Module,
        num_queries: int,
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        with_graph_cut_post_process,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.clip_adapter: ClipAdapter = clip_adapter
        self._region_clip_adapter = region_clip_adapter

        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight

        self.embed_criterion = embed_criterion

        self.graph_cut_post_process = with_graph_cut_post_process
        self.debug = False

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        embed_weight = cfg.MODEL.MASK_FORMER.EMBED_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=1,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )

        weight_dict = {
            "loss_ce": 1, 
            "loss_mask": mask_weight, 
            "loss_dice": dice_weight,
            "loss_embed": embed_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
        )
        embed_criterion = EmbeddingCriterion(
            total_sample_num=cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.TOTAL_SAMPLE_NUM,
            sample_method=cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.SAMPLE_METHOD,
            temperature=cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.TEMPERATURE,
            weight=weight_dict["loss_embed"],
            per_image=cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.PER_IMAGE,
        )
        
        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER)
        region_clip_adapter = None
        if cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER:
            log_first_n(
                logging.WARNING,
                "Using different head for region classification and query classification",
            )
            cls_prompt_learner = build_prompt_learner(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER
            )
            region_clip_adapter = MaskFormerClipAdapter(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME,
                cls_prompt_learner,
                mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
                mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
                mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
                mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
                region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            )

        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "embed_criterion": embed_criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_adapter": clip_adapter,
            "region_clip_adapter": region_clip_adapter,
            "clip_ensemble": cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE,
            "clip_ensemble_weight": cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT,
            "with_graph_cut_post_process": cfg.MODEL.V3_TEST.WITH_GRAPH_CUT_POST_PROCESS,
        }

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
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        class_names = self.get_class_name_list(dataset_name)
        text_features = self.clip_adapter.get_text_features(class_names)
        outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
            text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
        )
        if self.training:
            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i][
                        "pred_logits"
                    ] = self.clip_adapter.get_sim_logits(
                        text_features,
                        self.clip_adapter.normalize_feature(
                            outputs["aux_outputs"][i]["pred_logits"]
                        ),
                    )
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)
            sem_seg_targets = self.prepare_sem_seg_targets(targets)
            sem_seg_targets["meta"] = MetadataCatalog.get(dataset_name)
            losses.update(self.embed_criterion(outputs, sem_seg_targets))

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            pix_embedding = F.interpolate(
                outputs["pix_embedding"]
                / outputs["pix_embedding"].norm(dim=1, keepdim=True),
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            for i, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            )):
                height = image_size[0]
                width = image_size[1]
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )
                image = input_per_image["image"].to(self.device)
                # semantic segmentation inference
                r = self.semantic_inference(
                    mask_cls_result, mask_pred_result, image, class_names, dataset_name
                )
                # graph cut
                if self.graph_cut_post_process:
                    feat_h, feat_w = pix_embedding[i].shape[-2:]
                    emb = sem_seg_postprocess(
                        pix_embedding[i], image_size, height, width
                    )
                    rh, rw = r.shape[-2:]
                    r = F.interpolate(
                        r[None], size=(feat_h, feat_w), mode="bilinear", align_corners=False,
                    )[0]
                    emb = F.interpolate(
                        emb[None], size=(feat_h, feat_w), mode="bilinear", align_corners=False,
                    )[0]
                    r = graph_cut_post_process(
                        logits=r.cpu(), 
                        pix_embedding=emb.cpu().permute(1, 2, 0),
                        logits_temperature=0.1,
                        pix_temperature=0.01,
                    )
                    r = F.one_hot(
                        torch.from_numpy(r).long(), num_classes=len(class_names)
                    ).permute(2, 0, 1).float()
                    r = F.interpolate(
                        r[None], size=(rh, rw), mode="nearest"
                    )[0]
                
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = sem_seg_postprocess(r, image_size, height, width)
                if self.debug:
                    torch.save(
                        sem_seg_postprocess(pix_embedding[i], image_size, height, width)
                        .detach()
                        .cpu()
                        .half(),
                        os.path.join(
                            "output/tmp_embedding_res",
                            os.path.basename(input_per_image["file_name"]).split(".")[0]
                            + ".pth",
                        ),
                    )
                    torch.save(
                        r.detach()
                        .cpu()
                        .half(),
                        os.path.join(
                            "output/tmp_score_res",
                            os.path.basename(input_per_image["file_name"]).split(".")[0]
                            + ".pth",
                        ),
                    )

                processed_results.append({"sem_seg": r})

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = self.panoptic_inference(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

            return processed_results

    def semantic_inference(self, mask_cls, mask_pred, image, class_names, dataset_name):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        # get the classification result from clip model
        if self.clip_ensemble:
            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, mask_pred, normalize=True
            )
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            # softmax before index or after?
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                    1 - trained_mask
                ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                # only clip model predictions are used
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def prepare_targets(self, targets, images, ori_images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for i, targets_per_image in enumerate(targets):
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "image": ori_images[i],
                }
            )
        return new_targets

    def prepare_sem_seg_targets(self, targets, ignore_label=255):
        new_targets = []

        for i, targets_per_image in enumerate(targets):
            _, h, w = targets_per_image["masks"].shape
            new_target = targets_per_image["masks"].new_ones(h, w) * ignore_label
            for m, cls in zip(targets_per_image["masks"], targets_per_image["labels"]):
                new_target[m == 1] = cls
            new_targets.append(new_target)

        return {
            "sem_seg": F.interpolate(
                torch.stack(new_targets).unsqueeze(1).float(), scale_factor=0.25
            )
            .squeeze(1)
            .long(),
            "ignore_label": ignore_label,
            "images": [t["image"] for t in targets],
        }

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names

    @property
    def region_clip_adapter(self):
        if self._region_clip_adapter is None:
            return self.clip_adapter
        return self._region_clip_adapter

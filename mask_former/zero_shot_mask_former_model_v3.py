# Copyright (c) Facebook, Inc. and its affiliates.
import os
import logging

import torch
from torch import nn
from torch.nn import functional as F


from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_every_n, log_first_n
from detectron2.structures import BitMasks, PolygonMasks
from .modeling.clip_utils.mask_utils import expand_box

from .mask_former_model import MaskFormer
from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.coop import CustomCLIP
from .modeling.embedding_criterion import EmbeddingCriterion
from .utils.post_process_utils import dense_crf_post_process


@META_ARCH_REGISTRY.register()
class ZeroShotMaskFormerV3(MaskFormer):
    @configurable
    def __init__(
        self,
        clip_helper,
        embed_criterion,
        freeze_pretrained: bool = True,
        *,
        mask_thr,
        expand_ratios,
        with_dense_crf_post_process,
        dense_crf_feat_std,
        **kwargs,
    ):
        super(ZeroShotMaskFormerV3, self).__init__(**kwargs)
        self.clip_helper = clip_helper
        self.embed_criterion = embed_criterion
        for name, param in self.clip_helper.named_parameters():
            if "background_features" not in name:
                param.requires_grad = False
        self.expand_ratios = expand_ratios
        assert hasattr(self.sem_seg_head, "freeze_pretrained")
        if freeze_pretrained:
            self.sem_seg_head.freeze_pretrained()
        self.mask_thr = mask_thr
        self.dense_crf_post_process = with_dense_crf_post_process
        self.dense_crf_feat_std = dense_crf_feat_std
        # for debug
        names = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                names.append(name)
        while len(names) > 0:
            if len(names) > 20:
                log_first_n(logging.INFO, names[:20], n=100)
                names = names[20:]
            else:
                log_first_n(logging.INFO, names)
                names = []

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

        clip_helper = CustomCLIP(cfg)
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight
            if clip_helper.with_background_embedding
            else 0,
            losses=losses,
        )
        embed_criterion = EmbeddingCriterion(
            total_sample_num=cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.TOTAL_SAMPLE_NUM,
            sample_method=cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.SAMPLE_METHOD,
            temperature=cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.TEMPERATURE,
            weight=weight_dict["loss_embed"],
            per_image=cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.PER_IMAGE,
        )
        return {
            "clip_helper": clip_helper,
            "embed_criterion": embed_criterion,
            "freeze_pretrained": cfg.MODEL.ZERO_SHOT_MASK_FORMER.FREEZE_PRETRAINED,
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
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
            "expand_ratios": cfg.MODEL.CLIP.INPUT_CROP_EXPAND_RATIO,
            "mask_thr": cfg.MODEL.CLIP.INPUT_MASK_THRSHOLD,
            "with_dense_crf_post_process": cfg.MODEL.V3_TEST.WITH_DENSE_CRF_POST_PROCESS,
            "dense_crf_feat_std": cfg.MODEL.V3_TEST.DENSE_CRF_FEAT_STD,
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
        ori_images = images
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        text_features = self.clip_helper.get_text_features(class_names)

        outputs["pred_logits"] = self.clip_helper.text_cosine_similarity(
            outputs["pred_logits"], text_features
        )

        # if "instances" in input_per_image:
        #             #visualize only
        if self.training:
            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i][
                        "pred_logits"
                    ] = self.clip_helper.text_cosine_similarity(
                        outputs["aux_outputs"][i]["pred_logits"], text_features
                    )
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images, ori_images)
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
            self._updated = False
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
            for i, (
                mask_cls_result,
                mask_pred_result,
                input_per_image,
                image_size,
            ) in enumerate(
                zip(
                    mask_cls_results,
                    mask_pred_results,
                    batched_inputs,
                    images.image_sizes,
                )
            ):

                height = image_size[0]
                width = image_size[1]

                # if self.sem_seg_postprocess_before_inference:

                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )
                if float(os.environ.get("WEIGHT", 0.9)) <= 1:
                    # normalize
                    image = input_per_image["image"].to(self.device)
                    valid, regions, region_masks = self.get_regions(
                        mask_pred_result, image, size=(224, 224), expand_ratio=1.0
                    )
                    if regions is not None:

                        clip_cls_result = self.clip_helper(
                            regions, class_names, region_masks
                        )
                        mask_pred_result = mask_pred_result[valid]
                        mask_cls_result = mask_cls_result[valid]
                    else:
                        dataset_name = input_per_image["meta"]["dataset_name"]
                        clip_cls_result = mask_pred_result.new_zeros(
                            mask_pred_result.shape[0], len(class_names)
                        )
                else:
                    clip_cls_result = None
                # semantic segmentation inference

                r = self.semantic_inference(
                    mask_cls_result,
                    clip_cls_result,
                    mask_pred_result,
                    dataset_name,
                    None,
                )
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                # if not self.sem_seg_postprocess_before_inference:
                r = sem_seg_postprocess(r, image_size, height, width)
                # torch.save(
                #     sem_seg_postprocess(pix_embedding[i], image_size, height, width)
                #     .detach()
                #     .cpu()
                #     .half(),
                #     os.path.join(
                #         "output/tmp_embedding_res",
                #         os.path.basename(input_per_image["file_name"]).split(".")[0]
                #         + ".pth",
                #     ),
                # )
                # torch.save(
                #     r.detach()
                #     .cpu()
                #     .half(),
                #     os.path.join(
                #         "output/tmp_score_res",
                #         os.path.basename(input_per_image["file_name"]).split(".")[0]
                #         + ".pth",
                #     ),
                # )
                # densecrf postprocess
                if self.dense_crf_post_process:
                    log_first_n(logging.INFO, "Use densecrf for postprocessing.", n=10)
                    r = dense_crf_post_process(
                        r,
                        sem_seg_postprocess(pix_embedding[i], image_size, height, width)
                        .cpu()
                        .permute(1, 2, 0)
                        .numpy(),
                        len(class_names),
                        bi_rgb_std=self.dense_crf_feat_std,
                    )

                processed_results.append({"sem_seg": r})

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = self.panoptic_inference(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

            return processed_results

    def semantic_inference(
        self, mask_cls, clip_cls, mask_pred, dataset_name, foreground_prob=1
    ):
        if clip_cls is not None:
            clip_cls = F.softmax(clip_cls[..., :-1], dim=-1)
        if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
            trained_mask = torch.Tensor(
                MetadataCatalog.get(dataset_name).trainable_flag
            ).to(mask_cls.device)[None, :]
            mask_cls = F.softmax(mask_cls, dim=-1)

            if self.clip_helper.with_background_embedding:
                unknown_part = mask_cls[..., -1]
                mask_cls = mask_cls[..., :-1]
                import os

                weight = float(os.environ.get("WEIGHT", 0.9))
                if weight > 1:
                    mask_cls = mask_cls
                elif weight < 0:
                    mask_cls = clip_cls
                else:
                    mask_cls = trained_mask * torch.pow(mask_cls, weight) * torch.pow(
                        clip_cls, 1 - weight
                    ) + (1 - trained_mask) * torch.pow(
                        mask_cls, 1 - weight
                    ) * torch.pow(
                        clip_cls, weight
                    )
            else:
                weight = 2 / 3
                mask_cls = trained_mask * torch.pow(mask_cls, weight) * torch.pow(
                    clip_cls, 1 - weight
                ) + (1 - trained_mask) * torch.pow(mask_cls, 1 - weight) * torch.pow(
                    clip_cls, weight
                )
        else:
            mask_cls = clip_cls
        if foreground_prob is not None:
            # TODO: In zero shot od, the prob should be geometric average of two prob and the weight is 2/3,1/3 for base,novel class respectively
            # TODO: In our case, we don't tune it now and may need further study.
            mask_cls = (
                mask_cls * foreground_prob.softmax(dim=-1)[..., : mask_cls.shape[-1]]
            )
        mask_pred = mask_pred.sigmoid()

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

    def get_regions(self, masks, image, size=(224, 224), expand_ratio=1.0):
        masks = masks > self.mask_thr
        # import pdb
        # pdb.set_trace()
        valid = masks.sum(dim=(-1, -2)) > 0
        masks = masks[valid]
        gt_masks = BitMasks(masks)

        gt_bboxes = gt_masks.get_bounding_boxes()

        def _mask_fill(image, rect, mask):
            rect = expand_box(
                *rect,
                expand_ratio=expand_ratio,
                max_h=image.shape[-2],
                max_w=image.shape[-1],
            )
            patch = image[:, rect[1] : rect[3] + 1, rect[0] : rect[2] + 1]
            if isinstance(mask, torch.Tensor):
                patch_mask = (
                    mask[rect[1] : rect[3] + 1, rect[0] : rect[2] + 1]
                    .to(patch.device)
                    .unsqueeze(0)
                )
            elif isinstance(mask, (PolygonMasks, list)):
                mask = BitMasks.from_polygon_masks(
                    [mask], image.shape[1], image.shape[2]
                )
                patch_mask = mask.tensor[
                    :, rect[1] : rect[3] + 1, rect[0] : rect[2] + 1
                ].to(patch.device)
            else:
                patch_mask = mask.tensor[
                    :, rect[1] : rect[3] + 1, rect[0] : rect[2] + 1
                ].to(patch.device)
            return patch, patch_mask

        if len(gt_bboxes) > 0:
            regions, region_masks = list(
                zip(
                    *[
                        _mask_fill(image, bbox, mask)
                        for bbox, mask in zip(gt_bboxes, masks)
                    ]
                )
            )

            return valid, regions, region_masks
        return valid, None, None

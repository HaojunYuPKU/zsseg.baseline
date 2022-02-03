# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_mask_former_default_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    cfg.ORACLE = False
    cfg.PSEUDO = False
    cfg.PSEUDO_WITH_PRIOR = True
    cfg.PSEUDO_FLAG_NAME = "trainable_flag"

    # data config
    # select the dataset mapper
    cfg.DATASETS.SAMPLE_PER_CLASS = -1
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.EXPAND_RATIO = [1.0]
    cfg.INPUT.EXPAND_MODE = "choice"
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1
    # test
    cfg.TEST.SLIDING_WINDOW = False
    cfg.TEST.SLIDING_TILE_SIZE = 224
    cfg.TEST.SLIDING_OVERLAP = 1 / 3
    cfg.TEST.DENSE_CRF = False
    cfg.TEST.PROPOSAL_REFINE = False
    # for clip mask testing
    cfg.TEST.BASE_TRELY_ON_PRED = True

    # cfg.TEST.ENSEMBLE_WEIGHT = 0.9
    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.TEST_IMS_PER_BATCH = 1
    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"
    # embedding learning
    cfg.MODEL.SEM_SEG_HEAD.EMBEDDING_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HIDDEN_DIM = 1024
    cfg.MODEL.SEM_SEG_HEAD.EMBED_LAYERS = 2

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.NORM_INDICES = (0, 1, 2, 3)
    cfg.MODEL.SWIN.PROJECTION = False
    cfg.MODEL.SWIN.PROJECT_DIM = 256
    #
    cfg.MODEL.MASK_FORMER.EMBED_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD = CN()
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.NUM_LAYERS = 3
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.TOTAL_SAMPLE_NUM = 512
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.SAMPLE_METHOD = "uniform"
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.TEMPERATURE = 0.05
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HEAD.PER_IMAGE = False
    # zero shot
    cfg.MODEL.ZERO_SHOT_MASK_FORMER = CN()
    cfg.MODEL.ZERO_SHOT_MASK_FORMER.FREEZE_PRETRAINED = False
    cfg.MODEL.ZERO_SHOT_MASK_FORMER.WITH_DISTILL_CRITERION = False
    cfg.MODEL.ZERO_SHOT_MASK_FORMER.DISTL_WEIGHT = 1.0
    cfg.MODEL.ZERO_SHOT_MASK_FORMER.CALIBRATED = False
    cfg.MODEL.ZERO_SHOT_MASK_FORMER.CALIBRATED_WEIGHT = 1.0
    cfg.MODEL.ZERO_SHOT_MASK_FORMER.CALIBRATED_BIAS = -0.1
    cfg.MODEL.ZERO_SHOT_MASK_FORMER.CALIBRATED_BEFORE_AGG = True
    cfg.MODEL.ZERO_SHOT_MASK_FORMER.CLIP_PRETRAINED = ""
    # proposal
    cfg.MODEL.OFFLINE_PROPOSAL = CN()
    cfg.MODEL.OFFLINE_PROPOSAL.PATH = ""
    # clip model
    cfg.MODEL.CLIP = CN()
    cfg.MODEL.CLIP.MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP.FIX_SIZE = True
    cfg.MODEL.CLIP.TEMPERATURE = 100.0
    cfg.MODEL.CLIP.WITH_BACKGROUND = False
    cfg.MODEL.CLIP.INPUT_MASK_FILL = "mean"
    cfg.MODEL.CLIP.INPUT_CROP_EXPAND_RATIO = (1.0,)
    cfg.MODEL.CLIP.PROMPT_NUM = [2]
    cfg.MODEL.CLIP.PROMPT_Sample_MODE = "range"
    cfg.MODEL.CLIP.INPUT_MASK_THRSHOLD = 0.6
    cfg.MODEL.CLIP.PROMPT_FREEZE = False
    # COOP
    cfg.MODEL.COOP = CN()
    cfg.MODEL.COOP.N_CTX = 16
    cfg.MODEL.COOP.CTX_INIT = "a sculpture of a {}."
    cfg.MODEL.COOP.CSC = False
    cfg.MODEL.COOP.CLASS_TOKEN_POSITION = "end"
    cfg.MODEL.COOP.FIX_TEXTPROMPT = False
    cfg.MODEL.COOP.N_PREFIX = 16
    cfg.MODEL.COOP.N_SUFFIX = 0
    cfg.MODEL.COOP.IMAGE_PROMPT_METHOD = "rgb"
    cfg.MODEL.COOP.MODEL_NAME = "ViT-B/16"
    cfg.MODEL.COOP.INPUT_MASK_FILL="mean"
    # V3
    cfg.MODEL.V3_TEST = CN()
    cfg.MODEL.V3_TEST.WITH_DENSE_CRF_POST_PROCESS = False
    cfg.MODEL.V3_TEST.DENSE_CRF_FEAT_STD = 0.1
    # WANDB
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "zero_shot_seg"
    cfg.WANDB.NAME = None


def add_our_config(cfg):
    cfg.ORACLE = False
    cfg.PSEUDO = False
    cfg.PSEUDO_WITH_PRIOR = True
    cfg.PSEUDO_REJECT_THRESHOLD = 0.0
    cfg.TEST.SLIDING_WINDOW = False
    cfg.TEST.SLIDING_TILE_SIZE = 224
    cfg.TEST.SLIDING_OVERLAP = 2 / 3.0
    cfg.PSEUDO_FLAG_NAME = "trainable_flag"
    cfg.SOLVER.TEST_IMS_PER_BATCH = 1
    cfg.DATASETS.SAMPLE_PER_CLASS = -1
    cfg.DATASETS.SAMPLE_SEED = 0
    # whether to use dense crf
    cfg.TEST.DENSE_CRF = False
    # embedding head
    cfg.MODEL.SEM_SEG_HEAD.EMBEDDING_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HIDDEN_DIM = 1024
    cfg.MODEL.SEM_SEG_HEAD.EMBED_LAYERS = 2
    # clip_adapter
    cfg.MODEL.CLIP_ADAPTER = CN()
    cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER = "imagenet"
    # for predefined
    cfg.MODEL.CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = ["a sculpture of a {}."]
    # for learnable prompt
    cfg.MODEL.CLIP_ADAPTER.PROMPT_DIM = 512
    cfg.MODEL.CLIP_ADAPTER.PROMPT_SHAPE = (16, 0)
    cfg.MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT = ""
    cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP_ADAPTER.MASK_FILL = "mean"
    cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO = 1.0
    cfg.MODEL.CLIP_ADAPTER.MASK_THR = 0.5
    cfg.MODEL.CLIP_ADAPTER.MASK_MATTING = False
    cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED = True
    cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE = True
    cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT = 0.8
    #
    cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER = False
    cfg.MODEL.CLIP_ADAPTER.USE_GT_CATEGORIES = False
    cfg.MODEL.CLIP_ADAPTER.COMMON_STRIDE = 32
    cfg.MODEL.CLIP_ADAPTER.USE_FPN = False
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER = CN()
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_LEARNER = "predefined"
    # for predefined
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = [
        "a sculpture of a {}."
    ]
    # for learnable prompt
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_DIM = 512
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_SHAPE = (16, 0)
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_CHECKPOINT = ""


def add_mask_former_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    add_mask_former_default_config(cfg)
    add_our_config(cfg)

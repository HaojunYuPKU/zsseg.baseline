export DETECTRON2_DATASETS="/mnt/haojun/code/zsseg.baseline/datasets/"
export CUDA_VISIBLE_DEVICES="0,1,2,3"


wandb off

python train_net.py --num-gpus 4 --eval-only \
--config-file configs/coco-stuff-164k-156/embedding_learn_text_only_fix_prompt_bg_zero_shot_mask_former_v2_mean_R101_coco_stuff_164k_bs32_60k.yaml \
OUTPUT_DIR output/debug \
MODEL.COOP.CTX_INIT none \
MODEL.MASK_FORMER.EMBED_WEIGHT 1.0 \
MODEL.SEM_SEG_HEAD.EMBED_HEAD.TEMPERATURE 0.02 \
MODEL.SEM_SEG_HEAD.EMBED_HEAD.TOTAL_SAMPLE_NUM 512 \
MODEL.WEIGHTS output/embedding_learn_text_only_fix_prompt_bg_zero_shot_mask_former_v2_mean_R101_coco_stuff_164k_bs32_60k_configs_coco-stuff2adde20k_pt_clip_text_prompt_learn_proposal_cls_coco_stuff_classification_v2_60000_0.02_fix_model_0049999_1.0_t0.02_bugfixed/model_final.pth

# MODEL.CLIP_ADAPTER.USE_GT_CATEGORIES True \
# ORACLE True \
# MODEL.CLIP_ADAPTER.USE_GT_CATEGORIES True \
# ORACLE True \
# MODEL.SEM_SEG_HEAD.IGNORE_VALUE 65535



#"coco_2017_test_stuff_sem_seg", "voc_sem_seg_test", "ade20k_sem_seg_val", "ade20k_panoptic_val", "ade20k_full_sem_seg_val"

# wandb online

# python tools/visualization.py

# 'OUTPUT_DIR', 'output/embedding_learn_text_only_fix_prompt_bg_zero_shot_mask_former_v2_mean_R101_coco_stuff_164k_bs32_60k_configs_coco-stuff2adde20k_pt_clip_text_prompt_learn_proposal_cls_coco_stuff_classification_v2_60000_0.02_fix_model_0049999_1.0_t0.02_bugfixed', 
# 'WANDB.NAME', 'embedding_learn_text_only_fix_prompt_bg_zero_shot_mask_former_v2_mean_R101_coco_stuff_164k_bs32_60k_configs_coco-stuff2adde20k_pt_clip_text_prompt_learn_proposal_cls_coco_stuff_classification_v2_60000_0.02_fix_model_0049999_1.0_t0.02_bugfixed', 
# 'MODEL.ZERO_SHOT_MASK_FORMER.CLIP_PRETRAINED', 'output/configs_coco-stuff2adde20k_pt_clip_text_prompt_learn_proposal_cls_coco_stuff_classification_v2_60000_0.02_fix/model_0049999.pth', 
# 'MODEL.COOP.CTX_INIT', 'none', 
# 'MODEL.MASK_FORMER.EMBED_WEIGHT', '1.0', 
# 'MODEL.SEM_SEG_HEAD.EMBED_HEAD.TEMPERATURE', '0.02', 
# 'MODEL.SEM_SEG_HEAD.EMBED_HEAD.TOTAL_SAMPLE_NUM', '512'], resume=True)

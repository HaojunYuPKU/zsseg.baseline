export DETECTRON2_DATASETS="/mnt/haojun/code/zsseg.baseline/datasets/"
export CUDA_VISIBLE_DEVICES="0,1,2,3"


wandb off

# python train_net.py --num-gpus 4 --eval-only \
# --config-file configs/coco-stuff-164k-156/embedding_learn_text_only_fix_prompt_bg_zero_shot_mask_former_v2_mean_R101_coco_stuff_164k_bs32_60k.yaml \
# OUTPUT_DIR output/debug \
# DATASETS.TEST "('ade20k_full_sem_seg_val',)" \
# MODEL.V3_TEST.WITH_GRAPH_CUT_POST_PROCESS False \
# MODEL.CLIP_ADAPTER.PROMPT_LEARNER learnable \
# MODEL.MASK_FORMER.EMBED_WEIGHT 1.0 \
# MODEL.SEM_SEG_HEAD.EMBED_HEAD.TEMPERATURE 0.02 \
# MODEL.SEM_SEG_HEAD.EMBED_HEAD.TOTAL_SAMPLE_NUM 512 \
# MODEL.WEIGHTS /mnt/haojun/code/zsseg.baseline/pretrained_models/zero_shot_mask_formerv3_new.pth \


python train_net.py --num-gpus 4 --eval-only \
--config-file configs/coco-stuff-164k-156/zero_shot_clip_only_perpixel_no_prompt.yaml \
OUTPUT_DIR output/debug \
DATASETS.TEST "('ade20k_full_sem_seg_val',)" \


# MODEL.CLIP_ADAPTER.USE_GT_CATEGORIES True \
# ORACLE True \
# MODEL.CLIP_ADAPTER.USE_GT_CATEGORIES True \
# ORACLE True \
# MODEL.SEM_SEG_HEAD.IGNORE_VALUE 65535

# pascal_context_val_sem_seg

#"coco_2017_test_stuff_sem_seg", "voc_sem_seg_test", "ade20k_sem_seg_val", "ade20k_panoptic_val", "ade20k_full_sem_seg_val"

# wandb online

# python tools/visualization.py

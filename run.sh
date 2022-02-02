export DETECTRON2_DATASETS="/mnt/haojun/code/zsseg.baseline/datasets/"
export CUDA_VISIBLE_DEVICES="0,1,2,3"


wandb off

python train_net.py --num-gpus 4 --resume \
--config-file configs/coco-stuff-164k-156/zero_shot_clip_fpn_perpixel_no_prompt.yaml \
OUTPUT_DIR output/clip_fpn_perpixel_coco \
SOLVER.IMS_PER_BATCH 16 \
SOLVER.BASE_LR 0.00005 \
# DATASETS.TEST "('ade20k_full_sem_seg_val',)" \
# MODEL.WEIGHTS output/clip_only_perpixel_coco/model_final.pth \
# MODEL.CLIP_ADAPTER.USE_GT_CATEGORIES True \
# ORACLE True \
# MODEL.SEM_SEG_HEAD.IGNORE_VALUE 65535



#"coco_2017_test_stuff_sem_seg", "voc_sem_seg_test", "ade20k_sem_seg_val", "ade20k_panoptic_val", "ade20k_full_sem_seg_val"

# wandb online

# python tools/visualization.py
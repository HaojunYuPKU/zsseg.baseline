export DETECTRON2_DATASETS="/mnt/haojun/code/zsseg.baseline/datasets/"


wandb off


CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --num-gpus 4 \
--config-file configs/coco-stuff-164k-156/zero_shot_clip_only_perpixel_R101c_single_prompt_bs4_60k.yaml \
ORACLE True \
MODEL.CLIP_ADAPTER.USE_GT_CATEGORIES True \
DATASETS.TRAIN "('coco_2017_train_stuff_sem_seg',)" \
DATASETS.TEST "('coco_2017_test_stuff_sem_seg',)" \
OUTPUT_DIR output/clip_only_perpixel_coco \
SOLVER.IMS_PER_BATCH 8


# "coco_2017_test_stuff_sem_seg", "voc_sem_seg_test", "ade20k_sem_seg_val", "ade20k_panoptic_val", "ade20k_full_sem_seg_val")

# wandb online

# python tools/visualization.py
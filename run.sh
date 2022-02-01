export DETECTRON2_DATASETS="/mnt/haojun/code/zsseg.baseline/datasets/"
export CUDA_VISIBLE_DEVICES="0,1,2,3"


wandb off



python train_net.py --num-gpus 4 --resume \
--config-file configs/coco-stuff-164k-156/zero_shot_clip_only_perpixel_bs8_60k.yaml \
OUTPUT_DIR output/clip_only_perpixel_coco \
SOLVER.IMS_PER_BATCH 8


# wandb online

# python tools/visualization.py
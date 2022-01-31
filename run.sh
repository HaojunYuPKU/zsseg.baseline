export DETECTRON2_DATASETS="/mnt/haojun/code/zsseg.baseline/datasets/"


# wandb off

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --num-gpus 1 --eval-only \
# --config-file configs/coco-stuff-164k-156/zero_shot_clip_only_perpixel_R101c_single_prompt_bs4_60k.yaml \
# DATASETS.TEST "('coco_2017_test_stuff_sem_seg',)" \
# OUTPUT_DIR output/clip_only_perpixel_coco 


wandb online

python tools/visualization.py
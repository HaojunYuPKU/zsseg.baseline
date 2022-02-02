#!/usr/bin/env bash
export $(grep -v '^#' .env | xargs)
set -x
wandb login --host=https://api.wandb.ai ${WANDB_TOKEN}
git clone https://${GIT_TOKEN}@github.com/HaojunYuPKU/zsseg.baseline code
cd code
pip install -r requirements.txt
mkdir -p /mnt/output/d2_result
ln -sT /mnt/output/d2_result output
cd third_party/CLIP
pip install -e .
cd ../../
#data
cd datasets
# ln -sT /itpeus4data/amldata/ade/ADEChallengeData2016 ADEChallengeData2016
ln -sT /itpeus4data/amldata/coco2017 coco
# ln -sT /itpeus4data/amldata/voc2012 VOC2012
cd ..

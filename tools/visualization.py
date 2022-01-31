"""
based on detectron2
"""
import os
import glob
import cv2
import functools
from tqdm import tqdm
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import numpy as np
from mask_former import *
from prettytable import PrettyTable
from pycocotools import mask as mask_utils
from PIL import Image

try:
    import wandb
except:
    wandb = None
import json


def mask2seg(
    mask_list, stuff_id_to_contiguous_id, ignore_label=255, include_label=None
):
    masks = [mask_utils.decode(m["segmentation"]) for m in mask_list]
    seg = np.full_like(masks[0], ignore_label)
    for i, mask in enumerate(mask_list):
        if (include_label is not None) and (i not in include_label):
            continue
        seg[masks[i] == 1] = stuff_id_to_contiguous_id[mask["category_id"]]
    return {"file_name": mask_list[0]["file_name"], "seg": seg}


def filter_label(seg,include_label,ignore_label=254,mask=None):
    if mask is not None:
        seg[mask]=ignore_label+1
    for i in np.unique(seg):
        if i not in include_label+[ignore_label]:
            seg[seg==i]=ignore_label
    return seg


def load_jsonfile(file_path, stuff_id_to_contiguous_id, include_label=None):
    with open(file_path) as f:
        pred_list = json.load(f)
    # {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
    # group pred
    print("Loading predictions....")
    preds = {}
    for pred in pred_list:
        if pred["file_name"] not in preds:
            preds[pred["file_name"]] = []
        preds[pred["file_name"]].append(pred)
    preds = [
        mask2seg(v, stuff_id_to_contiguous_id, include_label=include_label)
        for k, v in preds.items()
    ]
    return preds


def main(
    pred_jsonfile,
    wandb_title=None,
    gt_dir=None,
    img_dir=None,
    dataset_name="ade20k_sem_seg_val",
):

    metadata = MetadataCatalog.get(dataset_name)
    stuff_id_to_contiguous_id = metadata.stuff_dataset_id_to_contiguous_id
    class_labels = {i: name for i, name in enumerate(metadata.stuff_classes)}
    include_label = None
    class_labels[255]="ignore"
    if img_dir is None:
        img_dir = metadata.image_root
    if gt_dir is None:
        gt_dir = metadata.sem_seg_root
    if wandb_title is not None:
        wandb.init(project="open-voc-seg", entity="haojunyu", name=wandb_title)
    if "," in pred_jsonfile:
        pred_jsonfile = pred_jsonfile.split(",")
    else:
        pred_jsonfile = [pred_jsonfile]
    pred_jsonfile = [
        [p.split("=")[0], p.split("=")[1]] if "=" in p else ["pred", p]
        for p in pred_jsonfile
    ]
    preds = []
    table = PrettyTable(["File", "Size"])
    for f in pred_jsonfile:
        preds.append(
            [
                f[0],
                load_jsonfile(
                    f[1], stuff_id_to_contiguous_id, include_label=include_label
                ),
            ]
        )
        table.add_row([preds[-1][0], len(preds[-1][1])])
    gt_files = [
        os.path.join(gt_dir, os.path.basename(pred["file_name"]).replace("jpg", "png"))
        for pred in preds[0][1]
    ]
    img_files = [
        os.path.join(img_dir, os.path.basename(pred["file_name"]))
        for pred in preds[0][1]
    ]

    for i, (gfile, img_file) in tqdm(
        enumerate(zip(gt_files, img_files)), total=len(gt_files)
    ):

        gt = cv2.imread(gfile, cv2.IMREAD_GRAYSCALE)
        
        img = np.asarray(Image.open(img_file))

        masks = []
        for pred in preds:
            if wandb_title is not None:
                vis = Visualizer(img.copy(), MetadataCatalog.get(dataset_name))
                vis.draw_sem_seg(pred[1][i]["seg"])
                masks.append(wandb.Image(vis.get_output().get_image(), caption=pred[0]))
            # masks.append(
            #     {"pred": {"mask_data": pred[1][i]["seg"], "class_labels": class_labels}}
            # )
        if wandb_title is not None:
            vis = Visualizer(img.copy(), MetadataCatalog.get(dataset_name))
            vis.draw_sem_seg(gt)
            masks.append(wandb.Image(vis.get_output().get_image(), caption="gt"))
            wandb.log({"vis": masks})
        # masks.append({"gt": {"mask_data": gt, "class_labels": class_labels}})
        # for item in masks:
        #     try:
        #         mask_data = item["pred"]["mask_data"]
        #         print("pred", np.unique(mask_data.tolist()))
        #     except:
        #         mask_data = item["gt"]["mask_data"]
        #         print("gt", np.unique(mask_data.tolist()))
        # if wandb_title is not None:
        #     masks = [
        #         wandb.Image(img, masks=m, caption=c)
        #         for m, c in zip(
        #             masks, [pred_jsonfile[i] for i in range(len(pred_jsonfile))] + ["gt"]
        #         )
        #     ]
        #     wandb.log({"vis": masks})


if __name__ == "__main__":
    pred_jsonfile = "output/clip_only_perpixel_coco/inference/sem_seg_predictions.json"
    main(
        pred_jsonfile=pred_jsonfile,
        dataset_name="coco_2017_test_stuff_sem_seg",
        wandb_title="COCO visualization", #"PASCAL VOC visualization",
    )

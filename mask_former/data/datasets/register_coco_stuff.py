# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

from .utils import load_binary_mask


COCO_CATEGORIES = [
    {'id': 1, 'name': 'person', 'color': [220, 20, 60], 'isthing': 1},
    {'id': 2, 'name': 'bicycle', 'color': [119, 11, 32], 'isthing': 1},
    {'id': 3, 'name': 'car', 'color': [0, 0, 142], 'isthing': 1},
    {'id': 4, 'name': 'motorcycle', 'color': [0, 0, 230], 'isthing': 1},
    {'id': 5, 'name': 'airplane', 'color': [106, 0, 228], 'isthing': 1},
    {'id': 6, 'name': 'bus', 'color': [0, 60, 100], 'isthing': 1},
    {'id': 7, 'name': 'train', 'color': [0, 80, 100], 'isthing': 1},
    {'id': 8, 'name': 'truck', 'color': [0, 0, 70], 'isthing': 1},
    {'id': 9, 'name': 'boat', 'color': [0, 0, 192], 'isthing': 1},
    {'id': 10, 'name': 'traffic light', 'color': [250, 170, 30], 'isthing': 1},
    {'id': 11, 'name': 'fire hydrant', 'color': [100, 170, 30], 'isthing': 1},
    {'id': 12, 'name': 'stop sign', 'color': [220, 220, 0], 'isthing': 1},
    {'id': 13, 'name': 'parking meter', 'color': [175, 116, 175], 'isthing': 1},
    {'id': 14, 'name': 'bench', 'color': [250, 0, 30], 'isthing': 1},
    {'id': 15, 'name': 'bird', 'color': [165, 42, 42], 'isthing': 1},
    {'id': 16, 'name': 'cat', 'color': [255, 77, 255], 'isthing': 1},
    {'id': 17, 'name': 'dog', 'color': [0, 226, 252], 'isthing': 1},
    {'id': 18, 'name': 'horse', 'color': [182, 182, 255], 'isthing': 1},
    {'id': 19, 'name': 'sheep', 'color': [0, 82, 0], 'isthing': 1},
    {'id': 20, 'name': 'cow', 'color': [120, 166, 157], 'isthing': 1},
    {'id': 21, 'name': 'elephant', 'color': [110, 76, 0], 'isthing': 1},
    {'id': 22, 'name': 'bear', 'color': [174, 57, 255], 'isthing': 1},
    {'id': 23, 'name': 'zebra', 'color': [199, 100, 0], 'isthing': 1},
    {'id': 24, 'name': 'giraffe', 'color': [72, 0, 118], 'isthing': 1},
    {'id': 25, 'name': 'backpack', 'color': [255, 179, 240], 'isthing': 1},
    {'id': 26, 'name': 'umbrella', 'color': [0, 125, 92], 'isthing': 1},
    {'id': 27, 'name': 'handbag', 'color': [209, 0, 151], 'isthing': 1},
    {'id': 28, 'name': 'tie', 'color': [188, 208, 182], 'isthing': 1},
    {'id': 29, 'name': 'suitcase', 'color': [0, 220, 176], 'isthing': 1},
    {'id': 30, 'name': 'frisbee', 'color': [255, 99, 164], 'isthing': 1},
    {'id': 31, 'name': 'skis', 'color': [92, 0, 73], 'isthing': 1},
    {'id': 32, 'name': 'snowboard', 'color': [133, 129, 255], 'isthing': 1},
    {'id': 33, 'name': 'sports ball', 'color': [78, 180, 255], 'isthing': 1},
    {'id': 34, 'name': 'kite', 'color': [0, 228, 0], 'isthing': 1},
    {'id': 35, 'name': 'baseball bat', 'color': [174, 255, 243], 'isthing': 1},
    {'id': 36, 'name': 'baseball glove', 'color': [45, 89, 255], 'isthing': 1},
    {'id': 37, 'name': 'skateboard', 'color': [134, 134, 103], 'isthing': 1},
    {'id': 38, 'name': 'surfboard', 'color': [145, 148, 174], 'isthing': 1},
    {'id': 39, 'name': 'tennis racket', 'color': [255, 208, 186], 'isthing': 1},
    {'id': 40, 'name': 'bottle', 'color': [197, 226, 255], 'isthing': 1},
    {'id': 41, 'name': 'wine glass', 'color': [171, 134, 1], 'isthing': 1},
    {'id': 42, 'name': 'cup', 'color': [109, 63, 54], 'isthing': 1},
    {'id': 43, 'name': 'fork', 'color': [207, 138, 255], 'isthing': 1},
    {'id': 44, 'name': 'knife', 'color': [151, 0, 95], 'isthing': 1},
    {'id': 45, 'name': 'spoon', 'color': [9, 80, 61], 'isthing': 1},
    {'id': 46, 'name': 'bowl', 'color': [84, 105, 51], 'isthing': 1},
    {'id': 47, 'name': 'banana', 'color': [74, 65, 105], 'isthing': 1},
    {'id': 48, 'name': 'apple', 'color': [166, 196, 102], 'isthing': 1},
    {'id': 49, 'name': 'sandwich', 'color': [208, 195, 210], 'isthing': 1},
    {'id': 50, 'name': 'orange', 'color': [255, 109, 65], 'isthing': 1},
    {'id': 51, 'name': 'broccoli', 'color': [0, 143, 149], 'isthing': 1},
    {'id': 52, 'name': 'carrot', 'color': [179, 0, 194], 'isthing': 1},
    {'id': 53, 'name': 'hot dog', 'color': [209, 99, 106], 'isthing': 1},
    {'id': 54, 'name': 'pizza', 'color': [5, 121, 0], 'isthing': 1},
    {'id': 55, 'name': 'donut', 'color': [227, 255, 205], 'isthing': 1},
    {'id': 56, 'name': 'cake', 'color': [147, 186, 208], 'isthing': 1},
    {'id': 57, 'name': 'chair', 'color': [153, 69, 1], 'isthing': 1},
    {'id': 58, 'name': 'couch', 'color': [3, 95, 161], 'isthing': 1},
    {'id': 59, 'name': 'potted plant', 'color': [163, 255, 0], 'isthing': 1},
    {'id': 60, 'name': 'bed', 'color': [119, 0, 170], 'isthing': 1},
    {'id': 61, 'name': 'dining table', 'color': [0, 182, 199], 'isthing': 1},
    {'id': 62, 'name': 'toilet', 'color': [0, 165, 120], 'isthing': 1},
    {'id': 63, 'name': 'tv', 'color': [183, 130, 88], 'isthing': 1},
    {'id': 64, 'name': 'laptop', 'color': [95, 32, 0], 'isthing': 1},
    {'id': 65, 'name': 'mouse', 'color': [130, 114, 135], 'isthing': 1},
    {'id': 66, 'name': 'remote', 'color': [110, 129, 133], 'isthing': 1},
    {'id': 67, 'name': 'keyboard', 'color': [166, 74, 118], 'isthing': 1},
    {'id': 68, 'name': 'cell phone', 'color': [219, 142, 185], 'isthing': 1},
    {'id': 69, 'name': 'microwave', 'color': [79, 210, 114], 'isthing': 1},
    {'id': 70, 'name': 'oven', 'color': [178, 90, 62], 'isthing': 1},
    {'id': 71, 'name': 'toaster', 'color': [65, 70, 15], 'isthing': 1},
    {'id': 72, 'name': 'sink', 'color': [127, 167, 115], 'isthing': 1},
    {'id': 73, 'name': 'refrigerator', 'color': [59, 105, 106], 'isthing': 1},
    {'id': 74, 'name': 'book', 'color': [142, 108, 45], 'isthing': 1},
    {'id': 75, 'name': 'clock', 'color': [196, 172, 0], 'isthing': 1},
    {'id': 76, 'name': 'vase', 'color': [95, 54, 80], 'isthing': 1},
    {'id': 77, 'name': 'scissors', 'color': [128, 76, 255], 'isthing': 1},
    {'id': 78, 'name': 'teddy bear', 'color': [201, 57, 1], 'isthing': 1},
    {'id': 79, 'name': 'hair drier', 'color': [246, 0, 122], 'isthing': 1},
    {'id': 80, 'name': 'toothbrush', 'color': [191, 162, 208], 'isthing': 1},
    {'id': 81, 'name': 'banner', 'supercategory': 'textile'},
    {'id': 82, 'name': 'blanket', 'supercategory': 'textile'},
    {'id': 83, 'name': 'bridge', 'supercategory': 'building'},
    {'id': 84, 'name': 'cardboard', 'supercategory': 'raw-material'},
    {'id': 85, 'name': 'counter', 'supercategory': 'furniture-stuff'},
    {'id': 86, 'name': 'curtain', 'supercategory': 'textile'},
    {'id': 87, 'name': 'door-stuff', 'supercategory': 'furniture-stuff'},
    {'id': 88, 'name': 'floor-wood', 'supercategory': 'floor'},
    {'id': 89, 'name': 'flower', 'supercategory': 'plant'},
    {'id': 90, 'name': 'fruit', 'supercategory': 'food-stuff'},
    {'id': 91, 'name': 'gravel', 'supercategory': 'ground'},
    {'id': 92, 'name': 'house', 'supercategory': 'building'},
    {'id': 93, 'name': 'light', 'supercategory': 'furniture-stuff'},
    {'id': 94, 'name': 'mirror-stuff', 'supercategory': 'furniture-stuff'},
    {'id': 95, 'name': 'net', 'supercategory': 'structural'},
    {'id': 96, 'name': 'pillow', 'supercategory': 'textile'},
    {'id': 97, 'name': 'platform', 'supercategory': 'ground'},
    {'id': 98, 'name': 'playingfield', 'supercategory': 'ground'},
    {'id': 99, 'name': 'railroad', 'supercategory': 'ground'},
    {'id': 100, 'name': 'river', 'supercategory': 'water'},
    {'id': 101, 'name': 'road', 'supercategory': 'ground'},
    {'id': 102, 'name': 'roof', 'supercategory': 'building'},
    {'id': 103, 'name': 'sand', 'supercategory': 'ground'},
    {'id': 104, 'name': 'sea', 'supercategory': 'water'},
    {'id': 105, 'name': 'shelf', 'supercategory': 'furniture-stuff'},
    {'id': 106, 'name': 'snow', 'supercategory': 'ground'},
    {'id': 107, 'name': 'stairs', 'supercategory': 'furniture-stuff'},
    {'id': 108, 'name': 'tent', 'supercategory': 'building'},
    {'id': 109, 'name': 'towel', 'supercategory': 'textile'},
    {'id': 110, 'name': 'wall-brick', 'supercategory': 'wall'},
    {'id': 111, 'name': 'wall-stone', 'supercategory': 'wall'},
    {'id': 112, 'name': 'wall-tile', 'supercategory': 'wall'},
    {'id': 113, 'name': 'wall-wood', 'supercategory': 'wall'},
    {'id': 114, 'name': 'water-other', 'supercategory': 'water'},
    {'id': 115, 'name': 'window-blind', 'supercategory': 'window'},
    {'id': 116, 'name': 'window-other', 'supercategory': 'window'},
    {'id': 117, 'name': 'tree', 'supercategory': 'plant'},
    {'id': 118, 'name': 'fence', 'supercategory': 'structural'},
    {'id': 119, 'name': 'ceiling', 'supercategory': 'ceiling'},
    {'id': 120, 'name': 'sky-other', 'supercategory': 'sky'},
    {'id': 121, 'name': 'cabinet', 'supercategory': 'furniture-stuff'},
    {'id': 122, 'name': 'table', 'supercategory': 'furniture-stuff'},
    {'id': 123, 'name': 'floor-other', 'supercategory': 'floor'},
    {'id': 124, 'name': 'pavement', 'supercategory': 'floor'},
    {'id': 125, 'name': 'mountain', 'supercategory': 'solid'},
    {'id': 126, 'name': 'grass', 'supercategory': 'plant'},
    {'id': 127, 'name': 'dirt', 'supercategory': 'ground'},
    {'id': 128, 'name': 'paper', 'supercategory': 'textile'},
    {'id': 129, 'name': 'food-other', 'supercategory': 'food-stuff'},
    {'id': 130, 'name': 'building-other', 'supercategory': 'building'},
    {'id': 131, 'name': 'rock', 'supercategory': 'solid'},
    {'id': 132, 'name': 'wall-other', 'supercategory': 'wall'},
    {'id': 133, 'name': 'rug', 'supercategory': 'textile'}
]

COCO_BASE_CATEGORIES = [
    c
    for i, c in enumerate(COCO_CATEGORIES)
    if c["id"] - 1
    not in [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
]
COCO_NOVEL_CATEGORIES = [
    c
    for i, c in enumerate(COCO_CATEGORIES)
    if c["id"] - 1
    in [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
]


def _get_coco_stuff_meta(cat_list):
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in cat_list]

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in cat_list]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_coco_stuff_10k(root):
    root = os.path.join(root, "coco", "coco_stuff_10k")
    meta = _get_coco_stuff_meta(COCO_CATEGORIES)
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images_detectron2/train", "annotations_detectron2/train"),
        ("test", "images_detectron2/test", "annotations_detectron2/test"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"coco_2017_{name}_stuff_10k_sem_seg"
        DatasetCatalog.register(
            name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


def register_all_coco_stuff_164k(root):
    root = os.path.join(root, "coco")
    meta = _get_coco_stuff_meta(COCO_CATEGORIES)
    base_meta = _get_coco_stuff_meta(COCO_BASE_CATEGORIES)
    novel_meta = _get_coco_stuff_meta(COCO_NOVEL_CATEGORIES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "train2017", "stuffthingmaps_detectron2/train2017"),
        ("test", "val2017", "stuffthingmaps_detectron2/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"coco_2017_{name}_stuff_sem_seg"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
                "novel_thing": [
                    meta["stuff_classes"].index(n)
                    for i, n in enumerate(novel_meta["stuff_classes"])
                    if COCO_NOVEL_CATEGORIES[i].get("isthing", 0) == 1
                ],
                "novel_stuff": [
                    meta["stuff_classes"].index(n)
                    for i, n in enumerate(novel_meta["stuff_classes"])
                    if COCO_NOVEL_CATEGORIES[i].get("isthing", 0) == 0
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
            **meta,
        )
        # classification
        DatasetCatalog.register(
            all_name + "_classification",
            lambda x=image_dir, y=gt_dir: load_binary_mask(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name + "_classification").set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="classification",
            ignore_label=255,
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
            **meta,
        )

        # zero shot
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_base")
        base_name = f"coco_2017_{name}_stuff_base_sem_seg"

        DatasetCatalog.register(
            base_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(base_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **base_meta,
        )
        # classification
        DatasetCatalog.register(
            base_name + "_classification",
            lambda x=image_dir, y=gt_dir: load_binary_mask(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(base_name + "_classification").set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="classification",
            ignore_label=255,
            **base_meta,
        )
        # zero shot
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_novel")
        novel_name = f"coco_2017_{name}_stuff_novel_sem_seg"
        DatasetCatalog.register(
            novel_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(novel_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **novel_meta,
        )


def register_all_coco_stuff_164k_pseudo(root, pseudo_sem_dir):
    root = os.path.join(root, "coco")
    meta = _get_coco_stuff_meta(COCO_CATEGORIES)
    base_meta = _get_coco_stuff_meta(COCO_BASE_CATEGORIES)
    novel_meta = _get_coco_stuff_meta(COCO_NOVEL_CATEGORIES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "train2017", "stuffthingmaps_detectron2/train2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)

        all_name = f"coco_2017_{name}_stuff_sem_seg_pseudo"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=pseudo_sem_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=pseudo_sem_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
                "novel": [
                    meta["stuff_classes"].index(n) for n in novel_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
            **meta,
        )

# export DETECTRON2_DATASETS="/mnt/haojun/code/zsseg.baseline"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_stuff_10k(_root)
register_all_coco_stuff_164k(_root)

# _pseudo_dir = os.getenv("DETECTRON2_SEM_PSEUDO", "output/inference")
# register_all_coco_stuff_164k_pseudo(_root, _pseudo_dir)

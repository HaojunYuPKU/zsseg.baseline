import argparse
import os
import os.path as osp
import shutil
from functools import partial
from glob import glob

import mmcv
import numpy as np
from PIL import Image

"""
1-91% thing categories (80 total) 
92-182% original stuff categories (36 total) 
183-200% merged stuff categories (17 total)

The following new stuff categories were created by merging old stuff categories:
tree-merged: branch, tree, bush, leaves
fence-merged: cage, fence, railing
ceiling-merged: ceiling-tile, ceiling-other
sky-other-merged: clouds, sky-other, fog
cabinet-merged: cupboard, cabinet
table-merged: desk-stuff, table
floor-other-merged: floor-marble, floor-other, floor-tile
pavement-merged: floor-stone, pavement
mountain-merged: hill, mountain
grass-merged: moss, grass, straw
dirt-merged: mud, dirt
paper-merged: napkin, paper
food-other-merged: salad, vegetable, food-other
building-other-merged: skyscraper, building-other
rock-merged: stone, rock
wall-other-merged: wall-other, wall-concrete, wall-panel
rug-merged: mat, rug, carpet

The following stuff categories were removed (their pixels are set to void):
furniture-other, metal, plastic, solid-other, structural-other, waterdrops, textile-other, cloth, clothes, plant-other, wood, ground-other
"""

COCO_LEN = 123287

full_clsID_to_trID = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    26: 24,
    27: 25,
    30: 26,
    31: 27,
    32: 28,
    33: 29,
    34: 30,
    35: 31,
    36: 32,
    37: 33,
    38: 34,
    39: 35,
    40: 36,
    41: 37,
    42: 38,
    43: 39,
    45: 40,
    46: 41,
    47: 42,
    48: 43,
    49: 44,
    50: 45,
    51: 46,
    52: 47,
    53: 48,
    54: 49,
    55: 50,
    56: 51,
    57: 52,
    58: 53,
    59: 54,
    60: 55,
    61: 56,
    62: 57,
    63: 58,
    64: 59,
    66: 60,
    69: 61,
    71: 62,
    72: 63,
    73: 64,
    74: 65,
    75: 66,
    76: 67,
    77: 68,
    78: 69,
    79: 70,
    80: 71,
    81: 72,
    83: 73,
    84: 74,
    85: 75,
    86: 76,
    87: 77,
    88: 78,
    89: 79,
    91: 80,
    92: 81,
    93: 116,
    94: 82,
    95: 129,
    96: 116,
    97: 120,
    98: 117,
    99: 83,
    100: 132,
    101: 118,
    102: 118,
    103: 255,
    104: 255,
    105: 119,
    106: 84,
    107: 120,
    108: 85,
    109: 121,
    110: 126,
    111: 86,
    112: 117,
    113: 122,
    114: 122,
    115: 123,
    116: 122,
    117: 87,
    118: 88,
    119: 119,
    120: 128,
    121: 89,
    122: 255,
    123: 125,
    124: 90,
    125: 255,
    126: 124,
    127: 91,
    128: 116,
    129: 92,
    130: 132,
    131: 255,
    132: 93,
    133: 125,
    134: 124,
    135: 126,
    136: 127,
    137: 94,
    138: 127,
    139: 123,
    140: 95,
    141: 255,
    142: 255,
    143: 96,
    144: 97,
    145: 117,
    146: 98,
    147: 99,
    148: 100,
    149: 130,
    150: 101,
    151: 132,
    152: 128,
    153: 102,
    154: 103,
    155: 104,
    156: 119,
    157: 129,
    158: 105,
    159: 255,
    160: 106,
    161: 130,
    162: 125,
    163: 255,
    164: 121,
    165: 107,
    166: 255,
    167: 108,
    168: 116,
    169: 128,
    170: 109,
    171: 131,
    172: 131,
    173: 131,
    174: 110,
    175: 111,
    176: 112,
    177: 113,
    178: 255,
    179: 114,
    180: 115,
    181: 255,
    255: 255
}


def convert_to_trainID(
    maskpath, out_mask_dir, is_train, clsID_to_trID=full_clsID_to_trID, suffix=""
):
    mask = np.array(Image.open(maskpath))
    mask_copy = np.ones_like(mask, dtype=np.uint8) * 255
    for clsID, trID in clsID_to_trID.items():
        mask_copy[mask == clsID] = trID
    seg_filename = (
        osp.join(out_mask_dir, "train2017" + suffix, osp.basename(maskpath))
        if is_train
        else osp.join(out_mask_dir, "val2017" + suffix, osp.basename(maskpath))
    )
    if len(np.unique(mask_copy)) == 1 and np.unique(mask_copy)[0] == 255:
        return
    Image.fromarray(mask_copy).save(seg_filename, "PNG")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO Stuff 164k annotations to mmsegmentation format"
    )  # noqa
    parser.add_argument("coco_path", help="coco stuff path")
    parser.add_argument("-o", "--out_dir", help="output path")
    parser.add_argument("--nproc", default=16, type=int, help="number of process")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_path = args.coco_path
    nproc = args.nproc
    print(full_clsID_to_trID)
    out_dir = args.out_dir or coco_path
    out_mask_dir = osp.join(out_dir, "stuffthingmaps_detectron2")
    for dir_name in [
        "train2017",
        "val2017",
        "train2017_base",
        "train2017_novel",
        "val2017_base",
        "val2017_novel",
    ]:
        os.makedirs(osp.join(out_mask_dir, dir_name), exist_ok=True)
    train_list = glob(osp.join(coco_path, "stuffthingmaps", "train2017", "*.png"))
    test_list = glob(osp.join(coco_path, "stuffthingmaps", "val2017", "*.png"))
    assert (
        len(train_list) + len(test_list)
    ) == COCO_LEN, "Wrong length of list {} & {}".format(
        len(train_list), len(test_list)
    )

    if args.nproc > 1:
        mmcv.track_parallel_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=True),
            train_list,
            nproc=nproc,
        )
        mmcv.track_parallel_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=False),
            test_list,
            nproc=nproc,
        )
    else:
        mmcv.track_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=True),
            train_list,
        )
        mmcv.track_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=False),
            test_list,
        )
    print("Done!")


if __name__ == "__main__":
    main()

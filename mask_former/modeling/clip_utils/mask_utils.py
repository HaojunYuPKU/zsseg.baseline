import numpy as np
from scipy import ndimage
import cv2
import math
import torch

def expand_box(x1, y1, x2, y2, expand_ratio=1.0,max_h=None,max_w=None):
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    w = w * expand_ratio
    h = h * expand_ratio
    box = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    if max_h is not None:
        box[1]=max(0,box[1])
        box[3]=min(max_h-1,box[3])
    if max_w is not None:
        box[0]=max(0,box[0])
        box[2]=min(max_w-1,box[2])
    return [int(b) for b in box]

def mask2box(mask: torch.Tensor):
    # use naive way
    row = torch.nonzero(mask.sum(dim=0))[0]
    if len(row) == 0:
        return None
    x1 = row.min()
    x2 = row.max()
    col = np.nonzero(mask.sum(dim=1))[0]
    y1 = col.min()
    y2 = col.max()
    return x1, y1, x2 + 1, y2 + 1

def crop_with_mask(image:torch.Tensor,mask:torch.Tensor,fill=(0,0,0),expand_ratio=1.0,thr=None):
    if thr:
        mask = (mask>thr).float()
    #get box
    l,t,r,b =mask2box(mask)
    l,t,r,b = expand_box(l,t,r,b,expand_ratio)
    _,h,w=image.shape
    l = max(l, 0)
    t = max(t, 0)
    r = min(r, w)
    b = min(b, h)
    new_image = torch.cat([image.new_full((1,b-t,r-l),fill_value=val) for val in fill])
    return image[:,t:b,l:r]*mask[None,t:b,l:r]+(1-mask[None,t:b,l:r])*new_image
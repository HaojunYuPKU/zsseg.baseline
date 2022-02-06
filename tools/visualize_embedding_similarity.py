import os

import numpy as np
import torch
import wandb
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm
import cv2

# MaskFormer
from mask_former import *
from PIL import Image
from torch.nn import functional as F
from torchvision.utils import make_grid


def main(
    embedding_dir,
    score_dir=None,
    image_dir=None,
    gt_dir=None,
    novel_class_only=False,
    dataset_name="coco_2017_test_stuff_sem_seg",
):

    # load metadata
    metadata = MetadataCatalog.get(dataset_name)
    if novel_class_only:
        label2dis = metadata.evaluation_set["novel"]
    else:
        label2dis = np.arange(len(metadata.stuff_classes))

    # load embedding
    embedding_files = os.listdir(embedding_dir)
    if image_dir is None:
        image_dir = metadata.image_root
    if gt_dir is None:
        gt_dir = metadata.sem_seg_root
    wandb.init()
    # visualize
    for em_file in tqdm(embedding_files):

        image_path = os.path.join(
            image_dir, os.path.basename(em_file).replace(".pth", ".jpg")
        )
        gt_path = os.path.join(
            gt_dir, os.path.basename(em_file).replace(".pth", ".png")
        )
        image = np.asarray(Image.open(image_path))
        h, w, _ = image.shape
        gt = np.asarray(Image.open(gt_path))
        embedding = torch.load(os.path.join(embedding_dir, em_file)).float().cuda()
        print(
            "get image {} gt {} embedding {}".format(
                image.shape, gt.shape, embedding.shape
            )
        )
        embedding = embedding.reshape(embedding.shape[0], -1).permute(1, 0)

        vis = Visualizer(image.copy(), metadata=metadata)
        unique_label = np.unique(gt)
        if label2dis is not None:
            ignored = [u for u in unique_label if u not in label2dis]
        for l in ignored:
            gt[gt == l] = 255
        vis.draw_sem_seg(gt)
        vis_ims = [vis.get_output().get_image()]
        for label in unique_label:
            if (label2dis is not None) and (label not in label2dis):
                continue
            elif label == 255:
                continue
            mask = gt == label
            inds = np.nonzero(mask.reshape((-1,)))[0]
            inds = np.random.choice(inds, size=2)
            selected_embedding = embedding[torch.from_numpy(inds).cuda()]
            sim = F.cosine_similarity(
                selected_embedding[:, None, :], embedding[None, ...], dim=-1
            ).reshape(-1, h, w)
            coords = np.stack([inds % w, inds // w], axis=1)  # k,2
            for one_sim, coord in zip(sim, coords):
                one_sim = one_sim.cpu().numpy()
                heatmap = (one_sim - np.min(one_sim)) / np.max(one_sim)
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.cvtColor(
                    cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
                )
                vis = Visualizer(image.copy(), metadata=metadata)
                # draw heatmap
                vis.output.ax.imshow(heatmap, alpha=0.9)
                vis.draw_circle([coord[0], coord[1]], radius=5, color="r")
                vis_ims.append(vis.get_output().get_image())
        if len(vis_ims) == 1:
            continue
        imgs = [
            wandb.Image(vis_im, caption=em_file if i == 0 else f"sim_{i}")
            for i, vis_im in enumerate(vis_ims)
        ]

        if score_dir is not None:
            score = (
                torch.load(os.path.join(score_dir, em_file))
                .cpu()
                .numpy()
                .astype(np.float)
            )  # c,h,w
            
            if label2dis is not None:
                score = score[np.asarray(label2dis).astype(np.int32)]
                print(score.shape)
            pred = np.asarray(label2dis).astype(np.int32)[np.argmax(score,axis=0).reshape((-1,))].reshape((h,w))
            pred[gt==255] = 255
            vis = Visualizer(image.copy(), metadata=metadata)
            vis.draw_sem_seg(pred)
            imgs.append(wandb.Image(vis.get_output().get_image(), caption="novel_only_pred"))
            
            
            score = (score - np.max(score,axis=0,keepdims=True)) 
            score = score.transpose((1,2,0))
            score = (score-score[gt!=255].min())/ (score[gt!=255].max() - score[gt!=255].min())
            score[gt==255] =0
            score = score.transpose((2,0,1))

            for i in range(len(score)):
                vis = Visualizer(image.copy(), metadata=metadata)
                vis.output.ax.imshow(score[i], alpha=0.9)
                imgs.append(
                    wandb.Image(
                        vis.get_output().get_image(),
                        caption=metadata.stuff_classes[label2dis[i]],
                    )
                )
        wandb.log({dataset_name: imgs})


if __name__ == "__main__":
    import fire

    fire.Fire(main)

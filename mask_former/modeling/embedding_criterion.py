import torch
import numpy as np
from torch.nn import Module
from torch.nn import functional as F


class EmbeddingCriterion(Module):
    def __init__(
        self,
        total_sample_num=512,
        sample_method="uniform",
        temperature=0.05,
        weight=1.0,
        per_image=False,
    ):
        super().__init__()
        self.total_sample_num = total_sample_num
        self.sample_method = sample_method
        self.temperature = temperature
        self.weight = weight
        self.per_image = per_image

    def forward(self, input, target):
        if self.per_image:
            loss = [
                self.loss(
                    embed[None, ...],
                    seg_map[None, ...],
                    ignore_label=target["ignore_label"],
                )
                for embed, seg_map in zip(input["pix_embedding"], target["sem_seg"])
            ]
            loss = sum(loss) / len(loss)
        else:
            loss = self.loss(
                input["pix_embedding"],
                target["sem_seg"],
                ignore_label=target["ignore_label"],
            )
        return {"loss_embed": self.weight * loss}

    def loss(self, pix_embedding, sem_seg_target, ignore_label=255):
        """Sample points from per-pixel embedding and compute the similarity loss
        Args:
            pix_embedding (torch.Tensor): [B,C,H,W]
            sem_seg_target (torch.Tensor): [B,H,W]
            ignore_label (torch.Tensor): unlabeled area or ignored area in the image
        """
        b, _, h, w = pix_embedding.shape
        pix_embedding = pix_embedding.permute(0, 2, 3, 1).reshape(
            b * h * w, -1
        )  # [B,H,W,C]
        sem_seg_target = sem_seg_target.reshape(b * h * w)
        unique_label = torch.unique(sem_seg_target)
        pos_bucket = [
            torch.nonzero(sem_seg_target == l)[:, 0]
            for l in unique_label
            if l != ignore_label
        ]
        if len(pos_bucket) == 0:
            return pix_embedding[sem_seg_target != ignore_label].sum()

        pos_inds = self._sample(pos_bucket)
        sample_cls = torch.cat(
            [torch.Tensor([i for _ in range(len(p))]) for i, p in enumerate(pos_inds)],
            dim=0,
        ).to(pix_embedding.device)

        sample_embedding = torch.cat([pix_embedding[i] for i in pos_inds], dim=0)

        # compute loss
        return self.loss_similarity(sample_embedding, sample_cls)

    def _sample(self, buckets):
        """Sample points from each buckets
        Args:
            num_per_buckets (list): number of points in each class
        """
        num_per_buckets = [len(p) for p in buckets]
        if self.sample_method == "uniform":
            sample_per_bucket = [
                self.total_sample_num // len(buckets)
                for _ in range(len(num_per_buckets))
            ]
        elif self.sample_method == "sqrt":
            sqrt_size = [n ** 0.5 for n in num_per_buckets]
            total_sqrt_size = sum(sqrt_size)
            sample_per_bucket = [
                int(self.total_sample_num * n / total_sqrt_size) for n in sqrt_size
            ]
        else:
            raise NotImplementedError()
        if len(sample_per_bucket) > 1:
            sample_per_bucket[-1] = self.total_sample_num - sum(sample_per_bucket[:-1])
        else:
            sample_per_bucket[0] = self.total_sample_num
        samples = [
            p[
                torch.from_numpy(
                    np.random.choice(len(p), sample_per_bucket[i], replace=True)
                ).to(p.device)
            ]
            for i, p in enumerate(buckets)
        ]
        return samples

    def loss_similarity(self, embedding, label):
        """Compute the similarity loss
        Args:
            embedding (torch.Tensor): [B,C]
            label (torch.Tensor): [B]
        """
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        cos_sim = embedding @ embedding.T  # [B,B]
        exp_sim = torch.exp(cos_sim / self.temperature)
        pos_mask = (label[:, None] == label[None, :]).type(exp_sim.dtype)  # [B,B]
        neg_mask = 1 - pos_mask
        # remove self-to-self sim
        pos_mask[
            torch.arange(len(pos_mask)).to(pos_mask.device),
            torch.arange(len(pos_mask)).to(pos_mask.device),
        ] = 0

        neg_exp_sim_sum = (exp_sim * neg_mask).sum(dim=-1, keepdim=True)
        prob = exp_sim / (exp_sim + neg_exp_sim_sum).clamp(min=1e-8)
        # select positive pair
        pos_prob = prob[pos_mask == 1]
        loss = -torch.log(pos_prob + 1e-8).mean()
        return loss

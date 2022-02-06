import torch
from torch.nn import functional as F
import numpy as np
import time

try:
    from pygco import cut_from_graph
except ImportError:
    raise FileNotFoundError(
        "Missing Grah-Cut (GCO) library,"
        " please install it from https://github.com/Borda/pyGCO."
    )


def unary_from_logits(logits, temperature=1, eps=1e-8):
    # convert logits to probabilities
    probs = F.softmax(logits / temperature, dim=0)
    unary_cost = -torch.log(probs.clamp(min=eps))
    if isinstance(unary_cost, torch.Tensor):
        unary_cost = unary_cost.detach().cpu().numpy()
    return unary_cost


def construct_neightbor_edges(point_x, point_y, x_shift, y_shift, shape):
    """
    point_x: [N_Points,1]
    point_y: [N_Points,1]
    x_shift: [N_Neighbors,]
    y_shift: [N_Neighbors,]
    """
    h, w = shape
    x_shift = x_shift.reshape((1, -1))
    y_shift = y_shift.reshape((1, -1))
    point_x = point_x.reshape((-1, 1))
    point_y = point_y.reshape((-1, 1))
    end_x = point_x + x_shift  # N_Points,N_Neighbors
    end_y = point_y + y_shift  # N_Points,N_Neighbors
    start_x = np.repeat(point_x, x_shift.shape[-1], axis=1)  # N_Points,N_Neighbors
    start_y = np.repeat(point_y, y_shift.shape[-1], axis=1)  # N_Points,N_Neighbors
    start = start_y * w + start_x  # N_Points,N_Neighbors
    end = end_y * w + end_x  # N_Points,N_Neighbors
    start = start.reshape((-1, 1))
    end = end.reshape((-1, 1))
    edges = np.concatenate((start, end), axis=1)  # N_Points*N_Neighbors,2
    return edges


def graph_cut_post_process(
    logits,
    pix_embedding,
    logits_temperature=0.02,
    pix_temperature=0.02,
    edge_type="four_connect",
    label_compatibility=None,
    label_transition_cost=1,
    n_labels=None,
    n_iter=5,
    eps=1e-8,
):
    """
    logits: [N_Cls,H,W], logits for each pixel
    pix_embedding: [H,W,C], embedding for each pixel
    label_compatibility: [N_Cls,N_Cls], similarity between each label
    n_labels: int
    """
    _, h, w = logits.shape
    if n_labels is None:
        n_labels = logits.shape[0]
    # defina the unary energy of each pixel
    unary_cost = (
        unary_from_logits(logits, logits_temperature, eps).reshape((n_labels, -1)).T
    )  # [H*W,N_Cls]
    # define label compatibility cost
    if label_compatibility is None:
        pairwise_cost = (np.ones(n_labels) - np.eye(n_labels)) * label_transition_cost
    else:
        if isinstance(label_compatibility, torch.Tensor):
            label_compatibility = label_compatibility.detach().cpu().numpy()
        pairwise_cost = (
            1 / (label_compatibility + eps) - np.eye(n_labels)
        ) * label_transition_cost
    # define the graph with edges
    edges = None  # [N_Edges,2]
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    if edge_type == "four_connect":
        x_shift = np.array([[1, 0]]).reshape((1, -1))
        y_shift = np.array([[0, 1]]).reshape((1, -1))
        edges = construct_neightbor_edges(x, y, x_shift, y_shift, (h, w))
    elif edge_type == "eight_connect":
        x_shift = np.array([[-1, 1, -1, 0, 1]])
        y_shift = np.array([[0, 0, 1, 1, 1]])
        edges = construct_neightbor_edges(x, y, x_shift, y_shift, (h, w))
    else:
        # do not implement fully connected graph
        raise NotImplementedError("")

    pix_embedding = pix_embedding.reshape(-1, pix_embedding.shape[-1])
    num_nodes = pix_embedding.shape[0]
    valid_start = (edges[:, 0] >= 0) & (edges[:, 0] < num_nodes)
    valid_end = (edges[:, 1] >= 0) & (edges[:, 1] < num_nodes)
    valid_mask = valid_start & valid_end
    edges = edges[valid_mask]

    cos_sim = F.cosine_similarity(
        pix_embedding[torch.from_numpy(edges[:, 0]).to(pix_embedding.device)],
        pix_embedding[torch.from_numpy(edges[:, 1]).to(pix_embedding.device)],
        dim=-1,
    )
    # cos_sim = F.softmax(cos_sim / pix_temperature, dim=0)
    # edge_weight = -torch.log(cos_sim.clamp(min=eps))
    edge_weight = (cos_sim + 1) * unary_cost.std() #* 10
    # edge_weight = torch.exp(cos_sim / pix_temperature).detach().cpu().numpy()
    
    print(unary_cost.mean(), unary_cost.std())
    print(edge_weight.mean(), edge_weight.std())

    edges = np.concatenate([edges, edge_weight.reshape(-1, 1)], axis=1)

    graph_labels = cut_from_graph(
        np.ascontiguousarray(edges).astype(np.int32),
        np.ascontiguousarray(unary_cost).astype(np.int32),
        np.ascontiguousarray(pairwise_cost).astype(np.int32),
        algorithm="expansion",
        n_iter=n_iter,
    )
    graph_labels = graph_labels.reshape((h, w))
    return graph_labels


if __name__ == "__main__":
    embedding = torch.load("/mnt/haojun/code/zsseg.baseline/output/tmp_embedding_res/000000000139.pth")
    embedding = embedding.float().permute(1,2,0)
    embedding = embedding.norm(dim=-1, keepdim=True)
    r = torch.load("/mnt/haojun/code/zsseg.baseline/output/tmp_score_res/000000000139.pth")
    r = r.float()
    start = time.time()
    r = graph_cut_post_process(
        logits=r, 
        pix_embedding=embedding,
        logits_temperature=0.1,
        pix_temperature=0.01,
    )
    end = time.time()
    print(end-start)
    torch.save(torch.from_numpy(r), "/mnt/haojun/code/zsseg.baseline/output/tmp_img_res/000000000139.pth")
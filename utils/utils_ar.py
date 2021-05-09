
import torch
import numpy as np


def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def quantize_cluster(delta_x, delta_y, delta_w, delta_h, centroid_x, centroid_y, centroid_w, centroid_h, num_cluster):

    B, T = delta_x.shape[0], delta_x.shape[1]
    # compute distances to the cluster centroids
    batch_centroid_x = centroid_x.unsqueeze(0).repeat(B, T, 1)
    batch_gt_x = delta_x.repeat(1, 1, num_cluster)
    dist_gt_x = torch.abs(batch_centroid_x.float() - batch_gt_x.float())
    min_val_x, min_idx_x = torch.min(dist_gt_x, dim=-1)

    batch_centroid_y = centroid_y.unsqueeze(0).repeat(B, T, 1)
    batch_gt_y = delta_y.repeat(1, 1, num_cluster)
    dist_gt_y = torch.abs(batch_centroid_y.float() - batch_gt_y.float())
    min_val_y, min_idx_y = torch.min(dist_gt_y, dim=-1)

    batch_centroid_w = centroid_w.unsqueeze(0).repeat(B, T, 1)
    batch_gt_w = delta_w.repeat(1, 1, num_cluster)
    dist_gt_w = torch.abs(batch_centroid_w.float() - batch_gt_w.float())
    min_val_w, min_idx_w = torch.min(dist_gt_w, dim=-1)

    batch_centroid_h = centroid_h.unsqueeze(0).repeat(B, T, 1)
    batch_gt_h = delta_h.repeat(1, 1, num_cluster)
    dist_gt_h = torch.abs(batch_centroid_h.float() - batch_gt_h.float())
    min_val_h, min_idx_h = torch.min(dist_gt_h, dim=-1)

    return min_idx_x, min_idx_y, min_idx_w, min_idx_h


def quantize_bin(delta, bin):

    B, T = delta.shape[0], delta.shape[1]
    # compute distances to the cluster centroids
    batch_centroid = bin.unsqueeze(0).repeat(B, T, 1)
    batch_gt = delta.repeat(1, 1, bin.shape[1])
    dist_gt = torch.abs(batch_centroid.float() - batch_gt.float())
    min_val, min_idx = torch.min(dist_gt, dim=-1)

    return min_idx


def sample_cluster(dist, centroid):
    indices = (torch.topk(dist, k=1)[1]).squeeze(1)
    return centroid[0, list(indices.squeeze())].unsqueeze(1)


def infer_log_likelihood(dist_x, dist_y, dist_w, dist_h, x, y, w, h, centroid_x, centroid_y, centroid_w, centroid_h, num_clusters):
    probs = []
    idx_x, idx_y, idx_w, idx_h = quantize_cluster(x[:, -1:, :], y[:, -1:, :], w[:, -1:, :], h[:, -1:, :], centroid_x, centroid_y, centroid_w, centroid_h, num_clusters)
    prob_t = [dist_x[:, -1:, idx_x.long().item()].item(), dist_y[:, -1:, idx_y.long().item()].item(),
              dist_w[:, -1:, idx_w.long().item()].item(), dist_h[:, -1:, idx_h.long().item()].item()]
    probs.append(torch.sum(torch.log(torch.tensor(prob_t))))
    return probs

"""Official Evaluation Metric Implementation for Biohub - Cell Tracking During Development.

Metric Formulation:
    Score = adjusted_edge_jaccard + 0.1 * division_jaccard
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)  # (Z, Y, X)


def match_nodes_bipartite(
    pred_nodes: dict[int, tuple[int, int, int, int]], 
    gt_nodes: dict[int, tuple[int, int, int, int]], 
    max_dist: float = 7.0,
    voxel_scale: tuple[float, float, float] = VOXEL_SCALE_UM
) -> tuple[dict[int, int], dict[int, int]]:
    """Performs Hungarian matching of predicted nodes to ground truth per timepoint.

    Args:
        pred_nodes: dict mapping node_id -> (t, z, y, x)
        gt_nodes: dict mapping node_id -> (t, z, y, x)
        max_dist: distance threshold in physical micrometers (7.0 um)
        voxel_scale: (z_scale, y_scale, x_scale)

    Returns:
        (pred_to_gt, gt_to_pred) mapping dicts
    """
    pred_by_t: dict[int, list[int]] = {}
    for pid, (t, *_r) in pred_nodes.items():
        pred_by_t.setdefault(int(t), []).append(pid)
    gt_by_t: dict[int, list[int]] = {}
    for gid, (t, *_r) in gt_nodes.items():
        gt_by_t.setdefault(int(t), []).append(gid)

    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    scale_arr = np.array(voxel_scale, dtype=float)

    for t, p_ids in pred_by_t.items():
        g_ids = gt_by_t.get(t, [])
        if not g_ids:
            continue
        p_pos = np.array([pred_nodes[p][1:] for p in p_ids], dtype=float) * scale_arr
        g_pos = np.array([gt_nodes[g][1:] for g in g_ids], dtype=float) * scale_arr
        diff = p_pos[:, None, :] - g_pos[None, :, :]
        cost = np.sqrt((diff ** 2).sum(axis=-1))
        BIG = 1e6
        cost_gated = np.where(cost <= max_dist, cost, BIG)
        row_ind, col_ind = linear_sum_assignment(cost_gated)
        for r, c in zip(row_ind, col_ind):
            if cost_gated[r, c] >= BIG:
                continue
            pred_to_gt[p_ids[r]] = g_ids[c]
            gt_to_pred[g_ids[c]] = p_ids[r]

    return pred_to_gt, gt_to_pred


def compute_edge_confusion(
    pred_edges: list[tuple[int, int]], 
    gt_edges: list[tuple[int, int]], 
    pred_to_gt: dict[int, int], 
    gt_to_pred: dict[int, int]
) -> tuple[int, int, int]:
    """Computes TP, FP, FN for edges accounting for sparse labeling."""
    gt_edge_set = set(gt_edges)
    gt_outgoing: dict[int, set[int]] = {}
    gt_incoming_source: dict[int, int] = {}
    for s, t in gt_edge_set:
        gt_outgoing.setdefault(s, set()).add(t)
        gt_incoming_source[t] = s

    tp = 0
    fp = 0
    matched_gt_edges = set()
    for s, t in pred_edges:
        ms = pred_to_gt.get(s)
        mt = pred_to_gt.get(t)
        is_tp = ms is not None and mt is not None and mt in gt_outgoing.get(ms, ())
        if is_tp:
            tp += 1
            matched_gt_edges.add((ms, mt))
            continue
        is_fp = (mt is not None and mt in gt_incoming_source) or (
            ms is not None and bool(gt_outgoing.get(ms))
        )
        if is_fp:
            fp += 1
    fn = len(gt_edge_set - matched_gt_edges)
    return tp, fp, fn


def edge_jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom else 0.0


def adjusted_jaccard(jaccard: float, t_pred: int, t_true: int, a: float = 0.1) -> float:
    """Adjusts edge Jaccard by penalizing excess predicted nodes."""
    if not t_true or t_true <= 0:
        return jaccard
    return max(0.0, jaccard * (1.0 - a * max(0, t_pred - t_true) / t_true))


def compute_tracking_score(
    adjusted_edge_jaccard: float, 
    division_jaccard: float
) -> float:
    """Computes final competition score: adjusted_edge_jaccard + 0.1 * division_jaccard."""
    return adjusted_edge_jaccard + 0.1 * division_jaccard

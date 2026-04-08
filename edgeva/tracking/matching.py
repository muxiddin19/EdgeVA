"""
Association utilities for multi-object tracking.

Implements IoU, cosine, and fused cost matrices together with the
linear assignment (Hungarian algorithm) used by ByteTrack [1],
OC-SORT [2], and the LITE feature-reuse tracker [3].

References
----------
[1] Zhang et al., "ByteTrack", ECCV 2022.
[2] Cao et al., "Observation-Centric SORT", CVPR 2023.
[3] Alikhanov et al., "LITE", arXiv 2024.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    raise ImportError("scipy is required: pip install scipy")


# ---------------------------------------------------------------------------
# IoU utilities
# ---------------------------------------------------------------------------

def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes.

    Parameters
    ----------
    boxes_a : ndarray, shape (M, 4)  — [x1, y1, x2, y2]
    boxes_b : ndarray, shape (N, 4)  — [x1, y1, x2, y2]

    Returns
    -------
    iou : ndarray, shape (M, N)
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(
        inter_y2 - inter_y1, 0
    )
    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] → [cx,cy,w,h]."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx,cy,w,h] → [x1,y1,x2,y2]."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_distance(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine distance (1 − similarity).

    Parameters
    ----------
    features_a : ndarray, shape (M, D)
    features_b : ndarray, shape (N, D)

    Returns
    -------
    distances : ndarray, shape (M, N),  range [0, 2]
    """
    norm_a = features_a / (np.linalg.norm(features_a, axis=1, keepdims=True) + 1e-8)
    norm_b = features_b / (np.linalg.norm(features_b, axis=1, keepdims=True) + 1e-8)
    return 1.0 - norm_a @ norm_b.T


# ---------------------------------------------------------------------------
# Linear assignment
# ---------------------------------------------------------------------------

def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """
    Solve linear assignment and split into matches / unmatched rows / cols.

    Parameters
    ----------
    cost_matrix : ndarray, shape (M, N)
    thresh      : float — costs above this threshold are rejected

    Returns
    -------
    matches          : ndarray, shape (K, 2)  — [row_idx, col_idx]
    unmatched_rows   : ndarray, shape (R,)
    unmatched_cols   : ndarray, shape (C,)
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    valid = cost_matrix[row_ind, col_ind] <= thresh
    matches = np.stack([row_ind[valid], col_ind[valid]], axis=1)

    matched_rows = set(row_ind[valid])
    matched_cols = set(col_ind[valid])
    unmatched_rows = np.array(
        [i for i in range(cost_matrix.shape[0]) if i not in matched_rows], dtype=int
    )
    unmatched_cols = np.array(
        [j for j in range(cost_matrix.shape[1]) if j not in matched_cols], dtype=int
    )
    return matches, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# Fused cost
# ---------------------------------------------------------------------------

def fused_iou_cosine_cost(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    track_features: np.ndarray | None,
    det_features: np.ndarray | None,
    lambda_iou: float = 0.5,
) -> np.ndarray:
    """
    Fused cost = lambda_iou * (1 − IoU) + (1 − lambda_iou) * cosine_dist.

    Falls back to pure IoU cost when features are not provided.

    Parameters
    ----------
    track_boxes    : ndarray, shape (M, 4) — [x1,y1,x2,y2]
    det_boxes      : ndarray, shape (N, 4) — [x1,y1,x2,y2]
    track_features : ndarray, shape (M, D) or None
    det_features   : ndarray, shape (N, D) or None
    lambda_iou     : weight of IoU term (default 0.5)

    Returns
    -------
    cost : ndarray, shape (M, N)
    """
    iou_cost = 1.0 - box_iou_batch(track_boxes, det_boxes)

    if track_features is None or det_features is None:
        return iou_cost

    cos_cost = cosine_distance(track_features, det_features)
    return lambda_iou * iou_cost + (1.0 - lambda_iou) * cos_cost

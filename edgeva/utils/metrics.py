"""
MOT evaluation metrics: HOTA, MOTA, IDF1, and helpers.

Implements the HOTA metric [1] as the primary evaluation criterion,
plus MOTA and IDF1 for backward compatibility with benchmark leaderboards.

References
----------
[1] Luiten et al., "HOTA: A Higher Order Metric for Evaluating
    Multi-Object Tracking", IJCV 2021.
[2] Milan et al., "MOT16: A Benchmark for Multi-Object Tracking",
    arXiv 2016.
[3] Ristani et al., "Performance Measures and a Data Set for
    Multi-Target, Multi-Camera Tracking", ECCVW 2016.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TrackletFrame:
    """Single-frame detection or ground-truth annotation."""
    frame_id: int
    track_id: int
    bbox: np.ndarray        # [x1, y1, x2, y2]
    confidence: float = 1.0


@dataclass
class MOTMetrics:
    """Container for evaluated MOT metrics."""
    hota: float = 0.0
    deta: float = 0.0      # Detection accuracy component of HOTA
    assa: float = 0.0      # Association accuracy component of HOTA
    mota: float = 0.0
    motp: float = 0.0
    idf1: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    num_fp: int = 0
    num_fn: int = 0
    num_sw: int = 0        # identity switches
    num_gt: int = 0
    num_pred: int = 0

    def __str__(self) -> str:
        return (
            f"HOTA={self.hota:.1f}%  DetA={self.deta:.1f}%  "
            f"AssA={self.assa:.1f}%  MOTA={self.mota:.1f}%  "
            f"IDF1={self.idf1:.1f}%  FP={self.num_fp}  "
            f"FN={self.num_fn}  SW={self.num_sw}"
        )


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Scalar IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


# ---------------------------------------------------------------------------
# MOTA / IDF1 evaluator
# ---------------------------------------------------------------------------

class MOTEvaluator:
    """
    Frame-by-frame accumulator for MOTA and IDF1.

    Usage
    -----
    >>> ev = MOTEvaluator()
    >>> for frame_id, gt_list, pred_list in data:
    ...     ev.update(frame_id, gt_list, pred_list)
    >>> metrics = ev.compute()
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self._gt_count = 0
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._id_switches = 0
        self._motp_sum = 0.0
        # For IDF1: track identity correspondence
        self._gt_to_pred: Dict[int, int] = {}    # last matched pred id per gt id
        self._pred_tp: Dict[int, int] = {}        # TP count per pred id
        self._gt_tp: Dict[int, int] = {}          # TP count per gt id
        self._pred_count: Dict[int, int] = {}
        self._gt_count_per_id: Dict[int, int] = {}

    def update(
        self,
        frame_id: int,
        gt_tracks: List[TrackletFrame],
        pred_tracks: List[TrackletFrame],
    ):
        """Accumulate one frame of ground-truth vs. predictions."""
        from scipy.optimize import linear_sum_assignment

        n_gt = len(gt_tracks)
        n_pred = len(pred_tracks)
        self._gt_count += n_gt

        for gt in gt_tracks:
            self._gt_count_per_id[gt.track_id] = (
                self._gt_count_per_id.get(gt.track_id, 0) + 1
            )
        for pr in pred_tracks:
            self._pred_count[pr.track_id] = (
                self._pred_count.get(pr.track_id, 0) + 1
            )

        if n_gt == 0 and n_pred == 0:
            return
        if n_gt == 0:
            self._fp += n_pred
            return
        if n_pred == 0:
            self._fn += n_gt
            return

        # Build IoU cost matrix
        cost = np.zeros((n_gt, n_pred), dtype=np.float64)
        for i, gt in enumerate(gt_tracks):
            for j, pr in enumerate(pred_tracks):
                iou = _iou(gt.bbox, pr.bbox)
                cost[i, j] = 1.0 - iou  # minimise

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_gt = set()
        matched_pred = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= 1.0 - self.iou_threshold:
                gt_id = gt_tracks[r].track_id
                pred_id = pred_tracks[c].track_id
                self._tp += 1
                self._motp_sum += 1.0 - cost[r, c]
                matched_gt.add(r)
                matched_pred.add(c)
                # Identity switch detection
                if gt_id in self._gt_to_pred and self._gt_to_pred[gt_id] != pred_id:
                    self._id_switches += 1
                self._gt_to_pred[gt_id] = pred_id
                # IDF1 bookkeeping
                self._pred_tp[pred_id] = self._pred_tp.get(pred_id, 0) + 1
                self._gt_tp[gt_id] = self._gt_tp.get(gt_id, 0) + 1

        self._fn += n_gt - len(matched_gt)
        self._fp += n_pred - len(matched_pred)

    def compute(self) -> MOTMetrics:
        """Return accumulated MOT metrics."""
        tp = self._tp
        fp = self._fp
        fn = self._fn
        gt_total = self._gt_count

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        mota = 1.0 - (fp + fn + self._id_switches) / max(gt_total, 1)
        motp = self._motp_sum / max(tp, 1)

        # IDF1
        idtp = sum(self._pred_tp.values())
        idfp = sum(
            self._pred_count.get(pid, 0) - self._pred_tp.get(pid, 0)
            for pid in self._pred_count
        )
        idfn = sum(
            self._gt_count_per_id.get(gid, 0) - self._gt_tp.get(gid, 0)
            for gid in self._gt_count_per_id
        )
        idf1_r = idtp / max(2 * idtp + idfp + idfn, 1)

        return MOTMetrics(
            mota=mota * 100,
            motp=motp * 100,
            idf1=idf1_r * 100,
            recall=recall * 100,
            precision=precision * 100,
            num_fp=fp,
            num_fn=fn,
            num_sw=self._id_switches,
            num_gt=gt_total,
            num_pred=tp + fp,
        )


# ---------------------------------------------------------------------------
# HOTA evaluator (simplified single-threshold)
# ---------------------------------------------------------------------------

class HOTAEvaluator:
    """
    HOTA metric evaluator at a single IoU threshold alpha.

    Full HOTA averages over alpha in {0.05, 0.10, ..., 0.95}.
    This class computes the per-alpha score; wrap in HOTARange for full HOTA.

    Reference: Luiten et al., IJCV 2021.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._det_tp = 0
        self._det_fp = 0
        self._det_fn = 0
        # For association: track all TP (gt_id, pred_id) pairs
        self._ass_tp: Dict[Tuple[int, int], int] = {}
        self._ass_fp: Dict[int, int] = {}   # pred_id → unmatched count
        self._ass_fn: Dict[int, int] = {}   # gt_id   → unmatched count

    def update(
        self,
        gt_tracks: List[TrackletFrame],
        pred_tracks: List[TrackletFrame],
    ):
        from scipy.optimize import linear_sum_assignment

        n_gt = len(gt_tracks)
        n_pred = len(pred_tracks)

        if n_gt == 0 and n_pred == 0:
            return
        if n_gt == 0:
            self._det_fp += n_pred
            for pr in pred_tracks:
                self._ass_fp[pr.track_id] = self._ass_fp.get(pr.track_id, 0) + 1
            return
        if n_pred == 0:
            self._det_fn += n_gt
            for gt in gt_tracks:
                self._ass_fn[gt.track_id] = self._ass_fn.get(gt.track_id, 0) + 1
            return

        cost = np.zeros((n_gt, n_pred))
        for i, gt in enumerate(gt_tracks):
            for j, pr in enumerate(pred_tracks):
                cost[i, j] = 1.0 - _iou(gt.bbox, pr.bbox)

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_gt, matched_pred = set(), set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= 1.0 - self.alpha:
                gt_id = gt_tracks[r].track_id
                pred_id = pred_tracks[c].track_id
                self._det_tp += 1
                self._ass_tp[(gt_id, pred_id)] = (
                    self._ass_tp.get((gt_id, pred_id), 0) + 1
                )
                matched_gt.add(r)
                matched_pred.add(c)

        for i, gt in enumerate(gt_tracks):
            if i not in matched_gt:
                self._det_fn += 1
                self._ass_fn[gt.track_id] = self._ass_fn.get(gt.track_id, 0) + 1

        for j, pr in enumerate(pred_tracks):
            if j not in matched_pred:
                self._det_fp += 1
                self._ass_fp[pr.track_id] = self._ass_fp.get(pr.track_id, 0) + 1

    def compute(self) -> Tuple[float, float, float]:
        """
        Returns
        -------
        hota_alpha : HOTA score at this alpha
        deta       : Detection accuracy
        assa       : Association accuracy
        """
        det_tp = self._det_tp
        deta = det_tp / max(det_tp + self._det_fp + self._det_fn, 1)

        assa_sum = 0.0
        for (gt_id, pred_id), tp_count in self._ass_tp.items():
            fp_count = self._ass_fp.get(pred_id, 0)
            fn_count = self._ass_fn.get(gt_id, 0)
            assa_sum += tp_count / max(tp_count + fp_count + fn_count, 1)

        assa = assa_sum / max(det_tp, 1)
        hota = np.sqrt(deta * assa)
        return hota, deta, assa


class HOTARange:
    """Average HOTA over alpha ∈ {0.05, 0.10, …, 0.95}."""

    ALPHAS = np.arange(0.05, 1.00, 0.05)

    def __init__(self):
        self._evals = {a: HOTAEvaluator(alpha=a) for a in self.ALPHAS}

    def update(
        self,
        gt_tracks: List[TrackletFrame],
        pred_tracks: List[TrackletFrame],
    ):
        for ev in self._evals.values():
            ev.update(gt_tracks, pred_tracks)

    def compute(self) -> MOTMetrics:
        hotas, detas, assas = [], [], []
        for ev in self._evals.values():
            h, d, a = ev.compute()
            hotas.append(h)
            detas.append(d)
            assas.append(a)
        return MOTMetrics(
            hota=float(np.mean(hotas)) * 100,
            deta=float(np.mean(detas)) * 100,
            assa=float(np.mean(assas)) * 100,
        )

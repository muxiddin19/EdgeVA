"""
Feature-Reuse Tracker — LITE Paradigm Implementation.

The LITE paradigm [1] eliminates the dedicated ReID network by reusing
intermediate feature maps already computed by the detector backbone.
This yields a 2–4× end-to-end speedup with <1% HOTA degradation on
MOT17/MOT20 compared to appearance-based baselines (DeepSORT, BoT-SORT).

Pipeline
--------
1.  Detector (YOLOv8/v10/v12 or RT-DETR) produces detections AND
    exposes FPN feature maps {P3, P4, P5}.
2.  For each detection, RoI-pool the FPN feature at the chosen level
    to obtain a compact appearance descriptor (no separate network).
3.  Associate descriptors across frames using a fused
    (IoU + cosine) cost matrix solved by the Hungarian algorithm.
4.  Kalman filter predicts track positions between frames.

References
----------
[1] Alikhanov et al., "LITE: A Paradigm Shift in Multi-Object Tracking
    with Efficient ReID Feature Extraction", arXiv 2024.
[2] Alikhanov et al., "Practical Edge-Deployable Multi-Object Tracking
    via the LITE Paradigm", 2025.
[3] Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
    Every Detection Box", ECCV 2022.
[4] Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian
    Tracking", arXiv 2022.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from .kalman_filter import KalmanFilter
from .matching import (
    box_iou_batch,
    cxcywh_to_xyxy,
    xyxy_to_cxcywh,
    fused_iou_cosine_cost,
    linear_assignment,
)


# ---------------------------------------------------------------------------
# Track state machine
# ---------------------------------------------------------------------------

class TrackState(Enum):
    TENTATIVE = "tentative"   # waiting to be confirmed
    CONFIRMED = "confirmed"   # active track
    LOST      = "lost"        # temporarily not detected
    REMOVED   = "removed"     # permanently deleted


@dataclass
class STrack:
    """
    Single object track carrying Kalman state and appearance history.

    Attributes
    ----------
    track_id    : unique integer identity
    state       : TrackState
    hits        : total number of frames this track was matched
    age         : total number of frames since creation
    time_since_update : frames elapsed since last successful match
    mean        : Kalman state mean [cx,cy,w,h, vx,vy,vw,vh]
    covariance  : Kalman state covariance (8×8)
    features    : rolling appearance descriptor buffer
    """

    track_id: int
    bbox_xyxy: np.ndarray          # [x1,y1,x2,y2] — initialisation box
    score: float = 1.0
    feature: Optional[np.ndarray] = None

    # Kalman state (filled on first predict)
    mean: np.ndarray = field(default_factory=lambda: np.zeros(8))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(8))

    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 1
    time_since_update: int = 0

    # Appearance feature EMA buffer
    _feature_buffer: List[np.ndarray] = field(default_factory=list)
    _feature_ema: Optional[np.ndarray] = None
    _ema_alpha: float = 0.9        # EMA weight for smoothing

    def __post_init__(self):
        cxcywh = xyxy_to_cxcywh(self.bbox_xyxy[None])[0]
        kf = KalmanFilter()
        self.mean, self.covariance = kf.initiate(cxcywh)
        if self.feature is not None:
            self._update_feature(self.feature)

    # ------------------------------------------------------------------
    def predict(self, kf: KalmanFilter):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        kf: KalmanFilter,
        det_box: np.ndarray,
        det_score: float,
        det_feature: Optional[np.ndarray],
    ):
        cxcywh = xyxy_to_cxcywh(det_box[None])[0]
        self.mean, self.covariance = kf.update(self.mean, self.covariance, cxcywh)
        self.score = det_score
        self.hits += 1
        self.time_since_update = 0
        if det_feature is not None:
            self._update_feature(det_feature)

    def _update_feature(self, feat: np.ndarray):
        if self._feature_ema is None:
            self._feature_ema = feat.copy()
        else:
            self._feature_ema = (
                self._ema_alpha * self._feature_ema
                + (1.0 - self._ema_alpha) * feat
            )

    # ------------------------------------------------------------------
    @property
    def xyxy(self) -> np.ndarray:
        """Current bounding box [x1,y1,x2,y2] from Kalman state."""
        return cxcywh_to_xyxy(self.mean[:4][None])[0]

    @property
    def appearance(self) -> Optional[np.ndarray]:
        return self._feature_ema


# ---------------------------------------------------------------------------
# Feature extractor (RoI-pool from FPN maps)
# ---------------------------------------------------------------------------

def extract_roi_features(
    fpn_features: dict,
    boxes_xyxy: np.ndarray,
    level: str = "P4",
    output_size: int = 7,
) -> np.ndarray:
    """
    RoI-pool features from a single FPN level — the core of LITE [1].

    In production this is accelerated by torchvision.ops.roi_pool or
    the TensorRT plugin; this reference implementation uses NumPy for
    portability.

    Parameters
    ----------
    fpn_features : dict mapping level name to ndarray (H, W, C)
                   e.g. {"P3": ..., "P4": ..., "P5": ...}
    boxes_xyxy   : ndarray, shape (N, 4) — boxes in image coordinates
    level        : FPN level to pool from ("P3" | "P4" | "P5")
    output_size  : spatial output size after pooling (default 7)

    Returns
    -------
    features : ndarray, shape (N, C * output_size * output_size)
               Flattened, L2-normalised descriptors.

    Notes
    -----
    FPN level strides:  P3 → 8,  P4 → 16,  P5 → 32.
    Selecting P4 (stride 16) provides the best speed–accuracy trade-off
    according to ablation studies in [1].
    """
    stride_map = {"P3": 8, "P4": 16, "P5": 32}
    if level not in fpn_features:
        raise KeyError(f"FPN level '{level}' not in feature dict. "
                       f"Available: {list(fpn_features.keys())}")

    feat_map = fpn_features[level]          # (H_feat, W_feat, C)
    stride = stride_map[level]
    H_feat, W_feat, C = feat_map.shape
    N = len(boxes_xyxy)

    out = np.zeros((N, C, output_size, output_size), dtype=np.float32)

    for i, box in enumerate(boxes_xyxy):
        # Map box to feature-map coordinates
        x1_f = box[0] / stride
        y1_f = box[1] / stride
        x2_f = box[2] / stride
        y2_f = box[3] / stride

        x1_f = max(0.0, min(x1_f, W_feat - 1))
        y1_f = max(0.0, min(y1_f, H_feat - 1))
        x2_f = max(0.0, min(x2_f, W_feat - 1))
        y2_f = max(0.0, min(y2_f, H_feat - 1))

        # Crop and resize via max-pool bins
        roi = feat_map[int(y1_f):max(int(y2_f), int(y1_f) + 1),
                       int(x1_f):max(int(x2_f), int(x1_f) + 1), :]  # (rH, rW, C)
        if roi.size == 0:
            continue

        # Divide ROI into output_size × output_size bins, max-pool each
        rH, rW = roi.shape[:2]
        for ph in range(output_size):
            for pw in range(output_size):
                h0 = int(rH * ph / output_size)
                h1 = max(int(rH * (ph + 1) / output_size), h0 + 1)
                w0 = int(rW * pw / output_size)
                w1 = max(int(rW * (pw + 1) / output_size), w0 + 1)
                out[i, :, ph, pw] = roi[h0:h1, w0:w1, :].max(axis=(0, 1))

    # Flatten and L2-normalise
    out_flat = out.reshape(N, -1).astype(np.float32)
    norms = np.linalg.norm(out_flat, axis=1, keepdims=True)
    out_flat /= np.maximum(norms, 1e-8)
    return out_flat


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------

class FeatureReuseTracker:
    """
    LITE-paradigm multi-object tracker.

    Designed for real-time operation on edge hardware (Jetson Orin,
    Hailo-8, Google Coral) where a separate ReID forward pass is
    prohibitively expensive.

    Parameters
    ----------
    track_thresh      : detection score threshold for high-confidence dets
    match_thresh      : cost threshold for first-round association
    second_thresh     : cost threshold for second-round (low-conf) association
    max_time_lost     : frames to keep a lost track before removal
    min_hits          : frames before a tentative track is confirmed
    fpn_level         : FPN feature level for descriptor extraction
    lambda_iou        : weight of IoU term in fused cost (0 = pure appearance)
    use_byte          : if True, include low-confidence detections in 2nd round
                        (ByteTrack strategy [3])
    """

    _next_id = 1

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        second_thresh: float = 0.5,
        max_time_lost: int = 30,
        min_hits: int = 3,
        fpn_level: str = "P4",
        lambda_iou: float = 0.5,
        use_byte: bool = True,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.second_thresh = second_thresh
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits
        self.fpn_level = fpn_level
        self.lambda_iou = lambda_iou
        self.use_byte = use_byte

        self._kf = KalmanFilter()
        self._tracked: List[STrack] = []
        self._lost: List[STrack] = []
        self._frame_id = 0

    @classmethod
    def _next_track_id(cls) -> int:
        tid = cls._next_id
        cls._next_id += 1
        return tid

    # ------------------------------------------------------------------
    def reset(self):
        """Reset tracker state (e.g. between sequences)."""
        self._tracked.clear()
        self._lost.clear()
        self._frame_id = 0
        FeatureReuseTracker._next_id = 1

    # ------------------------------------------------------------------
    def update(
        self,
        detections: np.ndarray,
        scores: np.ndarray,
        fpn_features: Optional[dict] = None,
    ) -> List[STrack]:
        """
        Process one frame and return active confirmed tracks.

        Parameters
        ----------
        detections   : ndarray, shape (N, 4) — [x1,y1,x2,y2] in image coords
        scores       : ndarray, shape (N,)   — detection confidence scores
        fpn_features : dict {level: ndarray (H, W, C)} or None
                       When None, falls back to IoU-only association.

        Returns
        -------
        active_tracks : list of confirmed STrack objects (time_since_update == 0)
        """
        self._frame_id += 1

        # --- Split detections by confidence ---
        high_mask = scores >= self.track_thresh
        low_mask  = (~high_mask) & (scores >= 0.1)

        det_high  = detections[high_mask]
        score_high = scores[high_mask]
        det_low   = detections[low_mask]
        score_low  = scores[low_mask]

        # --- Extract features ---
        feat_high = None
        if fpn_features is not None and len(det_high) > 0:
            feat_high = extract_roi_features(fpn_features, det_high, self.fpn_level)

        # --- Predict all tracks ---
        all_tracks = self._tracked + self._lost
        for t in all_tracks:
            t.predict(self._kf)

        # --- Round 1: high-confidence dets vs. confirmed/lost tracks ---
        tracked_confirmed = [t for t in self._tracked
                             if t.state == TrackState.CONFIRMED]

        matches1, unmatched_tracks1, unmatched_dets1 = self._associate(
            tracked_confirmed, det_high, score_high, feat_high,
            thresh=self.match_thresh,
        )

        for t_idx, d_idx in matches1:
            feat = feat_high[d_idx] if feat_high is not None else None
            tracked_confirmed[t_idx].update(
                self._kf, det_high[d_idx], score_high[d_idx], feat
            )
            tracked_confirmed[t_idx].state = TrackState.CONFIRMED

        # --- Round 2 (ByteTrack): low-conf dets vs. unmatched confirmed ---
        remaining_confirmed = [tracked_confirmed[i] for i in unmatched_tracks1]
        if self.use_byte and len(det_low) > 0:
            matches2, unmatched_tracks2, _ = self._associate(
                remaining_confirmed, det_low, score_low, None,
                thresh=self.second_thresh,
            )
            for t_idx, d_idx in matches2:
                remaining_confirmed[t_idx].update(
                    self._kf, det_low[d_idx], score_low[d_idx], None
                )
                remaining_confirmed[t_idx].state = TrackState.CONFIRMED
            still_unmatched = [remaining_confirmed[i] for i in unmatched_tracks2]
        else:
            still_unmatched = remaining_confirmed

        # --- Round 3: unmatched high-conf dets vs. lost tracks ---
        unmatched_det_boxes = det_high[unmatched_dets1]
        unmatched_det_scores = score_high[unmatched_dets1]
        unmatched_det_feats = (feat_high[unmatched_dets1]
                               if feat_high is not None else None)

        matches3, _, unmatched_dets3 = self._associate(
            self._lost, unmatched_det_boxes, unmatched_det_scores,
            unmatched_det_feats, thresh=self.match_thresh,
        )
        recovered = []
        for t_idx, d_idx in matches3:
            feat = (unmatched_det_feats[d_idx]
                    if unmatched_det_feats is not None else None)
            self._lost[t_idx].update(
                self._kf,
                unmatched_det_boxes[d_idx],
                unmatched_det_scores[d_idx],
                feat,
            )
            self._lost[t_idx].state = TrackState.CONFIRMED
            recovered.append(self._lost[t_idx])

        # --- Mark unmatched confirmed as LOST ---
        for t in still_unmatched:
            t.state = TrackState.LOST

        # --- Create new tracks from unmatched high-conf detections ---
        new_tracks = []
        final_unmatched_boxes  = unmatched_det_boxes[unmatched_dets3]
        final_unmatched_scores = unmatched_det_scores[unmatched_dets3]
        final_unmatched_feats  = (unmatched_det_feats[unmatched_dets3]
                                  if unmatched_det_feats is not None else
                                  [None] * len(unmatched_dets3))

        for i in range(len(final_unmatched_boxes)):
            feat = (final_unmatched_feats[i]
                    if not isinstance(final_unmatched_feats, list) else None)
            new_tracks.append(
                STrack(
                    track_id=self._next_track_id(),
                    bbox_xyxy=final_unmatched_boxes[i],
                    score=float(final_unmatched_scores[i]),
                    feature=feat,
                )
            )

        # --- Remove stale lost tracks ---
        self._lost = [
            t for t in self._lost + [t for t in still_unmatched]
            if t not in recovered
            and t.time_since_update <= self.max_time_lost
        ]
        for t in self._lost:
            if t.time_since_update > self.max_time_lost:
                t.state = TrackState.REMOVED

        # --- Update tracked pool ---
        tentative_survived = [
            t for t in self._tracked
            if t.state == TrackState.TENTATIVE and t.time_since_update == 0
        ]
        self._tracked = (
            [t for t in tracked_confirmed if t.state == TrackState.CONFIRMED]
            + recovered
            + new_tracks
        )

        # Promote tentative → confirmed after min_hits
        for t in self._tracked:
            if t.state == TrackState.TENTATIVE and t.hits >= self.min_hits:
                t.state = TrackState.CONFIRMED

        return [t for t in self._tracked
                if t.state == TrackState.CONFIRMED and t.time_since_update == 0]

    # ------------------------------------------------------------------
    def _associate(
        self,
        tracks: List[STrack],
        det_boxes: np.ndarray,
        det_scores: np.ndarray,
        det_features: Optional[np.ndarray],
        thresh: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(tracks) == 0 or len(det_boxes) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(tracks), dtype=int),
                np.arange(len(det_boxes), dtype=int),
            )

        track_boxes = np.array([t.xyxy for t in tracks])
        track_feats = np.array([t.appearance for t in tracks
                                if t.appearance is not None])
        if len(track_feats) != len(tracks):
            track_feats = None

        cost = fused_iou_cosine_cost(
            track_boxes, det_boxes,
            track_feats, det_features,
            lambda_iou=self.lambda_iou,
        )
        return linear_assignment(cost, thresh)

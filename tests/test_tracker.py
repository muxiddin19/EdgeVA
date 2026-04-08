"""Unit tests for the feature-reuse tracker and Kalman filter."""

import numpy as np
import pytest

from edgeva.tracking import FeatureReuseTracker, KalmanFilter, STrack, TrackState
from edgeva.tracking.matching import box_iou_batch, cosine_distance, linear_assignment


# ---------------------------------------------------------------------------
# Kalman filter tests
# ---------------------------------------------------------------------------

class TestKalmanFilter:
    def setup_method(self):
        self.kf = KalmanFilter()

    def test_initiate_shape(self):
        box = np.array([100, 200, 50, 80], dtype=np.float32)  # cx,cy,w,h
        mean, cov = self.kf.initiate(box)
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)

    def test_predict_increases_uncertainty(self):
        box = np.array([100, 200, 50, 80], dtype=np.float32)
        mean, cov = self.kf.initiate(box)
        _, cov2 = self.kf.predict(mean, cov)
        # Predicted covariance should be >= initial (uncertainty grows)
        assert np.all(np.diag(cov2) >= np.diag(cov) * 0.99)

    def test_update_reduces_uncertainty(self):
        box = np.array([100, 200, 50, 80], dtype=np.float32)
        mean, cov = self.kf.initiate(box)
        mean_pred, cov_pred = self.kf.predict(mean, cov)
        _, cov_upd = self.kf.update(mean_pred, cov_pred, box)
        # Updated covariance should be <= predicted
        assert np.all(np.diag(cov_upd) <= np.diag(cov_pred) + 1e-6)

    def test_gating_distance_self(self):
        box = np.array([100, 200, 50, 80], dtype=np.float32)
        mean, cov = self.kf.initiate(box)
        dist = self.kf.gating_distance(mean, cov, box[None])
        assert dist.shape == (1,)
        assert dist[0] < 1.0   # self-distance should be near 0


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------

class TestMatching:
    def test_iou_identical_boxes(self):
        boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
        iou = box_iou_batch(boxes, boxes)
        assert abs(iou[0, 0] - 1.0) < 1e-5

    def test_iou_non_overlapping(self):
        a = np.array([[0, 0, 50, 50]], dtype=np.float32)
        b = np.array([[100, 100, 200, 200]], dtype=np.float32)
        iou = box_iou_batch(a, b)
        assert iou[0, 0] == pytest.approx(0.0)

    def test_cosine_identical(self):
        v = np.random.randn(1, 64).astype(np.float32)
        dist = cosine_distance(v, v)
        assert dist[0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_linear_assignment_basic(self):
        cost = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        matches, unm_r, unm_c = linear_assignment(cost, thresh=0.5)
        assert matches.shape[1] == 2
        assert len(unm_r) == 0
        assert len(unm_c) == 0

    def test_linear_assignment_threshold(self):
        cost = np.array([[0.9, 0.2], [0.3, 0.8]], dtype=np.float32)
        matches, unm_r, unm_c = linear_assignment(cost, thresh=0.5)
        # Only cost[0,1]=0.2 and cost[1,0]=0.3 should match
        assert len(matches) == 2

    def test_empty_cost_matrix(self):
        cost = np.empty((0, 3), dtype=np.float32)
        matches, unm_r, unm_c = linear_assignment(cost, thresh=0.5)
        assert len(matches) == 0
        assert len(unm_r) == 0
        assert len(unm_c) == 3


# ---------------------------------------------------------------------------
# Feature-reuse tracker integration tests
# ---------------------------------------------------------------------------

class TestFeatureReuseTracker:
    def _make_boxes(self, n: int = 5) -> tuple:
        """Generate n random boxes and scores."""
        boxes = np.random.uniform(0, 400, (n, 4)).astype(np.float32)
        # Ensure x2>x1, y2>y1
        boxes[:, 2] = boxes[:, 0] + np.random.uniform(20, 100, n)
        boxes[:, 3] = boxes[:, 1] + np.random.uniform(20, 100, n)
        scores = np.random.uniform(0.5, 1.0, n).astype(np.float32)
        return boxes, scores

    def test_first_frame_creates_tracks(self):
        tracker = FeatureReuseTracker(min_hits=1)
        boxes, scores = self._make_boxes(5)
        tracks = tracker.update(boxes, scores)
        # After 1 frame with min_hits=1, all should be confirmed
        assert len(tracks) == 5

    def test_track_ids_unique(self):
        tracker = FeatureReuseTracker(min_hits=1)
        boxes, scores = self._make_boxes(4)
        tracks = tracker.update(boxes, scores)
        ids = [t.track_id for t in tracks]
        assert len(ids) == len(set(ids))

    def test_empty_detections(self):
        tracker = FeatureReuseTracker(min_hits=1)
        tracks = tracker.update(np.empty((0, 4)), np.empty(0))
        assert len(tracks) == 0

    def test_track_persistence(self):
        """Tracks should persist across frames for static boxes."""
        tracker = FeatureReuseTracker(min_hits=1, max_time_lost=5)
        boxes = np.array([[100, 100, 200, 200],
                          [300, 300, 400, 400]], dtype=np.float32)
        scores = np.array([0.9, 0.85])

        ids_per_frame = []
        for _ in range(5):
            tracks = tracker.update(boxes.copy(), scores.copy())
            ids_per_frame.append(set(t.track_id for t in tracks))

        # IDs should remain consistent across frames
        assert ids_per_frame[0] == ids_per_frame[-1]

    def test_reset_clears_state(self):
        tracker = FeatureReuseTracker(min_hits=1)
        boxes, scores = self._make_boxes(3)
        tracker.update(boxes, scores)
        tracker.reset()
        assert len(tracker._tracked) == 0
        assert len(tracker._lost) == 0
        assert tracker._frame_id == 0

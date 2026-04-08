"""Unit tests for analytics modules."""

import time
import numpy as np
import pytest

from edgeva.analytics import (
    LineCrossingCounter,
    ZoneOccupancyMonitor,
    DwellTimeAnalyser,
    PPEComplianceMonitor,
    PPEClass,
    ZoneRule,
)


class TestLineCrossingCounter:
    def setup_method(self):
        # Vertical line at x=100
        self.counter = LineCrossingCounter(
            line_p1=(100, 0), line_p2=(100, 480), name="test"
        )

    def test_count_in(self):
        # Line p1=(100,0)→p2=(100,480): line_vec=(0,480)
        # cross = 0*y - 480*(x-100):  x<100 → positive side (+1), x>100 → negative side (-1)
        # Crossing from positive→negative means side changes to -1 → count_out increments.
        # "in" direction = right→left (negative→positive side change).
        self.counter.update(1, (150, 240))  # right of line: side = -1
        self.counter.update(1, (50, 240))   # left of line:  side = +1  → count_in
        assert self.counter.count_in == 1
        assert self.counter.count_out == 0

    def test_count_out(self):
        self.counter.update(2, (50, 240))   # left of line:  side = +1
        self.counter.update(2, (150, 240))  # right of line: side = -1  → count_out
        assert self.counter.count_out == 1
        assert self.counter.count_in == 0

    def test_net_count(self):
        self.counter.update(1, (50, 100))
        self.counter.update(1, (150, 100))
        self.counter.update(2, (150, 200))
        self.counter.update(2, (50, 200))
        assert self.counter.net_count() == 0   # 1 in, 1 out

    def test_reset(self):
        self.counter.update(1, (50, 100))
        self.counter.update(1, (150, 100))
        self.counter.reset()
        assert self.counter.count_in == 0
        assert self.counter.count_out == 0


class TestZoneOccupancyMonitor:
    def setup_method(self):
        self.monitor = ZoneOccupancyMonitor({
            "zone_a": [(0, 0), (200, 0), (200, 200), (0, 200)],
            "zone_b": [(200, 0), (400, 0), (400, 200), (200, 200)],
        })

    def test_point_inside(self):
        counts = self.monitor.update(
            active_track_ids=[1],
            centroids={1: (100, 100)},  # inside zone_a
        )
        assert counts["zone_a"] == 1
        assert counts["zone_b"] == 0

    def test_two_zones(self):
        counts = self.monitor.update(
            active_track_ids=[1, 2],
            centroids={1: (100, 100), 2: (300, 100)},
        )
        assert counts["zone_a"] == 1
        assert counts["zone_b"] == 1

    def test_outside_all_zones(self):
        counts = self.monitor.update(
            active_track_ids=[1],
            centroids={1: (500, 500)},  # outside both zones
        )
        assert counts["zone_a"] == 0
        assert counts["zone_b"] == 0


class TestPPECompliance:
    def setup_method(self):
        zones = {
            "work_area": [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]
        }
        rules = [
            ZoneRule(
                zone_id="work_area",
                required_ppe={PPEClass.HARD_HAT, PPEClass.SAFETY_VEST},
                name="Work Area",
            )
        ]
        self.monitor = PPEComplianceMonitor(zones, rules, iou_threshold=0.3)

    def test_compliant_worker(self):
        # Person box: [100,100,300,600] (200×500 = 100 000 px²)
        # PPE boxes overlap substantially with the person box.
        # IoU threshold is 0.30 — boxes must share ≥30% of union.
        # Use large PPE boxes that span most of the person area.
        person_boxes = np.array([[100, 100, 300, 600]], dtype=np.float32)
        ppe_boxes    = np.array([
            [100, 100, 300, 350],   # hard hat covers top half → IoU≈0.5
            [100, 300, 300, 600],   # safety vest covers bottom → IoU≈0.5
        ], dtype=np.float32)
        ppe_ids = np.array([PPEClass.HARD_HAT, PPEClass.SAFETY_VEST])
        violations = self.monitor.update(person_boxes, ppe_boxes, ppe_ids)
        assert len(violations) == 0

    def test_missing_hat_violation(self):
        person_boxes = np.array([[100, 100, 300, 600]], dtype=np.float32)
        # Only vest, no hard hat
        ppe_boxes = np.array([[120, 200, 280, 400]], dtype=np.float32)
        ppe_ids   = np.array([PPEClass.SAFETY_VEST])
        violations = self.monitor.update(person_boxes, ppe_boxes, ppe_ids)
        assert len(violations) == 1
        assert PPEClass.HARD_HAT in violations[0].missing_ppe

    def test_no_persons(self):
        violations = self.monitor.update(
            np.empty((0, 4)), np.empty((0, 4)), np.empty(0)
        )
        assert len(violations) == 0


class TestMetrics:
    def test_mot_evaluator_perfect(self):
        from edgeva.utils import MOTEvaluator, TrackletFrame
        ev = MOTEvaluator(iou_threshold=0.5)
        gt = [TrackletFrame(frame_id=0, track_id=1,
                            bbox=np.array([0, 0, 100, 100]))]
        pr = [TrackletFrame(frame_id=0, track_id=1,
                            bbox=np.array([0, 0, 100, 100]))]
        ev.update(0, gt, pr)
        m = ev.compute()
        assert m.mota == pytest.approx(100.0)
        assert m.num_fp == 0
        assert m.num_fn == 0

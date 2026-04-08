"""
PPE (Personal Protective Equipment) detection and zone compliance monitoring.

Implements the industrial safety vertical analytics described in the survey.
PPE detection uses a multi-class YOLO detector; zone compliance rules are
evaluated per-frame against detected PPE and personnel.

EU AI Act note: Industrial safety monitoring is classified as HIGH-RISK
under Annex III. This module provides detection support only;
enforcement decisions must involve a human in the loop.

References
----------
[1] Jocher et al., "Ultralytics YOLOv8", 2023.
[2] Cao et al., "Real-Time PPE Detection for Industrial Safety",
    IEEE Access 2022.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import time

import numpy as np

from .people_counting import ZoneOccupancyMonitor


# ---------------------------------------------------------------------------
# PPE class registry
# ---------------------------------------------------------------------------

class PPEClass:
    PERSON       = 0
    HARD_HAT     = 1
    SAFETY_VEST  = 2
    SAFETY_GLASSES = 3
    GLOVES       = 4
    SAFETY_BOOTS = 5
    FACE_MASK    = 6
    EAR_PROTECTION = 7

    NAMES = {
        0: "person",
        1: "hard_hat",
        2: "safety_vest",
        3: "safety_glasses",
        4: "gloves",
        5: "safety_boots",
        6: "face_mask",
        7: "ear_protection",
    }


# ---------------------------------------------------------------------------
# Zone compliance rule
# ---------------------------------------------------------------------------

@dataclass
class ZoneRule:
    """
    Required PPE for a named workplace zone.

    Attributes
    ----------
    zone_id         : identifier matching ZoneOccupancyMonitor zones
    required_ppe    : set of PPEClass IDs that must be worn
    max_occupancy   : optional maximum allowed persons in zone
    name            : human-readable zone name
    """
    zone_id: str
    required_ppe: Set[int]
    max_occupancy: Optional[int] = None
    name: str = ""

    def check_person(self, detected_ppe: Set[int]) -> List[int]:
        """Return list of missing required PPE class IDs."""
        return sorted(self.required_ppe - detected_ppe)


# ---------------------------------------------------------------------------
# Compliance event
# ---------------------------------------------------------------------------

@dataclass
class ComplianceViolation:
    """A single PPE compliance violation event."""
    timestamp: float
    zone_id: str
    zone_name: str
    person_box: np.ndarray         # [x1,y1,x2,y2] of the person
    missing_ppe: List[int]         # list of missing PPEClass IDs
    severity: str = "WARNING"      # "WARNING" | "ALERT"

    @property
    def missing_names(self) -> List[str]:
        return [PPEClass.NAMES.get(c, str(c)) for c in self.missing_ppe]

    def __str__(self) -> str:
        missing_str = ", ".join(self.missing_names)
        return (f"[{self.severity}] Zone '{self.zone_name}': "
                f"missing {missing_str}")


# ---------------------------------------------------------------------------
# Main PPE compliance monitor
# ---------------------------------------------------------------------------

class PPEComplianceMonitor:
    """
    Frame-level PPE compliance monitor.

    Associates detected persons with their PPE via spatial overlap,
    checks zone rules, and emits ComplianceViolation events.

    Parameters
    ----------
    zones         : dict {zone_id: [(x,y) vertex, …]} — see ZoneOccupancyMonitor
    rules         : list of ZoneRule objects
    iou_threshold : min IoU to associate PPE item with a person (default 0.3)
    alert_threshold : missing item count to escalate WARNING → ALERT
    """

    def __init__(
        self,
        zones: Dict[str, List[Tuple[float, float]]],
        rules: List[ZoneRule],
        iou_threshold: float = 0.3,
        alert_threshold: int = 2,
    ):
        self._zone_monitor = ZoneOccupancyMonitor(zones)
        self._rules: Dict[str, ZoneRule] = {r.zone_id: r for r in rules}
        self.iou_threshold = iou_threshold
        self.alert_threshold = alert_threshold
        self._violations: List[ComplianceViolation] = []

    # ------------------------------------------------------------------
    def update(
        self,
        person_boxes: np.ndarray,           # (N, 4) [x1,y1,x2,y2]
        ppe_boxes: np.ndarray,              # (M, 4) [x1,y1,x2,y2]
        ppe_class_ids: np.ndarray,          # (M,)   PPEClass IDs
        timestamp: Optional[float] = None,
    ) -> List[ComplianceViolation]:
        """
        Process one frame and return any new compliance violations.

        Parameters
        ----------
        person_boxes  : bounding boxes of detected persons
        ppe_boxes     : bounding boxes of detected PPE items
        ppe_class_ids : class ID for each PPE box
        timestamp     : frame timestamp (default: time.time())

        Returns
        -------
        violations : list of ComplianceViolation for this frame
        """
        ts = timestamp or time.time()
        frame_violations = []

        if len(person_boxes) == 0:
            return frame_violations

        # Identify which zone each person is in
        centroids = {
            i: (
                float((person_boxes[i, 0] + person_boxes[i, 2]) / 2),
                float((person_boxes[i, 1] + person_boxes[i, 3]) / 2),
            )
            for i in range(len(person_boxes))
        }
        self._zone_monitor.update(list(range(len(person_boxes))), centroids)
        person_zones = self._zone_monitor._track_zones   # person_idx → zone_id

        # Associate PPE with persons via IoU overlap
        person_ppe: Dict[int, Set[int]] = {i: set() for i in range(len(person_boxes))}
        if len(ppe_boxes) > 0:
            iou_matrix = self._compute_iou_matrix(person_boxes, ppe_boxes)
            for p_idx in range(len(person_boxes)):
                for ppe_idx in range(len(ppe_boxes)):
                    if iou_matrix[p_idx, ppe_idx] >= self.iou_threshold:
                        person_ppe[p_idx].add(int(ppe_class_ids[ppe_idx]))

        # Check rules for each person in a zone
        for p_idx, zone_id in person_zones.items():
            rule = self._rules.get(zone_id)
            if rule is None:
                continue
            missing = rule.check_person(person_ppe[p_idx])
            if missing:
                severity = "ALERT" if len(missing) >= self.alert_threshold else "WARNING"
                v = ComplianceViolation(
                    timestamp=ts,
                    zone_id=zone_id,
                    zone_name=rule.name,
                    person_box=person_boxes[p_idx],
                    missing_ppe=missing,
                    severity=severity,
                )
                frame_violations.append(v)
                self._violations.append(v)

        return frame_violations

    # ------------------------------------------------------------------
    def compliance_rate(self, last_n_seconds: float = 3600.0) -> Dict[str, float]:
        """
        Per-zone compliance rate over the last N seconds.

        Returns dict mapping zone_id to fraction [0, 1] of compliant checks.
        """
        cutoff = time.time() - last_n_seconds
        recent = [v for v in self._violations if v.timestamp >= cutoff]

        total: Dict[str, int] = {}
        violations: Dict[str, int] = {}
        for v in recent:
            total[v.zone_id] = total.get(v.zone_id, 0) + 1
            violations[v.zone_id] = violations.get(v.zone_id, 0) + 1

        return {
            zid: 1.0 - violations.get(zid, 0) / max(total.get(zid, 1), 1)
            for zid in total
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_iou_matrix(
        boxes_a: np.ndarray, boxes_b: np.ndarray
    ) -> np.ndarray:
        """Compute (N, M) IoU matrix between two sets of [x1,y1,x2,y2] boxes."""
        N, M = len(boxes_a), len(boxes_b)
        iou = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                x1 = max(boxes_a[i, 0], boxes_b[j, 0])
                y1 = max(boxes_a[i, 1], boxes_b[j, 1])
                x2 = min(boxes_a[i, 2], boxes_b[j, 2])
                y2 = min(boxes_a[i, 3], boxes_b[j, 3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = ((boxes_a[i, 2] - boxes_a[i, 0]) *
                          (boxes_a[i, 3] - boxes_a[i, 1]))
                area_b = ((boxes_b[j, 2] - boxes_b[j, 0]) *
                          (boxes_b[j, 3] - boxes_b[j, 1]))
                union = area_a + area_b - inter
                iou[i, j] = inter / max(union, 1e-6)
        return iou

"""
People counting and zone analytics for retail and smart office applications.

Implements line-crossing counting, zone occupancy, and dwell-time
analysis — the core analytics of the retail and smart office verticals
surveyed in the companion paper.

References
----------
[1] Li et al., "CSRNet: Dilated Convolutional Neural Networks for
    Understanding the Highly Congested Scenes", CVPR 2018.
[2] Song et al., "Rethinking Counting and Localization in Crowds",
    ICCV 2021.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ZoneCount:
    zone_id: str
    count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class DwellRecord:
    track_id: int
    zone_id: str
    enter_time: float
    exit_time: Optional[float] = None

    @property
    def dwell_seconds(self) -> Optional[float]:
        if self.exit_time is None:
            return None
        return self.exit_time - self.enter_time


# ---------------------------------------------------------------------------
# Line crossing counter
# ---------------------------------------------------------------------------

class LineCrossingCounter:
    """
    Count objects crossing a virtual line in a specified direction.

    The line is defined by two image-space points (x1,y1)–(x2,y2).
    Direction is determined by the sign of the cross product between
    the line vector and the object centroid displacement vector.

    Parameters
    ----------
    line_p1, line_p2 : (x, y) endpoints of the virtual line
    name             : human-readable label (e.g. "entrance")
    """

    def __init__(
        self,
        line_p1: Tuple[float, float],
        line_p2: Tuple[float, float],
        name: str = "line",
    ):
        self.line_p1 = np.array(line_p1, dtype=np.float32)
        self.line_p2 = np.array(line_p2, dtype=np.float32)
        self.name = name
        self._line_vec = self.line_p2 - self.line_p1

        self._prev_side: Dict[int, int] = {}   # track_id → side (-1 or +1)
        self.count_in  = 0
        self.count_out = 0

    def update(self, track_id: int, centroid: Tuple[float, float]):
        """
        Update counter for a single track's centroid position.

        Parameters
        ----------
        track_id : integer track identity
        centroid : (cx, cy) — centre of the bounding box
        """
        pt = np.array(centroid, dtype=np.float32) - self.line_p1
        # Use scalar 2-D cross product: line_vec × pt
        cross = float(self._line_vec[0]) * float(pt[1]) - float(self._line_vec[1]) * float(pt[0])
        side = int(np.sign(cross))
        if side == 0:
            return

        prev = self._prev_side.get(track_id)
        if prev is not None and prev != side:
            if side > 0:
                self.count_in += 1
            else:
                self.count_out += 1
        self._prev_side[track_id] = side

    def net_count(self) -> int:
        """Returns count_in − count_out."""
        return self.count_in - self.count_out

    def remove_track(self, track_id: int):
        self._prev_side.pop(track_id, None)

    def reset(self):
        self._prev_side.clear()
        self.count_in = 0
        self.count_out = 0

    def __repr__(self) -> str:
        return (f"LineCrossingCounter(name={self.name!r}, "
                f"in={self.count_in}, out={self.count_out})")


# ---------------------------------------------------------------------------
# Zone occupancy monitor
# ---------------------------------------------------------------------------

class ZoneOccupancyMonitor:
    """
    Monitor real-time occupancy of one or more polygonal zones.

    A zone is defined as a list of (x, y) vertices forming a
    closed polygon. Occupancy is determined by testing whether
    each track's centroid lies inside any zone.

    Parameters
    ----------
    zones : dict mapping zone_id (str) to list of (x,y) vertices
    """

    def __init__(self, zones: Dict[str, List[Tuple[float, float]]]):
        self.zones = {
            zid: np.array(verts, dtype=np.float32)
            for zid, verts in zones.items()
        }
        self._zone_counts: Dict[str, int] = {zid: 0 for zid in zones}
        self._track_zones: Dict[int, str] = {}   # track → current zone

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """Ray-casting algorithm for point-in-polygon test."""
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def update(
        self,
        active_track_ids: List[int],
        centroids: Dict[int, Tuple[float, float]],
    ) -> Dict[str, int]:
        """
        Update occupancy counts given current active tracks.

        Parameters
        ----------
        active_track_ids : list of currently active track IDs
        centroids        : dict mapping track_id to (cx, cy)

        Returns
        -------
        zone_counts : dict mapping zone_id to current occupant count
        """
        counts: Dict[str, int] = {zid: 0 for zid in self.zones}

        for tid in active_track_ids:
            if tid not in centroids:
                continue
            pt = np.array(centroids[tid], dtype=np.float32)
            for zid, poly in self.zones.items():
                if self._point_in_polygon(pt, poly):
                    counts[zid] += 1
                    self._track_zones[tid] = zid
                    break

        self._zone_counts = counts
        # Clean up stale tracks
        for tid in list(self._track_zones):
            if tid not in active_track_ids:
                del self._track_zones[tid]

        return counts

    def current_counts(self) -> Dict[str, int]:
        return dict(self._zone_counts)


# ---------------------------------------------------------------------------
# Dwell time analyser
# ---------------------------------------------------------------------------

class DwellTimeAnalyser:
    """
    Record and analyse per-track, per-zone dwell times.

    Parameters
    ----------
    zones : same zone definition as ZoneOccupancyMonitor
    min_dwell_s : minimum dwell seconds to record (filters noise)
    """

    def __init__(
        self,
        zones: Dict[str, List[Tuple[float, float]]],
        min_dwell_s: float = 2.0,
    ):
        self._monitor = ZoneOccupancyMonitor(zones)
        self.min_dwell_s = min_dwell_s

        self._active: Dict[int, DwellRecord] = {}     # tid → current record
        self._history: List[DwellRecord] = []

    def update(
        self,
        active_track_ids: List[int],
        centroids: Dict[int, Tuple[float, float]],
        timestamp: Optional[float] = None,
    ):
        """Process one frame. timestamp defaults to time.time()."""
        ts = timestamp or time.time()
        zone_counts = self._monitor.update(active_track_ids, centroids)

        current_zones = self._monitor._track_zones

        # Check for entries and exits
        for tid in active_track_ids:
            zone = current_zones.get(tid)
            if zone and tid not in self._active:
                self._active[tid] = DwellRecord(
                    track_id=tid, zone_id=zone, enter_time=ts
                )
            elif zone and tid in self._active and self._active[tid].zone_id != zone:
                # Zone change
                rec = self._active.pop(tid)
                rec.exit_time = ts
                if rec.dwell_seconds and rec.dwell_seconds >= self.min_dwell_s:
                    self._history.append(rec)
                self._active[tid] = DwellRecord(
                    track_id=tid, zone_id=zone, enter_time=ts
                )

        # Handle lost tracks
        for tid in list(self._active):
            if tid not in active_track_ids:
                rec = self._active.pop(tid)
                rec.exit_time = ts
                if rec.dwell_seconds and rec.dwell_seconds >= self.min_dwell_s:
                    self._history.append(rec)

    def stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-zone dwell-time statistics from history.

        Returns
        -------
        stats : dict mapping zone_id to {mean_s, median_s, count}
        """
        zone_dwells: Dict[str, List[float]] = defaultdict(list)
        for rec in self._history:
            if rec.dwell_seconds:
                zone_dwells[rec.zone_id].append(rec.dwell_seconds)

        result = {}
        for zid, dwells in zone_dwells.items():
            arr = np.array(dwells)
            result[zid] = {
                "mean_s":   float(arr.mean()),
                "median_s": float(np.median(arr)),
                "std_s":    float(arr.std()),
                "count":    int(len(arr)),
            }
        return result

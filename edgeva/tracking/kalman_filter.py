"""
Kalman Filter for multi-object tracking.

Implements a constant-velocity linear Kalman filter over the
(cx, cy, w, h) bounding-box state space, as used in SORT [1],
ByteTrack [2], and the LITE feature-reuse tracker [3].

References
----------
[1] Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
[2] Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
    Every Detection Box", ECCV 2022.
[3] Alikhanov et al., "LITE: A Paradigm Shift in Multi-Object Tracking
    with Efficient ReID Feature Extraction", arXiv 2024.
"""

import numpy as np


class KalmanFilter:
    """
    Constant-velocity Kalman filter for bounding-box state estimation.

    State vector:  x = [cx, cy, w, h, vx, vy, vw, vh]
    Observation:   z = [cx, cy, w, h]

    Parameters
    ----------
    std_weight_position : float
        Standard-deviation scale for position noise (default: 1/20).
    std_weight_velocity : float
        Standard-deviation scale for velocity noise (default: 1/160).
    """

    ndim = 4  # observation dimension
    dt = 1.0  # time step (one frame)

    def __init__(
        self,
        std_weight_position: float = 1.0 / 20,
        std_weight_velocity: float = 1.0 / 160,
    ):
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        # Transition matrix  F  (8×8)
        self._F = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self._F[i, self.ndim + i] = self.dt

        # Observation matrix  H  (4×8)
        self._H = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initiate(self, measurement: np.ndarray):
        """
        Create a new track from a raw bounding-box measurement.

        Parameters
        ----------
        measurement : ndarray, shape (4,)
            [cx, cy, w, h] of the detected bounding box.

        Returns
        -------
        mean : ndarray, shape (8,)
        covariance : ndarray, shape (8, 8)
        """
        mean_pos = measurement.astype(np.float32)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        std = np.array(
            [
                2 * self._std_weight_position * measurement[2],   # cx
                2 * self._std_weight_position * measurement[3],   # cy
                2 * self._std_weight_position * measurement[2],   # w
                2 * self._std_weight_position * measurement[3],   # h
                10 * self._std_weight_velocity * measurement[2],  # vx
                10 * self._std_weight_velocity * measurement[3],  # vy
                10 * self._std_weight_velocity * measurement[2],  # vw
                10 * self._std_weight_velocity * measurement[3],  # vh
            ],
            dtype=np.float32,
        )
        covariance = np.diag(std**2)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Run one step of the Kalman filter prediction.

        Returns
        -------
        mean_pred : ndarray, shape (8,)
        cov_pred  : ndarray, shape (8, 8)
        """
        std = np.array(
            [
                self._std_weight_position * mean[2],
                self._std_weight_position * mean[3],
                self._std_weight_position * mean[2],
                self._std_weight_position * mean[3],
                self._std_weight_velocity * mean[2],
                self._std_weight_velocity * mean[3],
                self._std_weight_velocity * mean[2],
                self._std_weight_velocity * mean[3],
            ],
            dtype=np.float32,
        )
        Q = np.diag(std**2)  # process noise

        mean_pred = self._F @ mean
        cov_pred = self._F @ covariance @ self._F.T + Q
        return mean_pred, cov_pred

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ):
        """
        Run one step of the Kalman filter correction.

        Parameters
        ----------
        mean        : predicted state, shape (8,)
        covariance  : predicted covariance, shape (8, 8)
        measurement : observed [cx, cy, w, h], shape (4,)

        Returns
        -------
        new_mean : ndarray, shape (8,)
        new_cov  : ndarray, shape (8, 8)
        """
        std = np.array(
            [
                self._std_weight_position * mean[2],
                self._std_weight_position * mean[3],
                self._std_weight_position * mean[2],
                self._std_weight_position * mean[3],
            ],
            dtype=np.float32,
        )
        R = np.diag(std**2)  # measurement noise

        S = self._H @ covariance @ self._H.T + R          # innovation cov
        K = covariance @ self._H.T @ np.linalg.inv(S)     # Kalman gain
        y = measurement - self._H @ mean                   # innovation

        new_mean = mean + K @ y
        new_cov = (np.eye(2 * self.ndim) - K @ self._H) @ covariance
        return new_mean, new_cov

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """
        Compute squared Mahalanobis distance between state and measurements.

        Parameters
        ----------
        mean         : shape (8,)
        covariance   : shape (8, 8)
        measurements : shape (N, 4)
        only_position : if True, use only (cx, cy) for gating

        Returns
        -------
        distances : ndarray, shape (N,)
        """
        projected_mean = self._H @ mean
        projected_cov = self._H @ covariance @ self._H.T

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        diff = measurements - projected_mean           # (N, 2 or 4)
        S_inv = np.linalg.inv(projected_cov)
        # Mahalanobis:  d_i^2 = diff_i @ S^{-1} @ diff_i^T
        return np.einsum("ni,ij,nj->n", diff, S_inv, diff)

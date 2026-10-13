from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.signal import find_peaks, peak_prominences

from .datastructures import StereogramImage


@dataclass
class PeriodEstimationResult:
    """
    Result of base period estimation on a stereogram.
    """

    period: int
    confidence: float
    method: str
    lags: np.ndarray
    metric_curve: np.ndarray  # Autocorrelation or Difference values at each lag
    peaks_or_valleys: List[int] = field(default_factory=list)
    harmonics: List[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PeriodEstimationResult(period={self.period}px, "
            f"confidence={self.confidence:.3f}, method='{self.method}')"
        )


class PeriodEstimator:
    """
    Estimates the fundamental base repeating period (S_0) of an autostereogram.

    Supports:
    - Normalized Autocorrelation (finds prominent correlation peaks)
    - Mean Absolute Difference (SAD, directly models Photoshop's 'Difference' layer shift)
    - Flexible row sampling (middle rows, distributed sample, or full image)
    - Robust aggregation (median across rows to eliminate foreground depth distortions)
    """

    def __init__(
        self,
        min_period: int = 25,
        max_period: Optional[int] = None,
        method: Literal["autocorrelation", "difference"] = "autocorrelation",
        row_sampling: Literal["distributed", "middle", "all"] = "distributed",
        row_count: int = 64,
        aggregation: Literal["median", "mean"] = "median",
    ):
        """
        Parameters:
        -----------
        min_period: Minimum lag in pixels to consider (avoids trivial micro-texture correlations near lag 0).
        max_period: Maximum lag in pixels. If None, defaults to width // 2.
        method: 'autocorrelation' (peaks) or 'difference' (Photoshop-like valleys).
        row_sampling: 'distributed' (sample evenly across height), 'middle' (center slice), 'all' (every row).
        row_count: Number of rows to sample if row_sampling is 'distributed' or 'middle'.
        aggregation: 'median' (outlier-resistant against foreground objects) or 'mean'.
        """
        self.min_period = min_period
        self.max_period = max_period
        self.method = method
        self.row_sampling = row_sampling
        self.row_count = row_count
        self.aggregation = aggregation

    def _select_rows(self, image: StereogramImage) -> np.ndarray:
        """
        Extract row indices based on the sampling strategy.
        """
        h = image.height
        if self.row_sampling == "all":
            return np.arange(h)
        elif self.row_sampling == "middle":
            # Center row_count rows
            count = min(self.row_count, h)
            start = max(0, (h - count) // 2)
            return np.arange(start, start + count)
        elif self.row_sampling == "distributed":
            count = min(self.row_count, h)
            return np.linspace(0, h - 1, count, dtype=int)
        else:
            raise ValueError(f"Unknown row_sampling: {self.row_sampling}")

    def compute_autocorrelation_curve(
        self, rows: np.ndarray, min_lag: int, max_lag: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute row-wise normalized autocorrelation curve for lags in [min_lag, max_lag].
        """
        num_rows, width = rows.shape
        lags = np.arange(min_lag, max_lag + 1)
        corr_matrix = np.zeros((num_rows, len(lags)), dtype=np.float32)

        # Standardize each row (zero mean, unit variance)
        row_means = np.mean(rows, axis=1, keepdims=True)
        row_stds = np.std(rows, axis=1, keepdims=True)
        row_stds[row_stds == 0] = 1.0  # avoid division by zero
        norm_rows = (rows - row_means) / row_stds

        for idx, lag in enumerate(lags):
            # Overlapping window: x in [0, width - lag] and x + lag in [lag, width]
            left = norm_rows[:, : width - lag]
            right = norm_rows[:, lag:]
            # Normalized cross-correlation for this lag
            corr_matrix[:, idx] = np.mean(left * right, axis=1)

        # Aggregate across rows
        if self.aggregation == "median":
            curve = np.median(corr_matrix, axis=0)
        else:
            curve = np.mean(corr_matrix, axis=0)

        return lags, curve

    def compute_difference_curve(
        self, rows: np.ndarray, min_lag: int, max_lag: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Mean Absolute Difference (SAD) curve for lags in [min_lag, max_lag].
        Simulates sliding the image horizontally in Photoshop with Difference blend mode.
        """
        num_rows, width = rows.shape
        lags = np.arange(min_lag, max_lag + 1)
        diff_matrix = np.zeros((num_rows, len(lags)), dtype=np.float32)

        for idx, lag in enumerate(lags):
            left = rows[:, : width - lag]
            right = rows[:, lag:]
            diff_matrix[:, idx] = np.mean(np.abs(left - right), axis=1)

        if self.aggregation == "median":
            curve = np.median(diff_matrix, axis=0)
        else:
            curve = np.mean(diff_matrix, axis=0)

        return lags, curve

    def estimate(self, image: StereogramImage) -> PeriodEstimationResult:
        """
        Analyze the stereogram and return the estimated base period with diagnostic data.
        """
        max_lag = self.max_period or min(image.width // 2, 400)
        min_lag = max(2, self.min_period)

        if max_lag <= min_lag:
            raise ValueError(f"max_period ({max_lag}) must be greater than min_period ({min_lag})")

        row_indices = self._select_rows(image)
        sampled_rows = image.gray[row_indices]

        if self.method == "autocorrelation":
            lags, curve = self.compute_autocorrelation_curve(sampled_rows, min_lag, max_lag)
            peaks_idx, _ = find_peaks(
                curve,
                distance=max(5, self.min_period // 2),
            )
            detected_lags = lags[peaks_idx] if len(peaks_idx) > 0 else np.array([])

            if len(peaks_idx) == 0:
                best_idx = int(np.argmax(curve))
                best_period = int(lags[best_idx])
                confidence = float(curve[best_idx])
                harmonics = []
            else:
                prominences = peak_prominences(curve, peaks_idx)[0]
                max_prom = np.max(prominences)
                # Keep peaks with at least 30% of maximum prominence to avoid local object ripples
                sig_mask = prominences >= max(0.04, 0.3 * max_prom)
                sig_lags = detected_lags[sig_mask]

                if len(sig_lags) > 0:
                    best_period = int(sig_lags[0])
                    confidence = float(np.clip(max_prom / (np.ptp(curve) + 1e-6), 0.0, 1.0))
                    harmonics = [int(lag) for lag in sig_lags[1:]]
                else:
                    best_period = int(detected_lags[np.argmax(prominences)])
                    confidence = float(curve[np.where(lags == best_period)[0][0]])
                    harmonics = []

        elif self.method == "difference":
            lags, curve = self.compute_difference_curve(sampled_rows, min_lag, max_lag)
            # In difference, we search for local VALLEYS (minima), so invert curve for find_peaks
            inverted = -curve
            valleys_idx, _ = find_peaks(
                inverted,
                distance=max(5, self.min_period // 2),
            )
            detected_lags = lags[valleys_idx] if len(valleys_idx) > 0 else np.array([])

            if len(valleys_idx) == 0:
                best_idx = int(np.argmin(curve))
                best_period = int(lags[best_idx])
                confidence = 0.5
                harmonics = []
            else:
                prominences = peak_prominences(inverted, valleys_idx)[0]
                max_prom = np.max(prominences)
                # Keep valleys with at least 30% of maximum prominence
                sig_mask = prominences >= max(0.02, 0.3 * max_prom)
                sig_lags = detected_lags[sig_mask]

                if len(sig_lags) > 0:
                    best_period = int(sig_lags[0])
                    confidence = float(np.clip(max_prom / (np.ptp(curve) + 1e-6), 0.0, 1.0))
                    harmonics = [int(lag) for lag in sig_lags[1:]]
                else:
                    best_period = int(detected_lags[np.argmax(prominences)])
                    confidence = 0.5
                    harmonics = []

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return PeriodEstimationResult(
            period=best_period,
            confidence=confidence,
            method=self.method,
            lags=lags,
            metric_curve=curve,
            peaks_or_valleys=detected_lags,
            harmonics=harmonics,
            metadata={
                "row_sampling": self.row_sampling,
                "row_count": len(row_indices),
                "aggregation": self.aggregation,
                "min_lag": min_lag,
                "max_lag": max_lag,
            },
        )


def estimate_base_period(
    image: Union[StereogramImage, np.ndarray, str],
    method: Literal["autocorrelation", "difference"] = "autocorrelation",
    sampling: Literal["distributed", "middle", "all"] = "distributed",
    aggregation: Literal["median", "mean"] = "median",
    min_period: int = 25,
    max_period: Optional[int] = None,
) -> PeriodEstimationResult:
    """
    Convenience function to estimate base period of an autostereogram.
    """
    if isinstance(image, str):
        stereo_img = StereogramImage.from_file(image)
    elif isinstance(image, np.ndarray):
        stereo_img = StereogramImage.from_numpy(image)
    else:
        stereo_img = image

    estimator = PeriodEstimator(
        min_period=min_period,
        max_period=max_period,
        method=method,
        row_sampling=sampling,
        aggregation=aggregation,
    )
    return estimator.estimate(stereo_img)

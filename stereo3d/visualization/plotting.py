"""
Visualization utilities for period estimation and disparity analysis.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from ..core.datastructures import StereogramImage
from ..core.period import PeriodEstimationResult


def plot_period_analysis(
    result: PeriodEstimationResult,
    image: StereogramImage,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot diagnostic analysis showing:
    1. Metric curve (Autocorrelation or Photoshop Difference) with identified period & harmonics.
    2. The shifted Difference Map at the detected base period, visualizing how background cancels
       out and foreground object contours emerge.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Metric curve
    ax_curve = axes[0]
    ax_curve.plot(result.lags, result.metric_curve, color="#1f77b4", lw=2, label=f"{result.method.title()} Curve")

    # Mark the fundamental period
    period_idx = np.where(result.lags == result.period)[0]
    if len(period_idx) > 0:
        val = result.metric_curve[period_idx[0]]
        marker_color = "#2ca02c" if result.method == "autocorrelation" else "#d62728"
        ax_curve.scatter([result.period], [val], color=marker_color, s=90, zorder=5, label=f"Base Period: {result.period}px")
        ax_curve.axvline(result.period, color=marker_color, linestyle="--", alpha=0.7)

    # Mark harmonics if found
    for h in result.harmonics:
        h_idx = np.where(result.lags == h)[0]
        if len(h_idx) > 0:
            h_val = result.metric_curve[h_idx[0]]
            ax_curve.scatter([h], [h_val], color="#ff7f0e", s=40, zorder=4)
            ax_curve.axvline(h, color="#ff7f0e", linestyle=":", alpha=0.4)

    ax_curve.set_title(f"Period Analysis ({result.method.title()}): Base Period = {result.period}px (Conf: {result.confidence:.2f})")
    ax_curve.set_xlabel("Horizontal Shift / Lag (pixels)")
    ax_curve.set_ylabel("Autocorrelation" if result.method == "autocorrelation" else "Mean Absolute Difference")
    ax_curve.grid(True, linestyle="--", alpha=0.5)
    ax_curve.legend()

    # Panel 2: Difference map at the detected period
    ax_diff = axes[1]
    diff_map = image.compute_difference_map(result.period)
    im = ax_diff.imshow(diff_map, cmap="inferno", vmin=0.0, vmax=np.percentile(diff_map, 99))
    ax_diff.set_title(f"Shifted Difference Map at Lag = {result.period}px\n(Photoshop Difference: Silhouette Emergence)")
    ax_diff.set_axis_off()
    fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04, label="Pixel Difference")

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig

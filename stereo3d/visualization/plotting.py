"""
Visualization utilities for period estimation and depth map analysis.
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
    1. Metric curve (Autocorrelation or Mean Absolute Difference) with identified period and harmonics.
    2. The shifted difference map at the detected base period, visualizing pattern cancellation.
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
    ax_diff.set_title(f"Shifted Difference Map at Lag = {result.period}px\n(Background Cancellation and Contour Emergence)")
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


def plot_depth_analysis(
    image: StereogramImage,
    depth_map: np.ndarray,
    period_result: Optional[PeriodEstimationResult] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot comprehensive depth reconstruction analysis:
    1. Original input stereogram image.
    2. Difference map at the base period (if available).
    3. Reconstructed depth map with colormap.
    """
    num_panels = 3 if period_result is not None else 2
    fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 5))

    # Panel 1: Original Stereogram
    axes[0].imshow(image.rgb)
    axes[0].set_title(f"Input Stereogram ({image.width}x{image.height})")
    axes[0].set_axis_off()

    idx = 1
    # Panel 2: Difference map at period
    if period_result is not None:
        diff_map = image.compute_difference_map(period_result.period)
        axes[idx].imshow(diff_map, cmap="inferno", vmin=0.0, vmax=np.percentile(diff_map, 99))
        axes[idx].set_title(f"Shifted Difference Map ({period_result.period}px)")
        axes[idx].set_axis_off()
        idx += 1

    # Panel 3: Recovered Depth Map
    im_depth = axes[idx].imshow(depth_map, cmap="turbo", vmin=0.0, vmax=1.0)
    axes[idx].set_title("Reconstructed Depth Map")
    axes[idx].set_axis_off()
    fig.colorbar(im_depth, ax=axes[idx], fraction=0.046, pad=0.04, label="Normalized Depth")

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_postprocess_analysis(
    raw_depth: np.ndarray,
    spike_mask: np.ndarray,
    post_depth: np.ndarray,
    cropped_margins: Tuple[int, int] = (0, 0),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot 3-panel comparative analysis for depth map post-processing:
    1. Raw Extracted Depth Map (with margin crop indicators if applicable).
    2. Flagged Noise / Spike Mask (inpaint targets).
    3. Clean Post-Processed Depth Map.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Raw Depth
    im_raw = axes[0].imshow(raw_depth, cmap="turbo", vmin=0.0, vmax=1.0)
    m_left, m_right = cropped_margins
    if m_left > 0 or m_right > 0:
        axes[0].axvline(m_left, color="#ff3333", linestyle="--", lw=1.5, label=f"Left Margin ({m_left}px)")
        axes[0].axvline(raw_depth.shape[1] - m_right, color="#ff3333", linestyle="--", lw=1.5, label=f"Right Margin ({m_right}px)")
        axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"1. Raw Depth ({raw_depth.shape[1]}x{raw_depth.shape[0]})")
    axes[0].set_axis_off()

    # Panel 2: Detected Noise / Spike Mask
    spike_pct = float(np.mean(spike_mask > 0) * 100.0)
    axes[1].imshow(spike_mask, cmap="inferno", vmin=0, vmax=255)
    axes[1].set_title(f"2. Flagged Noise Mask ({spike_pct:.2f}% Inpainted)")
    axes[1].set_axis_off()

    # Panel 3: Post-Processed Depth
    im_post = axes[2].imshow(post_depth, cmap="turbo", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"3. Post-Processed Depth ({post_depth.shape[1]}x{post_depth.shape[0]})")
    axes[2].set_axis_off()
    fig.colorbar(im_post, ax=axes[2], fraction=0.046, pad=0.04, label="Normalized Depth")

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig

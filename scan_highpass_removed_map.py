#!/usr/bin/env python3
"""Scan an image with local Gaussian-windowed FFT and map a local FFT metric.

The scan samples the image on a stride grid (default step=10), prints an
estimated CPU time before processing, and shows progress markers when the
estimated runtime is greater than 15 seconds.
"""

from __future__ import annotations

import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from fft_image_explorer import find_top_fft_peaks


def load_image_and_luminance(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    rgb = np.asarray(img).astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return rgb, y.astype(np.float32)


def make_gaussian_window(size: int = 100, softness: float = 0.2) -> np.ndarray:
    # Match the window behavior used in the explorer: normalized coords over full width/height.
    half = size / 2.0
    sigma = max(softness, 0.03)
    coords = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    xn = xx / max(half, 1.0)
    yn = yy / max(half, 1.0)
    win = np.exp(-0.5 * ((xn / sigma) ** 2 + (yn / sigma) ** 2)).astype(np.float32)
    return win


def nearest_coord_indices(length: int, coords: np.ndarray) -> np.ndarray:
    """For each integer index [0..length-1], choose nearest value in sorted coords."""
    idx = np.arange(length, dtype=np.float32)
    right = np.searchsorted(coords, idx, side="left")
    right = np.clip(right, 0, len(coords) - 1)
    left = np.clip(right - 1, 0, len(coords) - 1)

    choose_right = np.abs(coords[right] - idx) < np.abs(coords[left] - idx)
    out = left.copy()
    out[choose_right] = right[choose_right]
    return out.astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map local FFT metric using 100x100 Gaussian-windowed FFT."
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "--highpass-percent",
        type=float,
        required=True,
        help="High-pass radius percentage (0..100), same meaning as in fft_image_explorer.py",
    )
    parser.add_argument(
        "--metric",
        choices=["hp_removed", "first_non_origin_peak"],
        default="hp_removed",
        help="Metric to map (default: hp_removed)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Stride in pixels for sampling (default: 10)",
    )
    parser.add_argument(
        "--display-scale",
        choices=["linear", "near100"],
        default="linear",
        help="Color scaling for output map (default: linear)",
    )
    parser.add_argument(
        "--near100-alpha",
        type=float,
        default=0.35,
        help="Exponent for near100 scaling; smaller values increase detail near 100 (default: 0.35)",
    )
    args = parser.parse_args()

    hp_percent = float(np.clip(args.highpass_percent, 0.0, 100.0))
    metric = args.metric
    step = max(1, int(args.step))
    display_scale = args.display_scale
    near100_alpha = max(0.05, float(args.near100_alpha))

    rgb, y = load_image_and_luminance(args.image)
    h, w = y.shape

    win_size = 100
    softness = 0.2
    window = make_gaussian_window(size=win_size, softness=softness)

    # Edge padding so every image pixel can be used as a window center.
    pad = win_size // 2
    y_pad = np.pad(y, ((pad, pad), (pad, pad)), mode="edge")

    # High-pass mask for a fixed 100x100 local FFT grid.
    fh = fw = win_size
    fcy, fcx = fh // 2, fw // 2
    yy, xx = np.ogrid[:fh, :fw]
    max_radius_px = math.hypot(fh / 2.0, fw / 2.0)
    hp_radius_px = hp_percent * 0.01 * max_radius_px
    low_freq_mask = (yy - fcy) ** 2 + (xx - fcx) ** 2 <= hp_radius_px ** 2
    hp_keep_mask = ~low_freq_mask

    def eval_metric_at(iy: int, ix: int) -> float:
        patch = y_pad[iy:iy + win_size, ix:ix + win_size]
        patch_w = patch * window
        f = np.fft.fftshift(np.fft.fft2(patch_w))
        base_power = float((np.abs(f) ** 2).sum())

        if base_power <= 1e-12:
            return 0.0
        if metric == "hp_removed":
            hp_only_power = float((np.abs(f * hp_keep_mask) ** 2).sum())
            return 100.0 * (base_power - hp_only_power) / base_power

        mag_raw = np.log1p(np.abs(f))
        peaks = find_top_fft_peaks(mag_raw)
        for p in peaks:
            if float(p.get("distance", 0.0)) > 0.0:
                return float(p.get("peak_val", 0.0))
        return 0.0

    ys = np.arange(0, h, step, dtype=np.int32)
    xs = np.arange(0, w, step, dtype=np.int32)
    ly_n = int(ys.shape[0])
    lx_n = int(xs.shape[0])
    total = ly_n * lx_n

    calib_n = min(200, total)
    calib_ids = np.linspace(0, max(total - 1, 0), num=calib_n, dtype=np.int64)
    t_cal0 = time.process_time()
    for fid in calib_ids:
        iy = int(ys[int(fid // lx_n)])
        ix = int(xs[int(fid % lx_n)])
        _ = eval_metric_at(iy, ix)
    t_cal = time.process_time() - t_cal0
    pps_est = calib_n / max(t_cal, 1e-9)
    est_cpu_s = total / max(pps_est, 1e-9)
    print(
        f"Estimated CPU time: {est_cpu_s:.3f}s "
        f"for {total} points (step={step}) at ~{pps_est:.1f} points/sec CPU"
    )

    show_progress = est_cpu_s > 15.0
    next_progress_pct = 5

    lattice_vals = np.zeros((ly_n, lx_n), dtype=np.float32)
    t_run0 = time.process_time()
    done = 0
    for ly, iy in enumerate(ys):
        for lx, ix in enumerate(xs):
            lattice_vals[ly, lx] = eval_metric_at(int(iy), int(ix))
            done += 1
            if show_progress:
                pct = int((100.0 * done) / max(total, 1))
                while pct >= next_progress_pct and next_progress_pct <= 100:
                    print(f"Progress: {next_progress_pct}% ({done}/{total})")
                    next_progress_pct += 5

    cpu_elapsed = time.process_time() - t_run0
    pps = done / max(cpu_elapsed, 1e-9)

    ny = nearest_coord_indices(h, ys.astype(np.float32))
    nx = nearest_coord_indices(w, xs.astype(np.float32))
    out = lattice_vals[ny[:, None], nx[None, :]].copy()

    print(
        f"Completed stride scan: CPU time {cpu_elapsed:.3f}s, "
        f"processed={done}/{total}, step={step}, {pps:.1f} points/sec CPU"
    )
    _show_map(
        rgb,
        out,
        hp_percent,
        metric=metric,
        processed=done,
        total=total,
        step=step,
        display_scale=display_scale,
        near100_alpha=near100_alpha,
    )


def _show_map(
    rgb: np.ndarray,
    out: np.ndarray,
    hp_percent: float,
    metric: str,
    processed: int,
    total: int,
    step: int,
    display_scale: str,
    near100_alpha: float,
) -> None:
    if metric == "hp_removed":
        disp = np.clip(out, 0.0, 100.0)
        vmin, vmax = 0.0, 100.0
        cbar_label = "High-pass removed power (%)"
        raw_ticks = [0, 20, 40, 60, 80, 90, 95, 98, 99, 100]
    else:
        disp = np.clip(out, 0.0, 4.0)
        vmin, vmax = 0.0, 4.0
        cbar_label = "First non-origin peak value (log magnitude)"
        raw_ticks = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

    ticks = [0, 20, 40, 60, 80, 90, 95, 98, 99, 100]
    tick_labels = None

    if display_scale == "near100" and metric == "hp_removed":
        # Expand contrast near 100%: y = 1 - (1-x)^alpha, alpha<1 magnifies near 1.
        x = disp / 100.0
        disp = 100.0 * (1.0 - np.power(1.0 - x, near100_alpha))
        cbar_label = f"Near-100 enhanced scale (alpha={near100_alpha:.2f})"
        ticks = [
            100.0 * (1.0 - (1.0 - t / 100.0) ** near100_alpha)
            for t in [0, 20, 40, 60, 80, 90, 95, 98, 99, 100]
        ]
        tick_labels = ["0", "20", "40", "60", "80", "90", "95", "98", "99", "100"]
    else:
        ticks = raw_ticks

    fig, (ax_src, ax_map) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    ax_src.imshow(rgb, interpolation="nearest")
    ax_src.set_title("Original image")

    im = ax_map.imshow(disp, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_map.set_title("FFT-derived map")

    h, w = out.shape

    def _fmt_xy_values(x: float, y: float) -> str:
        ix = int(round(x))
        iy = int(round(y))
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return f"x={x:.1f}, y={y:.1f}"

        raw_val = float(out[iy, ix])
        disp_val = float(disp[iy, ix])
        return (
            f"x={ix}, y={iy}, display={disp_val:.3f}, raw={raw_val:.3f}"
        )

    # Keep toolbar readout useful when non-linear display scaling is active.
    ax_map.format_coord = _fmt_xy_values
    ax_src.format_coord = _fmt_xy_values

    cbar = fig.colorbar(im, ax=ax_map, label=cbar_label, fraction=0.046, pad=0.04)
    cbar.set_ticks(ticks)
    if tick_labels is not None:
        cbar.set_ticklabels(tick_labels)
    fig.suptitle(
        "Local Gaussian FFT high-pass removed map\n"
        "window=100x100, softness=0.2, stride sampling, "
        f"step={step}, hp={hp_percent:.2f}%, metric={metric}, "
        f"processed={processed}/{total}"
    )

    ax_src.axis("off")
    ax_map.axis("off")
    plt.show()


if __name__ == "__main__":
    main()

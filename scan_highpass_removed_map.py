#!/usr/bin/env python3
"""Scan an image with local Gaussian-windowed FFT and map a local FFT metric.

Pass 1: sample exactly 10,000 evenly spaced points (100x100 grid).
Pass 2: process every pixel in any pass-1 region below threshold and all adjacent
regions (8-neighborhood on the 100x100 region grid).

Before pass 2 starts, the script prints an estimated CPU time based on pass-1 throughput.
"""

from __future__ import annotations

import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from fft_image_explorer import find_top_fft_peaks


def load_luminance(path: str) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    rgb = np.asarray(img).astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return y.astype(np.float32)


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


def expand_adjacent_regions(mask: np.ndarray) -> np.ndarray:
    """Expand a coarse region mask to include all 8-connected neighbors."""
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= p[1 + dy:1 + dy + mask.shape[0], 1 + dx:1 + dx + mask.shape[1]]
    return out


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
        default="first_non_origin_peak",
        help="Metric to map (default: first_non_origin_peak)",
    )
    parser.add_argument(
        "--exclude-threshold",
        type=float,
        default=None,
        help="Exclude pixels >= this value from second-pass sampling; defaults by metric",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for second-pass random sampling",
    )
    parser.add_argument(
        "--no-second-pass",
        action="store_true",
        help="Only run pass 1 (10k points + nearest-neighbor fill)",
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
    if args.exclude_threshold is None:
        exclude_threshold = 99.0 if metric == "hp_removed" else 1.5
    else:
        exclude_threshold = float(args.exclude_threshold)
    _ = args.seed  # Reserved for future stochastic variants.
    no_second_pass = bool(args.no_second_pass)
    display_scale = args.display_scale
    near100_alpha = max(0.05, float(args.near100_alpha))

    y = load_luminance(args.image)
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

    # Exactly 10,000 evenly spaced sample points over image coordinates.
    grid_n = 100
    sample_ys = np.linspace(0, h - 1, num=grid_n, dtype=np.float32)
    sample_xs = np.linspace(0, w - 1, num=grid_n, dtype=np.float32)
    sample_ys_i = np.rint(sample_ys).astype(np.int32)
    sample_xs_i = np.rint(sample_xs).astype(np.int32)
    sampled = np.zeros((grid_n, grid_n), dtype=np.float32)

    processed = 0
    total = grid_n * grid_n
    cpu_start = time.process_time()

    for iy_idx, iy in enumerate(sample_ys_i):
        for ix_idx, ix in enumerate(sample_xs_i):
            patch = y_pad[iy:iy + win_size, ix:ix + win_size]
            patch_w = patch * window

            f = np.fft.fftshift(np.fft.fft2(patch_w))
            base_power = float((np.abs(f) ** 2).sum())

            if base_power <= 1e-12:
                metric_val = 0.0
            else:
                if metric == "hp_removed":
                    hp_only_power = float((np.abs(f * hp_keep_mask) ** 2).sum())
                    metric_val = 100.0 * (base_power - hp_only_power) / base_power
                else:
                    mag_raw = np.log1p(np.abs(f))
                    peaks = find_top_fft_peaks(mag_raw)
                    metric_val = 0.0
                    for p in peaks:
                        if float(p.get("distance", 0.0)) > 0.0:
                            metric_val = float(p.get("peak_val", 0.0))
                            break

            sampled[iy_idx, ix_idx] = metric_val
            processed += 1

    pass1_cpu_elapsed = time.process_time() - cpu_start

    # Fill unsampled pixels by nearest sampled neighbor on the 100x100 sample grid.
    ny = nearest_coord_indices(h, sample_ys)
    nx = nearest_coord_indices(w, sample_xs)
    out = sampled[ny[:, None], nx[None, :]].copy()

    second_count = 0
    second_target = 0
    if not no_second_pass:
        # Pass 2: process every pixel in low-threshold regions and all adjacent regions.
        low_regions = sampled < exclude_threshold
        region_mask = expand_adjacent_regions(low_regions)
        pixel_mask = region_mask[ny[:, None], nx[None, :]]

        second_pixels = np.argwhere(pixel_mask)
        second_count = int(second_pixels.shape[0])
        second_target = second_count

        pps_est = processed / max(pass1_cpu_elapsed, 1e-9)
        est_cpu_s = second_count / max(pps_est, 1e-9)
        print(
            f"Estimated pass-2 CPU time: {est_cpu_s:.3f}s "
            f"for {second_count} pixels at ~{pps_est:.1f} pixels/sec CPU"
        )

        if second_count > 0:
            for iy, ix in second_pixels:
                patch = y_pad[iy:iy + win_size, ix:ix + win_size]
                patch_w = patch * window

                f = np.fft.fftshift(np.fft.fft2(patch_w))
                base_power = float((np.abs(f) ** 2).sum())

                if base_power <= 1e-12:
                    metric_val = 0.0
                else:
                    if metric == "hp_removed":
                        hp_only_power = float((np.abs(f * hp_keep_mask) ** 2).sum())
                        metric_val = 100.0 * (base_power - hp_only_power) / base_power
                    else:
                        mag_raw = np.log1p(np.abs(f))
                        peaks = find_top_fft_peaks(mag_raw)
                        metric_val = 0.0
                        for p in peaks:
                            if float(p.get("distance", 0.0)) > 0.0:
                                metric_val = float(p.get("peak_val", 0.0))
                                break

                out[iy, ix] = metric_val

    total_processed = processed + second_count

    cpu_elapsed = time.process_time() - cpu_start
    pps = total_processed / max(cpu_elapsed, 1e-9)
    if no_second_pass:
        print(
            f"Completed one-pass scan: CPU time {cpu_elapsed:.3f}s, "
            f"pass1={processed}/{total}, total={total_processed}, "
            f"{pps:.1f} points/sec CPU"
        )
    else:
        print(
            f"Completed two-pass sampled scan: CPU time {cpu_elapsed:.3f}s, "
            f"pass1={processed}/{total}, pass2={second_count}/{second_target} "
            f"(region+adjacent), "
            f"total={total_processed}, {pps:.1f} points/sec CPU"
        )
    _show_map(
        out,
        hp_percent,
        metric=metric,
        processed=processed,
        total=total,
        second_count=second_count,
        second_target=second_target,
        exclude_threshold=exclude_threshold,
        display_scale=display_scale,
        near100_alpha=near100_alpha,
    )


def _show_map(
    out: np.ndarray,
    hp_percent: float,
    metric: str,
    processed: int,
    total: int,
    second_count: int,
    second_target: int,
    exclude_threshold: float,
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

    plt.figure(figsize=(8, 6))
    im = plt.imshow(disp, cmap="magma", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, label=cbar_label)
    cbar.set_ticks(ticks)
    if tick_labels is not None:
        cbar.set_ticklabels(tick_labels)
    if second_target > 0:
        title_mode = (
            f"pass2={second_count}/{second_target} (<{exclude_threshold:.2f} + adjacent)"
        )
    else:
        title_mode = "pass2=disabled"
    plt.title(
        "Local Gaussian FFT high-pass removed map\n"
        "window=100x100, softness=0.2, pass1=100x100 (NN fill), "
        f"{title_mode}, hp={hp_percent:.2f}%, metric={metric}, "
        f"pass1 processed={processed}/{total}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

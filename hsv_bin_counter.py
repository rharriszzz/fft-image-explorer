#!/usr/bin/env python3
"""Standalone HSV fixed-bin pixel counter and interactive visualizer.

This script is intentionally independent from other local Python files.
It loads an image, computes packed HSV values, counts pixels in fixed HSV bins,
and shows an interactive bar chart. Clicking a bar opens a window that shows only
pixels matching that HSV bin, with all other pixels set to white.
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image, ImageOps


def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def packed_hsv_from_rgb(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb_to_hsv(np.clip(rgb, 0.0, 1.0).astype(np.float32))
    hsv_u8 = np.clip(np.round(hsv * 255.0), 0, 255).astype(np.uint8)
    h = hsv_u8[..., 0].astype(np.uint32)
    s = hsv_u8[..., 1].astype(np.uint32)
    v = hsv_u8[..., 2].astype(np.uint32)
    return (h << 16) | (s << 8) | v


def top_exact_hsv_counts(
    packed_hsv: np.ndarray,
    mask: np.ndarray,
    top_k: int = 20,
) -> tuple[int, list[tuple[int, int, int, int]]]:
    vals = packed_hsv[mask]
    if vals.size == 0:
        return 0, []
    uniq, cnt = np.unique(vals, return_counts=True)
    order = np.argsort(cnt)[::-1]
    top: list[tuple[int, int, int, int]] = []
    for idx in order[:top_k]:
        p = int(uniq[idx])
        c = int(cnt[idx])
        h = (p >> 16) & 0xFF
        s = (p >> 8) & 0xFF
        v = p & 0xFF
        top.append((h, s, v, c))
    return int(uniq.size), top


def top_hsv_volume_counts(
    packed_hsv: np.ndarray,
    mask: np.ndarray,
    coverage_target: float = 0.995,
    max_bars: int = 140,
    h_bins: int = 24,
    s_bins: int = 3,
    v_bins: int = 4,
    extreme_v_fraction: float = 0.05,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    vals = packed_hsv[mask]
    if vals.size == 0:
        return [], {
            "coverage": 0.0,
            "h_bins": 0,
            "s_bins": 0,
            "v_bins": 0,
            "bars": 0,
            "occupied_bins": 0,
            "total_pixels": 0,
            "mode": "fixed-bins-with-extreme-v",
        }

    h = ((vals >> 16) & 0xFF).astype(np.int32)
    s = ((vals >> 8) & 0xFF).astype(np.int32)
    v = (vals & 0xFF).astype(np.int32)

    target = float(np.clip(coverage_target, 0.0, 1.0))
    max_bars = max(1, int(max_bars))
    total = int(vals.size)
    target_pixels = int(math.ceil(target * float(total)))

    frac = float(np.clip(extreme_v_fraction, 0.0, 0.49))
    edge_span = int(math.ceil(frac * 256.0))
    if edge_span <= 0:
        low_v0, low_v1 = 0, -1
        high_v0, high_v1 = 256, 255
    else:
        low_v0, low_v1 = 0, min(255, edge_span - 1)
        high_v0, high_v1 = max(0, 255 - edge_span), 255

    rows: list[dict[str, float | int]] = []

    low_mask = (v >= low_v0) & (v <= low_v1)
    low_count = int(np.count_nonzero(low_mask))
    if low_count > 0:
        rows.append(
            {
                "hc": 127.5,
                "sc": 127.5,
                "vc": float(0.5 * (low_v0 + low_v1)),
                "count": low_count,
                "h0": 0,
                "h1": 255,
                "s0": 0,
                "s1": 255,
                "v0": int(low_v0),
                "v1": int(low_v1),
            }
        )

    high_mask = (v >= high_v0) & (v <= high_v1)
    high_count = int(np.count_nonzero(high_mask))
    if high_count > 0:
        rows.append(
            {
                "hc": 127.5,
                "sc": 127.5,
                "vc": float(0.5 * (high_v0 + high_v1)),
                "count": high_count,
                "h0": 0,
                "h1": 255,
                "s0": 0,
                "s1": 255,
                "v0": int(high_v0),
                "v1": int(high_v1),
            }
        )

    if edge_span <= 0:
        mid_lo, mid_hi = 0, 255
    else:
        mid_lo, mid_hi = low_v1 + 1, high_v0 - 1

    middle_groups = max(1, int(v_bins))
    middle_v_ranges: list[tuple[int, int]] = []
    if mid_lo <= mid_hi:
        mid_len = mid_hi - mid_lo + 1
        for i in range(middle_groups):
            a = mid_lo + (i * mid_len) // middle_groups
            b = mid_lo + ((i + 1) * mid_len) // middle_groups - 1
            a = int(np.clip(a, mid_lo, mid_hi))
            b = int(np.clip(b, mid_lo, mid_hi))
            if a <= b:
                middle_v_ranges.append((a, b))

    if middle_v_ranges:
        hb = np.clip((h * h_bins) // 256, 0, h_bins - 1)
        sb = np.clip((s * s_bins) // 256, 0, s_bins - 1)

        for v_idx, (v0, v1) in enumerate(middle_v_ranges):
            vm = (v >= v0) & (v <= v1)
            if not np.any(vm):
                continue

            pb = (np.int64(v_idx) << 40) | (hb[vm].astype(np.int64) << 20) | sb[vm].astype(np.int64)
            uniq_pb, cnt_pb = np.unique(pb, return_counts=True)
            for p, c in zip(uniq_pb, cnt_pb):
                hbi = int((int(p) >> 20) & 0xFFFFF)
                sbi = int(int(p) & 0xFFFFF)
                h0 = int(np.clip((hbi * 256) // h_bins, 0, 255))
                h1 = int(np.clip((((hbi + 1) * 256) // h_bins) - 1, 0, 255))
                s0 = int(np.clip((sbi * 256) // s_bins, 0, 255))
                s1 = int(np.clip((((sbi + 1) * 256) // s_bins) - 1, 0, 255))
                rows.append(
                    {
                        "hc": float(0.5 * (h0 + h1)),
                        "sc": float(0.5 * (s0 + s1)),
                        "vc": float(0.5 * (v0 + v1)),
                        "count": int(c),
                        "h0": h0,
                        "h1": h1,
                        "s0": s0,
                        "s1": s1,
                        "v0": int(v0),
                        "v1": int(v1),
                    }
                )

    rows.sort(key=lambda row: int(row["count"]), reverse=True)
    if rows:
        counts = np.asarray([int(r["count"]) for r in rows], dtype=np.int64)
        cumulative = np.cumsum(counts)
        need = int(np.searchsorted(cumulative, target_pixels, side="left") + 1)
        use_n = int(np.clip(need, 1, min(max_bars, len(rows))))
        chosen_rows = rows[:use_n]
        covered_pixels = int(cumulative[use_n - 1])
    else:
        chosen_rows = []
        covered_pixels = 0

    coverage = covered_pixels / float(total) if total > 0 else 0.0
    meta: dict[str, float | int] = {
        "coverage": float(coverage),
        "h_bins": int(h_bins),
        "s_bins": int(s_bins),
        "v_bins": int(v_bins),
        "bars": int(len(chosen_rows)),
        "occupied_bins": int(len(rows)),
        "total_pixels": total,
        "extreme_v_fraction": float(frac),
        "extreme_v_span": int(edge_span),
        "mode": "fixed-bins-with-extreme-v",
    }
    return chosen_rows, meta


def plot_hsv_count_summary(
    rgb_source: np.ndarray,
    packed_hsv: np.ndarray,
    mask: np.ndarray,
    extreme_v_fraction: float = 0.10,
    h_bins: int = 24,
    s_bins: int = 3,
    v_bins: int = 4,
) -> None:
    d, _top = top_exact_hsv_counts(packed_hsv, mask, top_k=20)
    rows, meta = top_hsv_volume_counts(
        packed_hsv,
        mask,
        coverage_target=0.995,
        max_bars=144,
        h_bins=h_bins,
        s_bins=s_bins,
        v_bins=v_bins,
        extreme_v_fraction=extreme_v_fraction,
    )

    print(
        "HSV distinct values: "
        f"whole-image={d}"
    )
    print(
        "HSV volume coverage: "
        f"whole-image={100.0 * float(meta['coverage']):.2f}% "
        f"with fixed bins size=({int(meta['h_bins'])},{int(meta['s_bins'])},{int(meta['v_bins'])}) "
        f"bars={int(meta['bars'])}; "
        f"edgeV={float(meta.get('extreme_v_fraction', 0.0)):.3f}."
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)

    h_u8 = ((packed_hsv >> 16) & 0xFF).astype(np.uint8)
    s_u8 = ((packed_hsv >> 8) & 0xFF).astype(np.uint8)
    v_u8 = (packed_hsv & 0xFF).astype(np.uint8)

    def show_hsv_match_window(row: dict[str, float | int]) -> None:
        h0, h1 = int(row["h0"]), int(row["h1"])
        s0, s1 = int(row["s0"]), int(row["s1"])
        v0, v1 = int(row["v0"]), int(row["v1"])

        hsv_match = (
            (h_u8 >= h0) & (h_u8 <= h1)
            & (s_u8 >= s0) & (s_u8 <= s1)
            & (v_u8 >= v0) & (v_u8 <= v1)
        )
        show_mask = hsv_match & mask

        out = np.ones_like(rgb_source, dtype=np.float32)
        out[show_mask] = rgb_source[show_mask]

        fig_match, ax_match = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        ax_match.imshow(out, interpolation="nearest")
        ax_match.set_title(
            "HSV bin match\n"
            f"H[{h0}-{h1}] S[{s0}-{s1}] V[{v0}-{v1}] matched={int(np.count_nonzero(show_mask))}"
        )
        ax_match.set_xticks([])
        ax_match.set_yticks([])
        plt.show(block=False)
        plt.pause(0.001)

    if not rows:
        ax.set_title("HSV fixed-bin counts\n(no pixels)")
        ax.axis("off")
        plt.show()
        return

    counts = np.asarray([int(row["count"]) for row in rows], dtype=np.int64)
    plot_counts = np.log10(np.maximum(counts, 1))
    colors = [
        hsv_to_rgb(
            np.array(
                [
                    np.clip(float(row["hc"]) / 255.0, 0.0, 1.0),
                    np.clip(float(row["sc"]) / 255.0, 0.0, 1.0),
                    np.clip(float(row["vc"]) / 255.0, 0.0, 1.0),
                ],
                dtype=np.float32,
            )
        )
        for row in rows
    ]

    y = np.arange(len(rows))
    bars = ax.barh(y, plot_counts, color=colors, alpha=0.95, edgecolor="black", linewidth=0.6)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("log10(count)")
    ax.set_title(
        "HSV fixed-bin counts (whole image)\n"
        f"coverage={100.0 * float(meta['coverage']):.2f}% "
        f"size(H,S,V)=({int(meta['h_bins'])},{int(meta['s_bins'])},{int(meta['v_bins'])}), "
        f"bars={int(meta['bars'])}, edgeV={float(meta.get('extreme_v_fraction', 0.0)):.3f}"
    )
    ax.grid(True, axis="x", alpha=0.25)

    labels = [
        (
            f"H[{int(row['h0'])}-{int(row['h1'])}] "
            f"S[{int(row['s0'])}-{int(row['s1'])}] "
            f"V[{int(row['v0'])}-{int(row['v1'])}]\n"
            f"count={int(row['count'])} (click to show pixels)"
        )
        for row in rows
    ]

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "black", "ec": "white", "alpha": 0.85},
        color="white",
        fontsize=8,
        zorder=10,
    )
    annot.set_visible(False)

    def on_move(event):
        if event.inaxes != ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        for i, bar in enumerate(bars):
            hit, _ = bar.contains(event)
            if hit:
                x = event.xdata if event.xdata is not None else float(bar.get_width())
                yloc = event.ydata if event.ydata is not None else float(bar.get_y() + 0.5 * bar.get_height())
                annot.xy = (x, yloc)
                annot.set_text(labels[i])
                if not annot.get_visible():
                    annot.set_visible(True)
                fig.canvas.draw_idle()
                return

        if annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or getattr(event, "button", None) != 1:
            return
        for i, bar in enumerate(bars):
            hit, _ = bar.contains(event)
            if hit:
                show_hsv_match_window(rows[i])
                return

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Count image pixels in fixed HSV bins and show an interactive summary.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "--extreme-v-fraction",
        type=float,
        default=0.05,
        help="Fraction of V range near black/white treated as full H/S coverage in fixed HSV bins (default: 0.05).",
    )
    parser.add_argument("--hsv-h-bins", type=int, default=24, help="Number of H bins (default: 24).")
    parser.add_argument("--hsv-s-bins", type=int, default=3, help="Number of S bins (default: 3).")
    parser.add_argument("--hsv-v-bins", type=int, default=4, help="Number of middle V bins (default: 4).")
    args = parser.parse_args()

    rgb = load_image_rgb(args.image)
    packed_hsv = packed_hsv_from_rgb(rgb)
    mask = np.ones(packed_hsv.shape, dtype=bool)

    plot_hsv_count_summary(
        rgb_source=rgb,
        packed_hsv=packed_hsv,
        mask=mask,
        extreme_v_fraction=float(np.clip(args.extreme_v_fraction, 0.0, 0.49)),
        h_bins=max(1, int(args.hsv_h_bins)),
        s_bins=max(1, int(args.hsv_s_bins)),
        v_bins=max(1, int(args.hsv_v_bins)),
    )


if __name__ == "__main__":
    main()

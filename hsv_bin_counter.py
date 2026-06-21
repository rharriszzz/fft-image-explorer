#!/usr/bin/env python3
"""Standalone HSV pixel counter and visualizer.

This script is intentionally independent from other local Python files.
It loads an image, computes packed HSV values, and can either:
1) count pixels in fixed HSV bins (interactive), or
2) count exact unique HSV values in a spline-derived region and plot log-counts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.path import Path as MplPath
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


def _coerce_closed_curve_xy(curve_xy: object) -> np.ndarray | None:
    arr = np.asarray(curve_xy, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        return None
    if np.hypot(*(arr[0] - arr[-1])) > 1e-9:
        arr = np.vstack([arr, arr[0]])
    return arr


def _inside_polygon_mask(shape: tuple[int, int], curve_xy: np.ndarray) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = MplPath(curve_xy).contains_points(points, radius=1e-9)
    return inside.reshape(h, w)


def _spline_region_mask(packed_hsv: np.ndarray, spline_json_path: str) -> np.ndarray:
    data = json.loads(Path(spline_json_path).read_text(encoding="utf-8"))
    outer_curve = _coerce_closed_curve_xy(data.get("outer_spline"))
    inner_curve = _coerce_closed_curve_xy(data.get("inner_spline"))
    if outer_curve is None:
        raise RuntimeError("Spline JSON does not contain a valid outer_spline.")
    if inner_curve is None:
        raise RuntimeError("Spline JSON does not contain a valid inner_spline.")

    shape = packed_hsv.shape
    inside_outer = _inside_polygon_mask(shape, outer_curve)
    inside_inner = _inside_polygon_mask(shape, inner_curve)

    # Region requested by user: pixels outside outer OR inside inner.
    return (~inside_outer) | inside_inner


def plot_exact_hsv_counts_log(packed_hsv: np.ndarray, mask: np.ndarray, title: str) -> None:
    # Exclude the dominant HSV box before exact counting.
    # Requested exclusion: H 219..242, S 100..180, V 88..250.
    h_u8_full = ((packed_hsv >> 16) & 0xFF).astype(np.uint8)
    s_u8_full = ((packed_hsv >> 8) & 0xFF).astype(np.uint8)
    v_u8_full = (packed_hsv & 0xFF).astype(np.uint8)
    excluded_box = (
        (h_u8_full >= 219) & (h_u8_full <= 242)
        & (s_u8_full >= 100) & (s_u8_full <= 200)
        & (v_u8_full >= 80) & (v_u8_full <= 250)
    )

    effective_mask = mask & (~excluded_box)
    vals = packed_hsv[effective_mask]
    if vals.size == 0:
        print("No pixels selected by mask.")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax.set_title(title + "\n(no selected pixels)")
        ax.axis("off")
        plt.show()
        return

    uniq, cnt = np.unique(vals, return_counts=True)
    order = np.argsort(cnt)[::-1]
    uniq = uniq[order]
    cnt = cnt[order]

    h_u8 = ((uniq >> 16) & 0xFF).astype(np.uint8)
    s_u8 = ((uniq >> 8) & 0xFF).astype(np.uint8)
    v_u8 = (uniq & 0xFF).astype(np.uint8)

    selected_pixels = int(vals.size)
    excluded_pixels = int(np.count_nonzero(mask & excluded_box))
    unique_n = int(uniq.size)
    print(f"Selected pixels: {selected_pixels}")
    print(f"Excluded by HSV box: {excluded_pixels}")
    print(f"Unique exact HSV values: {unique_n}")
    print("Top 20 exact HSV values by count:")
    for i in range(min(20, unique_n)):
        c = int(cnt[i])
        h = int(h_u8[i])
        s = int(s_u8[i])
        v = int(v_u8[i])
        print(f"  {i + 1:2d}. HSV=({h:3d},{s:3d},{v:3d}) count={c}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 7), constrained_layout=True)
    ranks = np.arange(1, unique_n + 1, dtype=np.int64)
    log_counts = np.log10(cnt.astype(np.float64))
    ax.plot(ranks, log_counts, color="#1f77b4", linewidth=1.4)

    # Draw colored points for the most frequent exact HSV values so the
    # actual represented colors are visible directly on the rank plot.
    max_scatter_points = 4000
    show_n = min(unique_n, max_scatter_points)
    hsv_top = np.column_stack(
        [
            h_u8[:show_n].astype(np.float32) / 255.0,
            s_u8[:show_n].astype(np.float32) / 255.0,
            v_u8[:show_n].astype(np.float32) / 255.0,
        ]
    )
    rgb_top = hsv_to_rgb(hsv_top)
    scat = ax.scatter(
        ranks[:show_n],
        log_counts[:show_n],
        c=rgb_top,
        s=18,
        edgecolors="black",
        linewidths=0.2,
        alpha=0.95,
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_xlabel("HSV rank by frequency (log scale)")
    ax.set_ylabel("log10(count)")
    ax.set_title(
        f"{title}\n"
        f"selected_pixels={selected_pixels}, excluded={excluded_pixels}, unique_hsv={unique_n}, shown_colors={show_n}"
    )
    ax.grid(True, alpha=0.3)

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

    def _tooltip_text(i: int) -> str:
        return (
            f"rank={i + 1}\n"
            f"HSV=({int(h_u8[i])},{int(s_u8[i])},{int(v_u8[i])})\n"
            f"count={int(cnt[i])}, log10={float(log_counts[i]):.4f}"
        )

    def on_move(event):
        if event.inaxes != ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return
        hit, info = scat.contains(event)
        inds = info.get("ind")
        if (not hit) or inds is None or len(inds) == 0:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        i = int(inds[0])
        annot.xy = (float(ranks[i]), float(log_counts[i]))
        annot.set_text(_tooltip_text(i))
        if not annot.get_visible():
            annot.set_visible(True)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or getattr(event, "button", None) != 1:
            return
        hit, info = scat.contains(event)
        inds = info.get("ind")
        if (not hit) or inds is None or len(inds) == 0:
            return
        i = int(inds[0])
        print(
            "Clicked exact HSV: "
            f"rank={i + 1}, HSV=({int(h_u8[i])},{int(s_u8[i])},{int(v_u8[i])}), "
            f"count={int(cnt[i])}, log10={float(log_counts[i]):.4f}"
        )

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Also provide a compact top-N swatch panel with explicit HSV labels.
    swatch_n = min(40, unique_n)
    fig_sw, ax_sw = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)
    y = np.arange(swatch_n)
    sw_hsv = np.column_stack(
        [
            h_u8[:swatch_n].astype(np.float32) / 255.0,
            s_u8[:swatch_n].astype(np.float32) / 255.0,
            v_u8[:swatch_n].astype(np.float32) / 255.0,
        ]
    )
    sw_rgb = hsv_to_rgb(sw_hsv)
    sw_log_counts = np.log10(cnt[:swatch_n].astype(np.float64))
    ax_sw.barh(y, sw_log_counts, color=sw_rgb, edgecolor="black", linewidth=0.4)
    xmax = float(np.max(sw_log_counts)) if swatch_n > 0 else 1.0
    ax_sw.set_xlim(0.0, max(1.0, xmax * 1.05))
    ax_sw.set_xlabel("log10(count)")
    ax_sw.set_yticks(y)
    ax_sw.set_yticklabels(
        [
            (
                f"#{i + 1:02d}  HSV=({int(h_u8[i])},{int(s_u8[i])},{int(v_u8[i])})  "
                f"count={int(cnt[i])}"
            )
            for i in range(swatch_n)
        ],
        fontsize=8,
    )
    ax_sw.invert_yaxis()
    ax_sw.set_title(
        f"Top {swatch_n} exact HSV values with colors "
        f"(excluding H[219-242], S[100-180], V[88-250])"
    )

    plt.show()


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
        "--spline-json",
        default=None,
        help=(
            "Optional spline JSON path. When provided, counts exact HSV values only in the "
            "region: outside outer spline OR inside inner spline, and plots log-counts."
        ),
    )
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

    if args.spline_json:
        region_mask = _spline_region_mask(packed_hsv, args.spline_json)
        plot_exact_hsv_counts_log(
            packed_hsv=packed_hsv,
            mask=region_mask,
            title="Exact HSV counts (outside outer OR inside inner)",
        )
        return

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

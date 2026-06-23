#!/usr/bin/env python3
"""Interactive HSV axis explorer.

Usage example:
  python hsv_axis_slice_explorer.py beads-photo-2.jpg --axis h

The main figure shows a 1D profile:
- x-axis: selected HSV channel (H, S, or V)
- y-axis: log10(count + 1)

Click anywhere on the 1D plot to inspect that channel bin in a second figure,
which shows the other two HSV channels as a 2D histogram:
- color: log10(count + 1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.widgets import RadioButtons, Slider
from PIL import Image, ImageOps


def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def hsv_u8_from_rgb(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb_to_hsv(np.clip(rgb, 0.0, 1.0).astype(np.float32))
    return np.clip(np.round(hsv * 255.0), 0, 255).astype(np.uint8)


def axis_index(axis: str) -> int:
    a = axis.lower()
    if a in {"h", "hue"}:
        return 0
    if a in {"s", "sat", "saturation"}:
        return 1
    if a in {"v", "val", "value"}:
        return 2
    raise ValueError(f"Unsupported axis: {axis}")


def axis_name(idx: int) -> str:
    return ["Hue", "Saturation", "Value"][idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive HSV axis and slice visualizer.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--axis",
        default="h",
        choices=["h", "s", "v", "hue", "saturation", "value"],
        help="Channel to place on the main horizontal axis",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=256,
        help="Number of bins for the selected axis profile (default: 256)",
    )
    parser.add_argument(
        "--slice-bins",
        type=int,
        default=256,
        help="Number of bins per axis in the 2D slice heatmap (default: 256)",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    rgb = load_image_rgb(str(image_path))
    hsv_u8 = hsv_u8_from_rgb(rgb)

    h = hsv_u8[..., 0].ravel()
    s = hsv_u8[..., 1].ravel()
    v = hsv_u8[..., 2].ravel()
    channels = [h, s, v]

    bins_1d = max(2, int(args.bins))
    bins_2d = max(2, int(args.slice_bins))

    edges_1d = np.linspace(0.0, 256.0, num=bins_1d + 1, dtype=np.float64)
    centers_1d = 0.5 * (edges_1d[:-1] + edges_1d[1:])
    fig_main, ax_main = plt.subplots(1, 1, figsize=(12, 5.5))
    fig_main.subplots_adjust(left=0.08, right=0.78, bottom=0.24, top=0.9)

    state = {
        "main_idx": axis_index(args.axis),
        "width": 1,
        "selected_x": 127.5,
    }

    axis_vals = channels[state["main_idx"]]
    counts_1d, _ = np.histogram(axis_vals, bins=bins_1d, range=(0, 256))
    log_counts_1d = np.log10(counts_1d.astype(np.float64) + 1.0)

    (line_main,) = ax_main.plot(centers_1d, log_counts_1d, color="#1f77b4", linewidth=1.8)
    ax_main.set_ylabel("log10(count + 1)")
    ax_main.grid(alpha=0.3)
    ax_main.set_xlim(0, 255)

    selected_line = ax_main.axvline(state["selected_x"], color="crimson", linestyle="--", linewidth=1.2, alpha=0.9)
    selected_line_min = ax_main.axvline(state["selected_x"], color="crimson", linestyle="--", linewidth=1.2, alpha=0.9)
    selected_line_max = ax_main.axvline(state["selected_x"], color="crimson", linestyle="--", linewidth=1.2, alpha=0.9)
    selected_line_min.set_visible(False)
    selected_line_max.set_visible(False)
    strip_ax = None

    def get_axis_context() -> tuple[int, str, int, int, str, str, np.ndarray]:
        main_idx = int(state["main_idx"])
        main_name = axis_name(main_idx)
        other_idxs = [i for i in (0, 1, 2) if i != main_idx]
        x2_idx, y2_idx = other_idxs[0], other_idxs[1]
        x2_name = axis_name(x2_idx)
        y2_name = axis_name(y2_idx)
        return main_idx, main_name, x2_idx, y2_idx, x2_name, y2_name, channels[main_idx]

    def configure_main_axis() -> None:
        nonlocal strip_ax
        main_idx, main_name, _, _, x2_name, y2_name, _ = get_axis_context()
        ax_main.set_title(f"{main_name} profile (click to inspect {x2_name}/{y2_name})")
        if main_idx == 0:
            ax_main.set_xlabel("")
            ax_main.tick_params(axis="x", labelbottom=False)
            if strip_ax is None:
                hue_strip = np.linspace(0.0, 1.0, num=256, dtype=np.float32)
                hsv_strip = np.zeros((1, 256, 3), dtype=np.float32)
                hsv_strip[0, :, 0] = hue_strip
                hsv_strip[0, :, 1] = 1.0
                hsv_strip[0, :, 2] = 1.0
                rgb_strip = hsv_to_rgb(hsv_strip)
                strip_ax = ax_main.inset_axes([0.0, -0.16, 1.0, 0.08], transform=ax_main.transAxes)
                strip_ax.imshow(rgb_strip, aspect="auto", interpolation="nearest", extent=[0, 255, 0, 1])
                strip_ax.set_yticks([])
                strip_ax.set_xlim(0, 255)
                strip_ax.set_xticks([0, 64, 128, 192, 255])
                strip_ax.set_xlabel("Hue (color strip)")
        else:
            ax_main.tick_params(axis="x", labelbottom=True)
            ax_main.set_xlabel(f"{main_name} bin")
            if strip_ax is not None:
                strip_ax.remove()
                strip_ax = None

    def update_profile() -> None:
        _, _, _, _, _, _, axis_vals_local = get_axis_context()
        counts_local, _ = np.histogram(axis_vals_local, bins=bins_1d, range=(0, 256))
        line_main.set_ydata(np.log10(counts_local.astype(np.float64) + 1.0))
        ax_main.relim()
        ax_main.autoscale_view(scalex=False, scaley=True)

    fig_slice: plt.Figure | None = None
    ax_slice = None
    ax_color = None
    im_slice = None
    im_color = None
    cbar_slice = None

    def compute_window(center_x: float, circular: bool) -> tuple[int, int]:
        width = int(state["width"])
        center_i = int(np.clip(np.round(center_x), 0, 255))
        lo_i = center_i - ((width - 1) // 2)
        if not circular:
            lo_i = int(np.clip(lo_i, 0, 256 - width))
        hi_i = lo_i + width
        return lo_i, hi_i

    def show_slice(center_x: float) -> None:
        nonlocal fig_slice, ax_slice, ax_color, im_slice, im_color, cbar_slice

        main_idx, main_name, x2_idx, y2_idx, x2_name, y2_name, axis_vals_local = get_axis_context()
        is_hue_axis = main_idx == 0
        lo_i, hi_i = compute_window(center_x, circular=is_hue_axis)
        lo = float(lo_i)
        hi = float(hi_i)

        if is_hue_axis:
            lo_mod = lo_i % 256
            hi_mod = hi_i % 256
            wraps = (lo_i < 0) or (hi_i > 256)
            state["selected_x"] = float(np.clip(np.round(center_x), 0, 255))
            if wraps:
                range_text = f"[{lo_mod:.0f}, 256) U [0, {hi_mod:.0f})"
                mask = (axis_vals_local >= lo_mod) | (axis_vals_local < hi_mod)
            else:
                range_text = f"[{lo_mod:.0f}, {hi_mod:.0f})"
                mask = (axis_vals_local >= lo_mod) & (axis_vals_local < hi_mod)
            line_min_x = float(lo_mod)
            line_max_x = float(hi_mod)
        else:
            state["selected_x"] = float(0.5 * (lo + hi))
            range_text = f"[{lo:.0f}, {hi:.0f})"
            mask = (axis_vals_local >= lo_i) & (axis_vals_local < hi_i)
            line_min_x = float(lo_i)
            line_max_x = float(hi_i)

        if int(state["width"]) == 1:
            selected_line.set_xdata([state["selected_x"], state["selected_x"]])
            selected_line.set_visible(True)
            selected_line_min.set_visible(False)
            selected_line_max.set_visible(False)
        else:
            selected_line.set_visible(False)
            selected_line_min.set_xdata([line_min_x, line_min_x])
            selected_line_max.set_xdata([line_max_x, line_max_x])
            selected_line_min.set_visible(True)
            selected_line_max.set_visible(True)
        n_selected = int(np.count_nonzero(mask))

        x2 = channels[x2_idx][mask]
        y2 = channels[y2_idx][mask]

        counts_2d, y_edges, x_edges = np.histogram2d(
            y2,
            x2,
            bins=[bins_2d, bins_2d],
            range=[[0, 256], [0, 256]],
        )
        log_counts_2d = np.log10(counts_2d + 1.0)
        max_count_2d = int(np.max(counts_2d)) if counts_2d.size > 0 else 0
        vmax_2d = float(np.max(log_counts_2d)) if log_counts_2d.size > 0 else 0.0
        vmax_2d = max(vmax_2d, 1e-6)

        x_centers_2d = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers_2d = 0.5 * (y_edges[:-1] + y_edges[1:])
        x_grid, y_grid = np.meshgrid(x_centers_2d, y_centers_2d)
        fixed_main_val = float(state["selected_x"])
        if is_hue_axis:
            fixed_main_val = float(fixed_main_val % 256.0)
        hsv_plane = np.zeros((bins_2d, bins_2d, 3), dtype=np.float32)
        hsv_plane[..., main_idx] = np.clip(fixed_main_val / 255.0, 0.0, 1.0)
        hsv_plane[..., x2_idx] = np.clip(x_grid / 255.0, 0.0, 1.0)
        hsv_plane[..., y2_idx] = np.clip(y_grid / 255.0, 0.0, 1.0)
        rgb_plane = hsv_to_rgb(hsv_plane)

        if fig_slice is None or ax_slice is None or ax_color is None:
            fig_slice, (ax_slice, ax_color) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
            im_slice = ax_slice.imshow(
                log_counts_2d,
                origin="lower",
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                cmap="magma",
                interpolation="nearest",
                aspect="auto",
                vmin=0.0,
                vmax=vmax_2d,
            )
            cbar_slice = fig_slice.colorbar(im_slice, ax=ax_slice)
            cbar_slice.set_label("log10(count + 1)")
            ax_slice.set_xlim(0, 255)
            ax_slice.set_ylim(0, 255)
            ax_slice.set_xlabel(x2_name)
            ax_slice.set_ylabel(y2_name)

            im_color = ax_color.imshow(
                rgb_plane,
                origin="lower",
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                interpolation="nearest",
                aspect="auto",
            )
            ax_color.set_xlim(0, 255)
            ax_color.set_ylim(0, 255)
            ax_color.set_xlabel(x2_name)
            ax_color.set_ylabel(y2_name)
        else:
            assert im_slice is not None
            assert im_color is not None
            im_slice.set_data(log_counts_2d)
            im_slice.set_extent([x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
            im_slice.set_clim(0.0, vmax_2d)
            im_color.set_data(rgb_plane)
            im_color.set_extent([x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
            ax_slice.set_xlabel(x2_name)
            ax_slice.set_ylabel(y2_name)
            ax_color.set_xlabel(x2_name)
            ax_color.set_ylabel(y2_name)

        ax_slice.set_title(
            f"{main_name} in {range_text} | width={int(state['width'])} | "
            f"pixels={n_selected}, max_bin_count={max_count_2d}"
        )
        ax_color.set_title(f"HSV colors at {main_name}={fixed_main_val:.1f}")

        fig_main.canvas.draw_idle()
        if fig_slice is not None:
            fig_slice.canvas.draw_idle()
            fig_slice.show()

    def on_click(event) -> None:
        if event.inaxes != ax_main:
            return
        if event.xdata is None:
            return
        show_slice(float(event.xdata))

    def on_axis_change(label: str) -> None:
        state["main_idx"] = ["Hue", "Saturation", "Value"].index(label)
        configure_main_axis()
        update_profile()
        show_slice(float(state["selected_x"]))
        fig_main.canvas.draw_idle()

    def on_width_change(width_val: float) -> None:
        state["width"] = int(np.clip(np.round(width_val), 1, 256))
        show_slice(float(state["selected_x"]))

    radio_ax = fig_main.add_axes([0.81, 0.50, 0.17, 0.22])
    radio = RadioButtons(radio_ax, ["Hue", "Saturation", "Value"], active=int(state["main_idx"]))
    radio_ax.set_title("Main axis", fontsize=10)
    radio.on_clicked(on_axis_change)

    width_ax = fig_main.add_axes([0.12, 0.04, 0.62, 0.05])
    width_slider = Slider(width_ax, "Width (values)", 1, 256, valinit=int(state["width"]), valstep=1)
    width_slider.on_changed(on_width_change)

    fig_main.canvas.mpl_connect("button_press_event", on_click)

    configure_main_axis()
    update_profile()

    # Show an initial slice for convenience.
    show_slice(float(state["selected_x"]))
    plt.show()


if __name__ == "__main__":
    main()

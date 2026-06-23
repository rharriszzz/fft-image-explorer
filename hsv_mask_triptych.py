#!/usr/bin/env python3
"""Interactive HSV mask triptych viewer.

Shows three synchronized views:
1) original image
2) original with masked pixels whitened
3) original with unmasked pixels whitened

Controls:
- Hue center + width (circular over 0..255)
- Saturation center + width
- Value center + width

Left-click on the original image to set H/S/V centers from that pixel.

Optional preset mode:
- Union of:
    1) red: 242 <= H <= 262 and V >= 128 (wraps to H<=6)
    2) yellow: 8 <= H <= 44
    3) black: V <= 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.widgets import CheckButtons, Slider
from PIL import Image, ImageOps


def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def hsv_u8_from_rgb(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb_to_hsv(np.clip(rgb, 0.0, 1.0).astype(np.float32))
    return np.clip(np.round(hsv * 255.0), 0, 255).astype(np.uint8)


def channel_window_mask(values_u8: np.ndarray, center: int, width: int, circular: bool) -> np.ndarray:
    width_i = int(np.clip(width, 1, 256))
    center_i = int(np.clip(center, 0, 255))

    if width_i >= 256:
        return np.ones(values_u8.shape, dtype=bool)

    lo = center_i - ((width_i - 1) // 2)
    hi = lo + width_i  # exclusive

    vals = values_u8.astype(np.int16)
    if circular:
        lo_mod = lo % 256
        hi_mod = hi % 256
        if lo >= 0 and hi <= 256:
            return (vals >= lo_mod) & (vals < hi_mod)
        return (vals >= lo_mod) | (vals < hi_mod)

    lo_clamped = int(np.clip(lo, 0, 256 - width_i))
    hi_clamped = lo_clamped + width_i
    return (vals >= lo_clamped) & (vals < hi_clamped)


def _hsv_preset_components(h_u8: np.ndarray, v_u8: np.ndarray) -> dict[str, np.ndarray]:
    # Condition 1: red hue range with minimum value (wrapped 242..262).
    red = ((h_u8 >= 242) | (h_u8 <= 6)) & (v_u8 >= 128)

    # Condition 2: yellow hue range.
    yellow = (h_u8 >= 8) & (h_u8 <= 44)

    # Condition 3: black by value threshold.
    black = v_u8 <= 64

    return {"red": red, "yellow": yellow, "black": black}


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive HSV mask viewer with center/width controls.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--h", type=int, default=128, help="Initial hue center (0..255)")
    parser.add_argument("--h-width", type=int, default=24, help="Initial hue width (1..256)")
    parser.add_argument("--s", type=int, default=128, help="Initial saturation center (0..255)")
    parser.add_argument("--s-width", type=int, default=64, help="Initial saturation width (1..256)")
    parser.add_argument("--v", type=int, default=128, help="Initial value center (0..255)")
    parser.add_argument("--v-width", type=int, default=64, help="Initial value width (1..256)")
    parser.add_argument(
        "--preset-union-mask",
        action="store_true",
        help=(
            "Use fixed union mask: red(H 242..262 wrap, V>=128) OR yellow(H 8..44) OR black(V<=64). "
            "When enabled, slider values do not affect mask output."
        ),
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    rgb = load_image_rgb(str(image_path))
    hsv_u8 = hsv_u8_from_rgb(rgb)

    h_ch = hsv_u8[..., 0]
    s_ch = hsv_u8[..., 1]
    v_ch = hsv_u8[..., 2]

    fig, (ax_orig, ax_masked_white, ax_unmasked_white) = plt.subplots(1, 3, figsize=(16, 7))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.30, wspace=0.03)

    im_orig = ax_orig.imshow(rgb, interpolation="nearest")
    _ = im_orig
    masked_img = rgb.copy()
    unmasked_img = rgb.copy()
    im_masked = ax_masked_white.imshow(masked_img, interpolation="nearest")
    im_unmasked = ax_unmasked_white.imshow(unmasked_img, interpolation="nearest")

    marker = ax_orig.scatter([0], [0], s=40, c="none", edgecolors="cyan", linewidths=1.2)
    marker.set_visible(False)

    for ax in (ax_orig, ax_masked_white, ax_unmasked_white):
        ax.set_xticks([])
        ax.set_yticks([])

    ax_orig.set_title("Original (left-click to set H/S/V centers)")

    slider_h_ax = fig.add_axes([0.10, 0.205, 0.82, 0.025])
    slider_hw_ax = fig.add_axes([0.10, 0.172, 0.82, 0.025])
    slider_s_ax = fig.add_axes([0.10, 0.125, 0.82, 0.025])
    slider_sw_ax = fig.add_axes([0.10, 0.092, 0.82, 0.025])
    slider_v_ax = fig.add_axes([0.10, 0.045, 0.82, 0.025])
    slider_vw_ax = fig.add_axes([0.10, 0.012, 0.82, 0.025])

    preview_h_ax = fig.add_axes([0.10, 0.232, 0.82, 0.015])
    preview_s_ax = fig.add_axes([0.10, 0.152, 0.82, 0.015])
    preview_v_ax = fig.add_axes([0.10, 0.072, 0.82, 0.015])

    for p_ax in (preview_h_ax, preview_s_ax, preview_v_ax):
        p_ax.set_yticks([])
        p_ax.set_xlim(0, 255)
        p_ax.set_xticks([0, 64, 128, 192, 255])

    preview_h_ax.set_xlabel("Hue sweep (current S,V)", fontsize=8)
    preview_s_ax.set_xlabel("Sat sweep (current H,V)", fontsize=8)
    preview_v_ax.set_xlabel("Val sweep (current H,S)", fontsize=8)

    slider_h = Slider(slider_h_ax, "Hue", 0, 255, valinit=int(np.clip(args.h, 0, 255)), valstep=1)
    slider_hw = Slider(slider_hw_ax, "Hue width", 1, 256, valinit=int(np.clip(args.h_width, 1, 256)), valstep=1)
    slider_s = Slider(slider_s_ax, "Sat", 0, 255, valinit=int(np.clip(args.s, 0, 255)), valstep=1)
    slider_sw = Slider(slider_sw_ax, "Sat width", 1, 256, valinit=int(np.clip(args.s_width, 1, 256)), valstep=1)
    slider_v = Slider(slider_v_ax, "Val", 0, 255, valinit=int(np.clip(args.v, 0, 255)), valstep=1)
    slider_vw = Slider(slider_vw_ax, "Val width", 1, 256, valinit=int(np.clip(args.v_width, 1, 256)), valstep=1)

    preset_components = _hsv_preset_components(h_ch, v_ch)
    preset_enabled = {"red": True, "yellow": True, "black": True}
    preset_checks = None
    if args.preset_union_mask:
        checks_ax = fig.add_axes([0.82, 0.73, 0.15, 0.14])
        preset_checks = CheckButtons(checks_ax, ["red", "yellow", "black"], [True, True, True])
        checks_ax.set_title("Preset parts", fontsize=9)

    x_vals = np.linspace(0.0, 1.0, num=256, dtype=np.float32)

    h_strip_hsv = np.zeros((1, 256, 3), dtype=np.float32)
    h_strip_hsv[0, :, 0] = x_vals
    h_strip_hsv[0, :, 1] = float(slider_s.val) / 255.0
    h_strip_hsv[0, :, 2] = float(slider_v.val) / 255.0
    im_h_preview = preview_h_ax.imshow(
        hsv_to_rgb(h_strip_hsv),
        interpolation="nearest",
        aspect="auto",
        extent=[0, 255, 0, 1],
    )

    s_strip_hsv = np.zeros((1, 256, 3), dtype=np.float32)
    s_strip_hsv[0, :, 0] = float(slider_h.val) / 255.0
    s_strip_hsv[0, :, 1] = x_vals
    s_strip_hsv[0, :, 2] = float(slider_v.val) / 255.0
    im_s_preview = preview_s_ax.imshow(
        hsv_to_rgb(s_strip_hsv),
        interpolation="nearest",
        aspect="auto",
        extent=[0, 255, 0, 1],
    )

    v_strip_hsv = np.zeros((1, 256, 3), dtype=np.float32)
    v_strip_hsv[0, :, 0] = float(slider_h.val) / 255.0
    v_strip_hsv[0, :, 1] = float(slider_s.val) / 255.0
    v_strip_hsv[0, :, 2] = x_vals
    im_v_preview = preview_v_ax.imshow(
        hsv_to_rgb(v_strip_hsv),
        interpolation="nearest",
        aspect="auto",
        extent=[0, 255, 0, 1],
    )

    def update_views() -> None:
        h_center = int(slider_h.val)
        h_width = int(slider_hw.val)
        s_center = int(slider_s.val)
        s_width = int(slider_sw.val)
        v_center = int(slider_v.val)
        v_width = int(slider_vw.val)

        h_strip_hsv[0, :, 0] = x_vals
        h_strip_hsv[0, :, 1] = float(s_center) / 255.0
        h_strip_hsv[0, :, 2] = float(v_center) / 255.0
        im_h_preview.set_data(hsv_to_rgb(h_strip_hsv))

        s_strip_hsv[0, :, 0] = float(h_center) / 255.0
        s_strip_hsv[0, :, 1] = x_vals
        s_strip_hsv[0, :, 2] = float(v_center) / 255.0
        im_s_preview.set_data(hsv_to_rgb(s_strip_hsv))

        v_strip_hsv[0, :, 0] = float(h_center) / 255.0
        v_strip_hsv[0, :, 1] = float(s_center) / 255.0
        v_strip_hsv[0, :, 2] = x_vals
        im_v_preview.set_data(hsv_to_rgb(v_strip_hsv))

        if args.preset_union_mask:
            mask = np.zeros(h_ch.shape, dtype=bool)
            for k, on in preset_enabled.items():
                if on:
                    mask |= preset_components[k]
        else:
            h_mask = channel_window_mask(h_ch, h_center, h_width, circular=True)
            s_mask = channel_window_mask(s_ch, s_center, s_width, circular=False)
            v_mask = channel_window_mask(v_ch, v_center, v_width, circular=False)
            mask = h_mask & s_mask & v_mask

        masked_img_local = rgb.copy()
        masked_img_local[mask] = 1.0
        unmasked_img_local = rgb.copy()
        unmasked_img_local[~mask] = 1.0

        im_masked.set_data(masked_img_local)
        im_unmasked.set_data(unmasked_img_local)

        selected = int(np.count_nonzero(mask))
        total = int(mask.size)
        unselected = total - selected
        pct = 100.0 * (selected / float(max(total, 1)))
        ax_masked_white.set_title("Masked pixels set to white")
        ax_unmasked_white.set_title("Unmasked pixels set to white")
        if args.preset_union_mask:
            enabled_names = [k for k, on in preset_enabled.items() if on]
            enabled_text = ",".join(enabled_names) if enabled_names else "none"
            fig.suptitle(
                (
                    "Preset mask: [red(H 242..262 wrap AND V>=128)] OR [yellow(H 8..44)] OR [black(V<=64)]"
                    f" | enabled={enabled_text} | mask true={selected}, mask false={unselected}, true%={pct:.2f}"
                ),
                fontsize=11,
            )
        else:
            fig.suptitle(
                (
                    f"H={h_center}±w{h_width} (wrap), S={s_center}±w{s_width}, V={v_center}±w{v_width}"
                    f" | mask true={selected}, mask false={unselected}, true%={pct:.2f}"
                ),
                fontsize=11,
            )
        fig.canvas.draw_idle()

    def on_slider_change(_val: float) -> None:
        update_views()

    slider_h.on_changed(on_slider_change)
    slider_hw.on_changed(on_slider_change)
    slider_s.on_changed(on_slider_change)
    slider_sw.on_changed(on_slider_change)
    slider_v.on_changed(on_slider_change)
    slider_vw.on_changed(on_slider_change)

    def on_preset_toggle(label: str) -> None:
        if label in preset_enabled:
            preset_enabled[label] = not preset_enabled[label]
            update_views()

    if preset_checks is not None:
        preset_checks.on_clicked(on_preset_toggle)

    def on_click(event) -> None:
        if event.inaxes != ax_orig or getattr(event, "button", None) != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(np.clip(np.round(event.xdata), 0, rgb.shape[1] - 1))
        y = int(np.clip(np.round(event.ydata), 0, rgb.shape[0] - 1))
        h0 = int(h_ch[y, x])
        s0 = int(s_ch[y, x])
        v0 = int(v_ch[y, x])

        marker.set_offsets(np.array([[x, y]], dtype=np.float32))
        marker.set_visible(True)

        slider_h.set_val(h0)
        slider_s.set_val(s0)
        slider_v.set_val(v0)

    fig.canvas.mpl_connect("button_press_event", on_click)

    update_views()
    plt.show()


if __name__ == "__main__":
    main()

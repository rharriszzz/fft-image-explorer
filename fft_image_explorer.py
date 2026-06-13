#!/usr/bin/env python3
"""
fft_image_explorer.py

Interactive 2D FFT explorer for color images.

Features:
- Load an image.
- Choose a scalar channel derived from RGB:
    R, G, B, luminance Y, HSV H/S/V,
    opponent channels R-G, B-Y, R+G-2B,
    simple PCA-like decorrelation channels.
- Choose a spatial mask/window:
    full image, rectangle, rotated rectangle, ellipse/circle.
- Choose soft edge:
    hard, cosine/tukey-like, gaussian.
- Adjust mask center, size, rotation, and edge softness.
- Display mask overlay and log-magnitude 2D FFT.

Dependencies:
    pip install numpy pillow matplotlib

Optional, for HEIC/HEIF images:
    pip install pillow-heif

Usage:
    python fft_image_explorer.py path/to/image.jpg
    python fft_image_explorer.py path/to/image.heic
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    # HEIC/HEIF support is optional.
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button
from matplotlib.patches import Rectangle, Ellipse


# -----------------------------
# Image loading and color spaces
# -----------------------------

def load_image_rgb(path: str, max_side: int | None = 1200) -> np.ndarray:
    """
    Load an image, respect EXIF orientation, convert to RGB, return float array in [0,1].

    max_side limits the displayed/processed size to keep the UI responsive.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    if max_side is not None:
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def rgb_to_hsv_np(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized RGB->HSV for rgb in [0,1].
    Returns H, S, V where H is normalized to [0,1].
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)

    mask = delta > 1e-12

    idx = mask & (cmax == r)
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0

    idx = mask & (cmax == g)
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0

    idx = mask & (cmax == b)
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0

    h = h / 6.0

    s = np.zeros_like(cmax)
    nz = cmax > 1e-12
    s[nz] = delta[nz] / cmax[nz]

    v = cmax
    return h, s, v


def pca_decorrelation_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA channels from RGB values in the current image.

    This is a simple image-specific color decorrelation. It is not a standard color space.
    The first component usually captures the largest brightness/color variation.
    """
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3).astype(np.float64)
    flat = flat - flat.mean(axis=0, keepdims=True)

    cov = np.cov(flat, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]

    pcs = flat @ vecs
    pcs = pcs.reshape(h, w, 3)

    # Normalize each component for display/FFT convenience.
    out = []
    for k in range(3):
        c = pcs[..., k]
        lo, hi = np.percentile(c, [1, 99])
        if hi > lo:
            c = (c - lo) / (hi - lo)
        else:
            c = np.zeros_like(c)
        out.append(np.clip(c, 0, 1).astype(np.float32))
    return out[0], out[1], out[2]


def robust_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize a scalar image to roughly [0,1] using percentiles."""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)


@dataclass
class ChannelBank:
    names: list[str]
    arrays: dict[str, np.ndarray]


def make_channel_bank(rgb: np.ndarray) -> ChannelBank:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    # Rec. 601 luma, common for JPEG-like Y.
    y = 0.299 * r + 0.587 * g + 0.114 * b

    h, s, v = rgb_to_hsv_np(rgb)

    # Simple opponent/decorrelation-style channels.
    # These are useful for colored bead/fabric/print patterns.
    rg = robust_normalize(r - g)
    by = robust_normalize(b - y)
    red_vs_cyan = robust_normalize(2 * r - g - b)
    green_vs_magenta = robust_normalize(2 * g - r - b)
    blue_vs_yellow = robust_normalize(2 * b - r - g)

    pc1, pc2, pc3 = pca_decorrelation_channels(rgb)

    arrays = {
        "Y luminance": y,
        "R": r,
        "G": g,
        "B": b,
        "HSV H": h,
        "HSV S": s,
        "HSV V": v,
        "R-G opponent": rg,
        "B-Y opponent": by,
        "2R-G-B": red_vs_cyan,
        "2G-R-B": green_vs_magenta,
        "2B-R-G": blue_vs_yellow,
        "PCA 1": pc1,
        "PCA 2": pc2,
        "PCA 3": pc3,
    }
    names = list(arrays.keys())
    return ChannelBank(names=names, arrays=arrays)


# -----------------------------
# Windows / masks
# -----------------------------

def smoothstep01(t: np.ndarray) -> np.ndarray:
    """Smooth 0..1 transition."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def make_mask(
    shape: tuple[int, int],
    mask_type: str,
    edge_type: str,
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_deg: float,
    softness: float,
) -> np.ndarray:
    """
    Return a 2D mask/window in [0,1].

    Coordinates:
        cx, cy are pixel coordinates.
        width, height are full dimensions in pixels.
        angle_deg rotates the local x/y axes.

    mask_type:
        "Full image", "Rectangle", "Rotated rectangle", "Ellipse/circle"

    edge_type:
        "Hard", "Cosine", "Gaussian"
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]

    if mask_type == "Full image":
        base = np.ones((h, w), dtype=np.float32)
        if edge_type == "Hard":
            return base
        # For full image with non-hard edge, use a separable taper at the image borders.
        local_x = (xx - (w - 1) / 2) / max(w / 2, 1)
        local_y = (yy - (h - 1) / 2) / max(h / 2, 1)
        return edge_falloff_rect(local_x, local_y, edge_type, softness).astype(np.float32)

    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)

    dx = xx - cx
    dy = yy - cy

    # Rotate image coordinates into local mask coordinates.
    # local x-axis is rotated by angle_deg in image coordinates.
    xloc = c * dx + s * dy
    yloc = -s * dx + c * dy

    half_w = max(width / 2.0, 1.0)
    half_h = max(height / 2.0, 1.0)

    if mask_type in ("Rectangle", "Rotated rectangle"):
        xn = xloc / half_w
        yn = yloc / half_h
        return edge_falloff_rect(xn, yn, edge_type, softness).astype(np.float32)

    if mask_type == "Ellipse/circle":
        # Ellipse equation: radius <= 1 inside.
        rn = np.sqrt((xloc / half_w) ** 2 + (yloc / half_h) ** 2)
        return edge_falloff_radial(rn, edge_type, softness).astype(np.float32)

    raise ValueError(f"Unknown mask type: {mask_type}")


def edge_falloff_rect(xn: np.ndarray, yn: np.ndarray, edge_type: str, softness: float) -> np.ndarray:
    """
    Rectangular window on normalized coords where |xn|<=1 and |yn|<=1 is nominally inside.

    softness is a fraction of half-size. Example: 0.2 means the outer 20% tapers.
    """
    ax = np.abs(xn)
    ay = np.abs(yn)

    if edge_type == "Hard":
        return ((ax <= 1) & (ay <= 1)).astype(np.float32)

    if edge_type == "Gaussian":
        # Gaussian centered at the window center. softness controls sigma relative to window size.
        # Make softness=1 broad, softness small tight.
        sigma = max(softness, 0.03)
        m = np.exp(-0.5 * ((xn / sigma) ** 2 + (yn / sigma) ** 2))
        # Truncate very far out for readability/performance.
        m[(ax > 1) | (ay > 1)] = 0.0
        return m.astype(np.float32)

    if edge_type == "Cosine":
        # Flat center, cosine fade near edge.
        soft = np.clip(softness, 0.001, 0.95)
        # Distance to nearest rectangle edge in normalized units.
        inside = (ax <= 1) & (ay <= 1)
        dist_to_edge = np.minimum(1 - ax, 1 - ay)
        # dist_to_edge >= soft => full value.
        t = dist_to_edge / soft
        m = smoothstep01(t)
        m[~inside] = 0.0
        return m.astype(np.float32)

    raise ValueError(f"Unknown edge type: {edge_type}")


def edge_falloff_radial(rn: np.ndarray, edge_type: str, softness: float) -> np.ndarray:
    """Radial/elliptical window where rn<=1 is nominally inside."""
    if edge_type == "Hard":
        return (rn <= 1).astype(np.float32)

    if edge_type == "Gaussian":
        sigma = max(softness, 0.03)
        m = np.exp(-0.5 * (rn / sigma) ** 2)
        m[rn > 1] = 0.0
        return m.astype(np.float32)

    if edge_type == "Cosine":
        soft = np.clip(softness, 0.001, 0.95)
        inside = rn <= 1
        dist_to_edge = 1 - rn
        t = dist_to_edge / soft
        m = smoothstep01(t)
        m[~inside] = 0.0
        return m.astype(np.float32)

    raise ValueError(f"Unknown edge type: {edge_type}")


# -----------------------------
# FFT display
# -----------------------------

def crop_to_mask_nonzero(xm: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Crop to the tight bounding box where mask is non-zero-ish."""
    idx = np.argwhere(mask > eps)
    if idx.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    y0, x0 = idx.min(axis=0)
    y1, x1 = idx.max(axis=0) + 1
    return xm[y0:y1, x0:x1].astype(np.float32)


def compute_fft_products(
    channel: np.ndarray,
    mask: np.ndarray,
    zero_pad: bool,
    fft_highpass_percent: float = 0.0,
    fft_lowpass_percent: float = 0.0,
    fft_threshold_percent: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Build all FFT-related displays.

    Returns:
        pre_fft_display: masked image cropped to non-zero mask support
        fft_logmag_display: shifted log magnitude of FFT
        ifft_display: inverse FFT (real part), cropped back to FFT input size
        low_pass_removed_percent: percent of original power removed by the low-pass mask
        high_pass_removed_percent: percent of original power removed by the high-pass mask
        remaining_percent: percent of original power remaining after all active filters
    """
    x = channel.astype(np.float32)
    m = mask.astype(np.float32)

    xm = x * m
    xm_crop = crop_to_mask_nonzero(xm, m)

    # The FFT is computed on the cropped non-zero region so the display focuses on useful content.
    fft_input = xm_crop
    in_h, in_w = fft_input.shape

    if zero_pad:
        padded = np.zeros((2 * in_h, 2 * in_w), dtype=np.float32)
        y0 = in_h // 2
        x0 = in_w // 2
        padded[y0:y0 + in_h, x0:x0 + in_w] = fft_input
        fft_input = padded

    f = np.fft.fftshift(np.fft.fft2(fft_input))
    base_power = float((np.abs(f) ** 2).sum())

    fh, fw = f.shape
    fcy, fcx = fh // 2, fw // 2
    yy, xx = np.ogrid[:fh, :fw]
    max_radius_px = math.hypot(fh / 2.0, fw / 2.0)

    hp_keep_mask = np.ones((fh, fw), dtype=bool)
    lp_keep_mask = np.ones((fh, fw), dtype=bool)

    # High-pass filter in frequency domain: zero bins near the FFT origin.
    if fft_highpass_percent > 0.0:
        hp_radius_px = fft_highpass_percent * 0.01 * max_radius_px
        low_freq_mask = (yy - fcy) ** 2 + (xx - fcx) ** 2 <= hp_radius_px ** 2
        hp_keep_mask = ~low_freq_mask
        f = f * hp_keep_mask

    # Low-pass filter in frequency domain: keep only bins near the FFT origin.
    if fft_lowpass_percent > 0.0:
        lp_radius_px = fft_lowpass_percent * 0.01 * max_radius_px
        lp_keep_mask = (yy - fcy) ** 2 + (xx - fcx) ** 2 <= lp_radius_px ** 2
        f = f * lp_keep_mask

    # Apply FFT thresholding if enabled (based on log-magnitude)
    if fft_threshold_percent > 0.0:
        mag = np.log1p(np.abs(f))
        mag_max = float(mag.max())
        threshold_val = mag_max * (fft_threshold_percent / 100.0)
        f_mask = mag >= threshold_val
        f = f * f_mask.astype(np.complex64)

    final_power = float((np.abs(f) ** 2).sum())
    hp_only_power = float((np.abs(np.fft.fftshift(np.fft.fft2(fft_input)) * hp_keep_mask) ** 2).sum()) if fft_highpass_percent > 0.0 else base_power
    lp_only_power = float((np.abs(np.fft.fftshift(np.fft.fft2(fft_input)) * lp_keep_mask) ** 2).sum()) if fft_lowpass_percent > 0.0 else base_power

    eps = 1e-12
    low_pass_removed_percent = 100.0 * (base_power - lp_only_power) / max(base_power, eps)
    high_pass_removed_percent = 100.0 * (base_power - hp_only_power) / max(base_power, eps)
    remaining_percent = 100.0 * final_power / max(base_power, eps)

    mag = np.log1p(np.abs(f))

    # Sparse-aware contrast for display: after strong filtering many bins are exactly 0,
    # so compute robust limits from non-zero bins when available.
    nz = mag[mag > 0]
    if nz.size >= 32:
        lo, hi = np.percentile(nz, [5, 99.5])
    else:
        lo, hi = np.percentile(mag, [1, 99.7])

    if hi > lo:
        mag = (mag - lo) / (hi - lo)
        mag = np.clip(mag, 0, 1)
        # Gentle gamma lift improves visibility of retained low-amplitude structure.
        mag = np.sqrt(mag)
    else:
        mag = np.zeros_like(mag)

    inv_full = np.real(np.fft.ifft2(np.fft.ifftshift(f))).astype(np.float32)
    if zero_pad:
        y0 = in_h // 2
        x0 = in_w // 2
        inv = inv_full[y0:y0 + in_h, x0:x0 + in_w]
    else:
        inv = inv_full

    pre_fft_display = robust_normalize(xm_crop)
    fft_logmag_display = np.clip(mag, 0, 1).astype(np.float32)
    ifft_display = robust_normalize(inv)
    return pre_fft_display, fft_logmag_display, ifft_display, low_pass_removed_percent, high_pass_removed_percent, remaining_percent


# -----------------------------
# Interactive UI
# -----------------------------

class FFTExplorer:
    def __init__(self, rgb: np.ndarray):
        self.rgb = rgb
        self.h, self.w, _ = rgb.shape
        self.channels = make_channel_bank(rgb)

        self.channel_name = "Y luminance"
        self.mask_type = "Ellipse/circle"
        self.edge_type = "Gaussian"
        self.zero_pad = False
        self.link_dimensions = True
        self.fft_highpass_percent = 0.0
        self.fft_lowpass_percent = 100.0
        self.fft_threshold_value = 0.0

        self.cx0 = (self.w - 1) / 2
        self.cy0 = (self.h - 1) / 2
        self.width0 = 100.0
        self.height0 = 100.0
        self._linking_sliders = False
        self.angle0 = 0.0
        self.softness0 = 0.20
        self.center_dot_scale0 = 1.5

        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("2D FFT Image Explorer")

        # Main axes
        self.ax_img = self.fig.add_axes([0.03, 0.28, 0.30, 0.67])
        self.ax_pre = self.fig.add_axes([0.35, 0.62, 0.28, 0.33])
        self.ax_fft = self.fig.add_axes([0.65, 0.62, 0.32, 0.33])
        self.ax_ifft = self.fig.add_axes([0.35, 0.35, 0.62, 0.23])

        # Controls
        self.ax_channel = self.fig.add_axes([0.02, 0.02, 0.16, 0.19])
        self.ax_mask = self.fig.add_axes([0.20, 0.02, 0.14, 0.19])
        self.ax_edge = self.fig.add_axes([0.36, 0.02, 0.10, 0.19])
        self.ax_checks = self.fig.add_axes([0.48, 0.06, 0.12, 0.12])
        self.ax_reset = self.fig.add_axes([0.48, 0.02, 0.12, 0.035])

        slider_left = 0.64
        slider_width = 0.31
        self.ax_cx = self.fig.add_axes([slider_left, 0.17, slider_width, 0.025])
        self.ax_cy = self.fig.add_axes([slider_left, 0.135, slider_width, 0.025])
        self.ax_width = self.fig.add_axes([slider_left, 0.10, slider_width, 0.025])
        self.ax_height = self.fig.add_axes([slider_left, 0.065, slider_width, 0.025])
        self.ax_angle = self.fig.add_axes([slider_left, 0.03, slider_width, 0.025])
        self.ax_soft = self.fig.add_axes([slider_left, 0.205, slider_width, 0.025])
        self.ax_fft_thresh = self.fig.add_axes([slider_left, 0.245, slider_width, 0.025])
        self.ax_fft_highpass = self.fig.add_axes([slider_left, 0.275, slider_width, 0.025])
        self.ax_fft_lowpass = self.fig.add_axes([slider_left, 0.305, slider_width, 0.025])

        self.im_img = self.ax_img.imshow(self.rgb)
        self.mask_overlay = self.ax_img.imshow(np.zeros((self.h, self.w)), alpha=0.35, cmap="magma", vmin=0, vmax=1)
        self.ax_img.set_title("Image with mask overlay")
        self.ax_img.set_axis_off()

        dummy_pre = np.zeros((self.h, self.w))
        self.im_pre = self.ax_pre.imshow(dummy_pre, cmap="gray", vmin=0, vmax=1)
        self.ax_pre.set_title("Masked image crop (input to FFT)")
        self.ax_pre.set_axis_off()
        self._pre_img_data = dummy_pre

        dummy_fft = np.zeros((self.h, self.w))
        self.im_fft = self.ax_fft.imshow(dummy_fft, cmap="gray", vmin=0, vmax=1)
        self.ax_fft.set_title("2D FFT log magnitude")
        self.ax_fft.set_axis_off()
        self._fft_img_data = dummy_fft
        self.txt_fft_metric = self.ax_fft.text(
            0.02,
            0.98,
            "",
            transform=self.ax_fft.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.55, boxstyle="round,pad=0.25"),
        )

        dummy_ifft = np.zeros((self.h, self.w))
        self.im_ifft = self.ax_ifft.imshow(dummy_ifft, cmap="gray", vmin=0, vmax=1)
        self.ax_ifft.set_title("Inverse FFT (real part)")
        self.ax_ifft.set_axis_off()
        self._ifft_img_data = dummy_ifft
        self._popup_views: list[dict] = []

        self.radio_channel = RadioButtons(self.ax_channel, self.channels.names, active=self.channels.names.index(self.channel_name))
        self.radio_mask = RadioButtons(self.ax_mask, ["Full image", "Rectangle", "Rotated rectangle", "Ellipse/circle"], active=3)
        self.radio_edge = RadioButtons(self.ax_edge, ["Hard", "Cosine", "Gaussian"], active=2)

        self.checks = CheckButtons(self.ax_checks, ["zero pad", "link w/h"], [self.zero_pad, self.link_dimensions])
        self.btn_reset = Button(self.ax_reset, "Reset")

        self.sl_cx = Slider(self.ax_cx, "center x", 0, self.w - 1, valinit=self.cx0, valstep=1)
        self.sl_cy = Slider(self.ax_cy, "center y", 0, self.h - 1, valinit=self.cy0, valstep=1)
        self.sl_width = Slider(self.ax_width, "width", 2, self.w, valinit=self.width0)
        self.sl_height = Slider(self.ax_height, "height", 2, self.h, valinit=self.height0)
        self.sl_angle = Slider(self.ax_angle, "angle", -180, 180, valinit=self.angle0)
        self.sl_soft = Slider(self.ax_soft, "softness", 0.02, 1.0, valinit=self.softness0)
        self.sl_fft_thresh = Slider(self.ax_fft_thresh, "FFT thresh %", 0.0, 100.0, valinit=self.fft_threshold_value)
        self.sl_fft_highpass = Slider(self.ax_fft_highpass, "FFT high-pass %", 0.0, 100.0, valinit=self.fft_highpass_percent)
        self.sl_fft_lowpass = Slider(self.ax_fft_lowpass, "FFT low-pass %", 0.0, 100.0, valinit=self.fft_lowpass_percent)

        self.radio_channel.on_clicked(self.on_channel)
        self.radio_mask.on_clicked(self.on_mask)
        self.radio_edge.on_clicked(self.on_edge)
        self.checks.on_clicked(self.on_check)
        self.btn_reset.on_clicked(self.on_reset)

        self.sl_width.on_changed(lambda _val: self._on_width_changed(_val))
        self.sl_height.on_changed(lambda _val: self._on_height_changed(_val))
        self.sl_fft_thresh.on_changed(lambda _val: self._on_fft_thresh_changed(_val))
        self.sl_fft_highpass.on_changed(lambda _val: self._on_fft_highpass_changed(_val))
        self.sl_fft_lowpass.on_changed(lambda _val: self._on_fft_lowpass_changed(_val))
        for sl in [self.sl_cx, self.sl_cy, self.sl_angle, self.sl_soft]:
            sl.on_changed(lambda _val: self.update())

        self._dragging = False
        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        self.update()

    def on_channel(self, label: str):
        self.channel_name = label
        self.update()

    def on_mask(self, label: str):
        self.mask_type = label
        self.update()

    def on_edge(self, label: str):
        self.edge_type = label
        self.update()

    def on_check(self, label: str):
        if label == "zero pad":
            self.zero_pad = not self.zero_pad
        elif label == "link w/h":
            self.link_dimensions = not self.link_dimensions
        self.update()

    def on_reset(self, _event):
        self.sl_cx.reset()
        self.sl_cy.reset()
        self.sl_width.reset()
        self.sl_height.reset()
        self.sl_angle.reset()
        self.sl_soft.reset()
        self.sl_fft_thresh.reset()
        self.sl_fft_highpass.reset()
        self.sl_fft_lowpass.reset()
        self.update()

    def on_mouse_press(self, event):
        if event.inaxes == self.ax_pre:
            self.open_popup_image("pre", "Masked image crop", cmap="gray", vmin=0, vmax=1)
            return
        if event.inaxes == self.ax_fft:
            self.open_popup_image("fft", "2D FFT log magnitude", cmap="gray", vmin=0, vmax=1)
            return
        if event.inaxes == self.ax_ifft:
            self.open_popup_image("ifft", "Inverse FFT (real part)", cmap="gray", vmin=0, vmax=1)
            return

        if event.inaxes == self.ax_img and event.xdata is not None and event.ydata is not None:
            self._dragging = True
            self.set_center(event.xdata, event.ydata)

    def on_mouse_release(self, event):
        self._dragging = False

    def on_mouse_move(self, event):
        if self._dragging and event.inaxes == self.ax_img and event.xdata is not None and event.ydata is not None:
            self.set_center(event.xdata, event.ydata)

    def set_center(self, x: float, y: float):
        self.sl_cx.set_val(np.clip(x, 0, self.w - 1))
        self.sl_cy.set_val(np.clip(y, 0, self.h - 1))

    def _on_width_changed(self, val: float):
        """Sync height to width if linking is enabled."""
        if self.link_dimensions and not self._linking_sliders:
            self._linking_sliders = True
            self.sl_height.set_val(val)
            self._linking_sliders = False
        self.update()

    def _on_height_changed(self, val: float):
        """Sync width to height if linking is enabled."""
        if self.link_dimensions and not self._linking_sliders:
            self._linking_sliders = True
            self.sl_width.set_val(val)
            self._linking_sliders = False
        self.update()

    def _on_fft_thresh_changed(self, val: float):
        """Update FFT threshold and refresh."""
        self.fft_threshold_value = float(val)
        self.update()

    def _on_fft_highpass_changed(self, val: float):
        """Update FFT high-pass radius and refresh."""
        self.fft_highpass_percent = float(val)
        self.update()

    def _on_fft_lowpass_changed(self, val: float):
        """Update FFT low-pass radius and refresh."""
        self.fft_lowpass_percent = float(val)
        self.update()

    def open_popup_image(
        self,
        source_key: str,
        title: str,
        cmap: str = "gray",
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> None:
        """Open a new figure window with the provided image for zoom/pan inspection."""
        image = self._get_source_image(source_key)
        extent = self._get_source_extent(source_key, image)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        if source_key == "fft":
            ax.set_xlabel("fx")
            ax.set_ylabel("fy")

        record = {
            "fig": fig,
            "ax": ax,
            "im": im,
            "source_key": source_key,
            "title": title,
        }
        self._popup_views.append(record)

        def on_close(_event):
            self._popup_views = [p for p in self._popup_views if p.get("fig") is not fig]

        fig.canvas.mpl_connect("close_event", on_close)

        fig.tight_layout()
        fig.canvas.manager.set_window_title(f"Detail: {title}")
        fig.show()

    def _get_source_image(self, source_key: str) -> np.ndarray:
        if source_key == "pre":
            return self._pre_img_data
        if source_key == "fft":
            return self._fft_img_data
        if source_key == "ifft":
            return self._ifft_img_data
        raise ValueError(f"Unknown popup source key: {source_key}")

    def _get_source_extent(self, source_key: str, image: np.ndarray) -> list[float]:
        h, w = image.shape[:2]
        if source_key == "fft":
            # Frequency-centered axes: (0,0) at image center.
            return [-w / 2.0, w / 2.0, h / 2.0, -h / 2.0]
        return [0, w, h, 0]

    def _refresh_open_popups(self) -> None:
        active: list[dict] = []
        for popup in self._popup_views:
            fig = popup["fig"]
            if not plt.fignum_exists(fig.number):
                continue

            img = self._get_source_image(popup["source_key"])
            im = popup["im"]
            ax = popup["ax"]
            extent = self._get_source_extent(popup["source_key"], img)
            im.set_data(img)
            im.set_extent(extent)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            fig.canvas.draw_idle()
            active.append(popup)

        self._popup_views = active

    def current_mask(self) -> np.ndarray:
        return make_mask(
            shape=(self.h, self.w),
            mask_type=self.mask_type,
            edge_type=self.edge_type,
            cx=float(self.sl_cx.val),
            cy=float(self.sl_cy.val),
            width=float(self.sl_width.val),
            height=float(self.sl_height.val),
            angle_deg=float(self.sl_angle.val),
            softness=float(self.sl_soft.val),
        )

    def update(self):
        channel = self.channels.arrays[self.channel_name]
        mask = self.current_mask()

        # Show scalar channel as background, not RGB, when useful?
        # Keep RGB image visible and overlay mask; title says selected channel.
        self.mask_overlay.set_data(mask)
        self.mask_overlay.set_alpha(0.35)

        pre_img, fft_img, ifft_img, low_pass_removed_percent, high_pass_removed_percent, remaining_percent = compute_fft_products(
            channel=channel,
            mask=mask,
            zero_pad=self.zero_pad,
            fft_highpass_percent=self.fft_highpass_percent,
            fft_lowpass_percent=self.fft_lowpass_percent,
            fft_threshold_percent=self.fft_threshold_value,
        )

        self.im_pre.set_data(pre_img)
        self.im_pre.set_extent([0, pre_img.shape[1], pre_img.shape[0], 0])
        self.ax_pre.set_xlim(0, pre_img.shape[1])
        self.ax_pre.set_ylim(pre_img.shape[0], 0)
        self._pre_img_data = pre_img

        self.im_fft.set_data(fft_img)
        self.im_fft.set_extent([0, fft_img.shape[1], fft_img.shape[0], 0])
        self.ax_fft.set_xlim(0, fft_img.shape[1])
        self.ax_fft.set_ylim(fft_img.shape[0], 0)
        self._fft_img_data = fft_img

        self.im_ifft.set_data(ifft_img)
        self.im_ifft.set_extent([0, ifft_img.shape[1], ifft_img.shape[0], 0])
        self.ax_ifft.set_xlim(0, ifft_img.shape[1])
        self.ax_ifft.set_ylim(ifft_img.shape[0], 0)
        self._ifft_img_data = ifft_img

        self.ax_img.set_title(f"Image with mask overlay | channel: {self.channel_name}")
        self.ax_pre.set_title("Masked image crop (zeros outside mask removed)")
        self.ax_fft.set_title("2D FFT log magnitude; center = low frequency")
        self.txt_fft_metric.set_text(
            f"low-pass removed: {low_pass_removed_percent:.2f}%\n"
            f"high-pass removed: {high_pass_removed_percent:.2f}%\n"
            f"remaining power: {remaining_percent:.2f}%"
        )
        self.ax_ifft.set_title("Inverse FFT of spectrum (real part)")

        self._refresh_open_popups()
        self.fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 2D FFT explorer for color images.")
    parser.add_argument("image", help="Input image path, e.g. JPG, PNG, TIFF; HEIC if pillow-heif is installed.")
    parser.add_argument("--max-side", type=int, default=1200, help="Resize largest dimension for speed. Use 0 for no resize.")
    args = parser.parse_args()

    max_side = None if args.max_side == 0 else args.max_side
    rgb = load_image_rgb(args.image, max_side=max_side)

    explorer = FFTExplorer(rgb)
    plt.show()


if __name__ == "__main__":
    main()

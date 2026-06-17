#!/usr/bin/env python3
"""
fft_image_explorer.py

Interactive 2D FFT explorer for color images.

Features:
- Load an image.
- Choose a scalar channel derived from RGB:
    R, G, B, luminance Y, HSV H/S/V,
    and CMYK C/M/Y/K.
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
    # import pillow_heif
    # pillow_heif.register_heif_opener()
    pass
except Exception:
    # HEIC/HEIF support is optional.
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button, TextBox
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

    # CMYK conversion from normalized RGB.
    # K is black key; C/M/Y are normalized by (1-K) when possible.
    k = 1.0 - np.maximum(np.maximum(r, g), b)
    denom = 1.0 - k
    c = np.zeros_like(k)
    m = np.zeros_like(k)
    y_cmyk = np.zeros_like(k)
    mask = denom > 1e-12
    c[mask] = (1.0 - r[mask] - k[mask]) / denom[mask]
    m[mask] = (1.0 - g[mask] - k[mask]) / denom[mask]
    y_cmyk[mask] = (1.0 - b[mask] - k[mask]) / denom[mask]
    c = np.clip(c, 0.0, 1.0)
    m = np.clip(m, 0.0, 1.0)
    y_cmyk = np.clip(y_cmyk, 0.0, 1.0)
    k = np.clip(k, 0.0, 1.0)

    arrays = {
        "Y luminance": y,
        "R": r,
        "G": g,
        "B": b,
        "HSV H": h,
        "HSV S": s,
        "HSV V": v,
        "CMYK C": c,
        "CMYK M": m,
        "CMYK Y": y_cmyk,
        "CMYK K": k,
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


def pad_with_tiled_edge_blocks(
    x: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    block_size: int = 20,
) -> np.ndarray:
    """Pad by repeating edge-adjacent blocks, not single edge pixels."""
    h, w = x.shape
    out_h = h + pad_top + pad_bottom
    out_w = w + pad_left + pad_right
    out = np.zeros((out_h, out_w), dtype=x.dtype)

    out[pad_top:pad_top + h, pad_left:pad_left + w] = x

    bh = max(1, min(block_size, h))
    bw = max(1, min(block_size, w))

    def pingpong_index(k: int, n: int) -> int:
        if n <= 1:
            return 0
        period = 2 * n
        t = k % period
        return t if t < n else (2 * n - 1 - t)

    # Build inward-oriented edge bands: index 0 is the nearest source sample at the edge.
    top_band = x[:bh, :]
    bottom_band = x[h - bh:, :][::-1, :]

    for i in range(pad_top):
        out[pad_top - 1 - i, pad_left:pad_left + w] = top_band[pingpong_index(i, bh), :]

    for i in range(pad_bottom):
        out[pad_top + h + i, pad_left:pad_left + w] = bottom_band[pingpong_index(i, bh), :]

    center_cols = out[:, pad_left:pad_left + w]
    left_band = center_cols[:, :bw]
    right_band = center_cols[:, w - bw:][:, ::-1]

    for j in range(pad_left):
        out[:, pad_left - 1 - j] = left_band[:, pingpong_index(j, bw)]

    for j in range(pad_right):
        out[:, pad_left + w + j] = right_band[:, pingpong_index(j, bw)]

    return out


def find_top_fft_peaks(logmag: np.ndarray, top_k: int = 7, min_distance: int = 6) -> list[dict[str, float]]:
    """Find the strongest local maxima in a 2D log-magnitude FFT image, merging conjugate pairs."""
    h, w = logmag.shape
    cy, cx = h // 2, w // 2

    def canonical_offset(yy: int, xx: int) -> tuple[int, int]:
        dx = int(xx - cx)
        dy = int(cy - yy)
        if dy < 0 or (dy == 0 and dx < 0):
            dx = -dx
            dy = -dy
        return dx, dy

    def peak_region_stats(yy: int, xx: int) -> dict[str, float]:
        peak_val = float(logmag[yy, xx])
        radius = 8
        y0 = max(0, yy - radius)
        y1 = min(h, yy + radius + 1)
        x0 = max(0, xx - radius)
        x1 = min(w, xx + radius + 1)
        patch = logmag[y0:y1, x0:x1]
        py, px = np.mgrid[y0:y1, x0:x1]

        support = patch >= max(0.5 * peak_val, 1e-9)
        if not np.any(support):
            support = patch > 0

        weights = patch[support]
        dx_support = px[support] - xx
        dy_support = yy - py[support]
        dist2 = dx_support ** 2 + dy_support ** 2
        broadness = float(np.sqrt((weights * dist2).sum() / max(weights.sum(), 1e-12)))
        log_power_sum = float(weights.sum())
        radius_support = np.sqrt(dist2)

        # Track the flat-top region (pixels equal to the local max) to compare
        # against what the detail popup shows as a uniform bright disk.
        plateau = patch >= (peak_val - 1e-12)
        if np.any(plateau):
            dx_plateau = px[plateau] - xx
            dy_plateau = yy - py[plateau]
            dist2_plateau = dx_plateau ** 2 + dy_plateau ** 2
            radius_plateau = np.sqrt(dist2_plateau)
            plateau_count = int(plateau.sum())
            plateau_rmax = float(radius_plateau.max())
            plateau_dx_min = float(dx_plateau.min())
            plateau_dx_max = float(dx_plateau.max())
            plateau_dy_min = float(dy_plateau.min())
            plateau_dy_max = float(dy_plateau.max())
        else:
            plateau_count = 0
            plateau_rmax = 0.0
            plateau_dx_min = 0.0
            plateau_dx_max = 0.0
            plateau_dy_min = 0.0
            plateau_dy_max = 0.0

        threshold = float(max(0.5 * peak_val, 1e-9))
        return {
            "peak_val": peak_val,
            "support_threshold": threshold,
            "log_power_sum": log_power_sum,
            "broadness": broadness,
            "support_count": float(support.sum()),
            "support_rmax": float(radius_support.max()) if radius_support.size else 0.0,
            "support_rms_unweighted": float(np.sqrt(dist2.mean())) if dist2.size else 0.0,
            "support_dx_min": float(dx_support.min()) if dx_support.size else 0.0,
            "support_dx_max": float(dx_support.max()) if dx_support.size else 0.0,
            "support_dy_min": float(dy_support.min()) if dy_support.size else 0.0,
            "support_dy_max": float(dy_support.max()) if dy_support.size else 0.0,
            "plateau_count": float(plateau_count),
            "plateau_rmax": plateau_rmax,
            "plateau_dx_min": plateau_dx_min,
            "plateau_dx_max": plateau_dx_max,
            "plateau_dy_min": plateau_dy_min,
            "plateau_dy_max": plateau_dy_max,
        }

    # Simple 3x3 local-maximum test without extra dependencies.
    padded = np.pad(logmag, 1, mode="constant", constant_values=-np.inf)
    local_max = np.ones((h, w), dtype=bool)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            if oy == 0 and ox == 0:
                continue
            neigh = padded[1 + oy:1 + oy + h, 1 + ox:1 + ox + w]
            local_max &= logmag >= neigh

    candidates = np.argwhere(local_max & (logmag > 0))
    if candidates.size == 0:
        candidates = np.array([[cy, cx]])

    values = logmag[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(values)[::-1]
    candidates = candidates[order]

    selected: list[tuple[int, int]] = [(cy, cx)]
    selected_keys: set[tuple[int, int]] = {canonical_offset(cy, cx)}

    for yy, xx in candidates:
        if len(selected) >= top_k:
            break
        yy_i = int(yy)
        xx_i = int(xx)
        key = canonical_offset(yy_i, xx_i)
        if key in selected_keys:
            continue
        if any((yy_i - sy) ** 2 + (xx_i - sx) ** 2 < min_distance ** 2 for sy, sx in selected):
            continue
        selected.append((yy_i, xx_i))
        selected_keys.add(key)

    if len(selected) < top_k:
        for yy, xx in candidates:
            if len(selected) >= top_k:
                break
            yy_i = int(yy)
            xx_i = int(xx)
            key = canonical_offset(yy_i, xx_i)
            if key in selected_keys:
                continue
            selected.append((yy_i, xx_i))
            selected_keys.add(key)

    peaks: list[dict[str, float]] = []
    for yy, xx in selected[:top_k]:
        stats = peak_region_stats(yy, xx)
        peak_val = float(stats["peak_val"])
        log_power_sum = float(stats["log_power_sum"])
        broadness = float(stats["broadness"])
        dx = float(xx - cx)
        dy = float(cy - yy)
        angle_deg = float(np.degrees(np.arctan2(dy, dx))) if dx != 0 or dy != 0 else 0.0
        distance = float(np.hypot(dx, dy))

        mirror_y = int(2 * cy - yy)
        mirror_x = int(2 * cx - xx)
        if 0 <= mirror_y < h and 0 <= mirror_x < w and not (mirror_y == yy and mirror_x == xx):
            mirror_stats = peak_region_stats(mirror_y, mirror_x)
            log_power_sum += float(mirror_stats["log_power_sum"])
            broadness = 0.5 * (broadness + float(mirror_stats["broadness"]))

        peaks.append(
            {
                "x": dx,
                "y": dy,
                "distance": distance,
                "angle_deg": angle_deg,
                "log_power_sum": log_power_sum,
                "broadness": broadness,
                "peak_val": peak_val,
                "support_threshold": float(stats["support_threshold"]),
                "support_count": float(stats["support_count"]),
                "support_rmax": float(stats["support_rmax"]),
                "support_rms_unweighted": float(stats["support_rms_unweighted"]),
                "support_dx_min": float(stats["support_dx_min"]),
                "support_dx_max": float(stats["support_dx_max"]),
                "support_dy_min": float(stats["support_dy_min"]),
                "support_dy_max": float(stats["support_dy_max"]),
                "plateau_count": float(stats["plateau_count"]),
                "plateau_rmax": float(stats["plateau_rmax"]),
                "plateau_dx_min": float(stats["plateau_dx_min"]),
                "plateau_dx_max": float(stats["plateau_dx_max"]),
                "plateau_dy_min": float(stats["plateau_dy_min"]),
                "plateau_dy_max": float(stats["plateau_dy_max"]),
            }
        )

    return peaks


def compute_fft_products(
    channel: np.ndarray,
    mask: np.ndarray,
    zero_pad: bool,
    square_fft: bool = True,
    square_pad_mode: str = "edge_block20",
    fft_highpass_percent: float = 0.0,
    fft_lowpass_percent: float = 0.0,
    fft_threshold_percent: float = 0.0,
    debug_peak_stats: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int | bool],
    float,
    float,
    float,
    list[dict[str, float]],
]:
    """
    Build all FFT-related displays.

    Returns:
        pre_fft_display: masked image cropped to non-zero mask support
        fft_logmag_display: shifted log magnitude of FFT after display normalization
        fft_logmag_raw: shifted log magnitude of FFT before display normalization
        radial_r: radial distance bins from FFT center (pixels)
        radial_log_power: log-compressed total power in each radial bin
        ifft_display: inverse FFT (real part), cropped back to FFT input size
        fft_complex_shifted: shifted complex FFT after all active filters
        fft_recon_meta: metadata needed to crop inverse FFT back to analysis ROI
        low_pass_removed_percent: percent of original power removed by the low-pass mask
        high_pass_removed_percent: percent of original power removed by the high-pass mask
        remaining_percent: percent of original power remaining after all active filters
        peaks: strongest local peaks in the current FFT log-magnitude image
    """
    x = channel.astype(np.float32)
    m = mask.astype(np.float32)

    xm = x * m
    xm_crop = crop_to_mask_nonzero(xm, m)

    # The FFT is computed on the cropped non-zero region so the display focuses on useful content.
    fft_input = xm_crop
    src_h, src_w = fft_input.shape
    pad_top = 0
    pad_left = 0
    if square_fft and src_h != src_w:
        side = max(src_h, src_w)
        pad_h = side - src_h
        pad_w = side - src_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if square_pad_mode not in {"edge", "reflect", "edge_block20"}:
            raise ValueError(f"Unsupported square pad mode: {square_pad_mode}")
        if square_pad_mode == "edge_block20":
            fft_input = pad_with_tiled_edge_blocks(
                fft_input,
                pad_top=pad_top,
                pad_bottom=pad_bottom,
                pad_left=pad_left,
                pad_right=pad_right,
                block_size=20,
            )
        else:
            # Propagate boundary values outward to avoid zero-padding discontinuities.
            fft_input = np.pad(
                fft_input,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode=square_pad_mode,
            )

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

    mag_raw = np.log1p(np.abs(f))
    peaks = find_top_fft_peaks(mag_raw, top_k=7)
    if debug_peak_stats and peaks:
        lines = ["\n[FFT peak debug] top-peak support metrics from raw log-magnitude"]
        for idx, peak in enumerate(peaks, start=1):
            lines.append(
                (
                    f"  peak#{idx} offset=({peak['x']:.0f},{peak['y']:.0f}) "
                    f"peak={peak['peak_val']:.6f} thr={peak['support_threshold']:.6f} "
                    f"broad={peak['broadness']:.4f}"
                )
            )
            lines.append(
                (
                    f"    support: count={peak['support_count']:.0f} "
                    f"rmax={peak['support_rmax']:.4f} "
                    f"rms_unw={peak['support_rms_unweighted']:.4f} "
                    f"dx=[{peak['support_dx_min']:.2f},{peak['support_dx_max']:.2f}] "
                    f"dy=[{peak['support_dy_min']:.2f},{peak['support_dy_max']:.2f}]"
                )
            )
            lines.append(
                (
                    f"    plateau(==peak): count={peak['plateau_count']:.0f} "
                    f"rmax={peak['plateau_rmax']:.4f} "
                    f"dx=[{peak['plateau_dx_min']:.2f},{peak['plateau_dx_max']:.2f}] "
                    f"dy=[{peak['plateau_dy_min']:.2f},{peak['plateau_dy_max']:.2f}]"
                )
            )
        print("\n".join(lines), flush=True)
    mag = mag_raw

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

    # Remove any square-padding frame so inverse display maps to original FFT ROI.
    if square_fft and (src_h != in_h or src_w != in_w):
        inv = inv[pad_top:pad_top + src_h, pad_left:pad_left + src_w]

    # Show the actual array sent to FFT so square/edge padding is visible in UI.
    pre_fft_display = robust_normalize(fft_input)
    fft_logmag_display = np.clip(mag, 0, 1).astype(np.float32)
    fft_logmag_raw = mag_raw.astype(np.float32)

    # Direction-independent spectrum summary: total FFT power by radius.
    yy_i, xx_i = np.indices(f.shape)
    rr = np.sqrt((yy_i - fcy) ** 2 + (xx_i - fcx) ** 2)
    rbin = np.floor(rr).astype(np.int32)
    power2d = np.abs(f) ** 2
    radial_power = np.bincount(rbin.ravel(), weights=power2d.ravel())
    radial_r = np.arange(radial_power.size, dtype=np.float32)
    radial_log_power = np.log1p(radial_power).astype(np.float32)

    ifft_display = robust_normalize(inv)
    fft_recon_meta: dict[str, int | bool] = {
        "src_h": int(src_h),
        "src_w": int(src_w),
        "in_h": int(in_h),
        "in_w": int(in_w),
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
        "zero_pad": bool(zero_pad),
        "square_fft": bool(square_fft),
    }
    return (
        pre_fft_display,
        fft_logmag_display,
        fft_logmag_raw,
        radial_r,
        radial_log_power,
        ifft_display,
        f.astype(np.complex64),
        fft_recon_meta,
        low_pass_removed_percent,
        high_pass_removed_percent,
        remaining_percent,
        peaks,
    )


# -----------------------------
# Interactive UI
# -----------------------------

class FFTExplorer:
    def __init__(
        self,
        rgb_original: np.ndarray,
        rgb_display: np.ndarray | None = None,
        debug_peak_stats: bool = False,
        figure_size: tuple[float, float] = (14.0, 9.5),
    ):
        self.rgb_original = rgb_original
        self.h, self.w, _ = rgb_original.shape
        self.channels = make_channel_bank(rgb_original)

        self.rgb_display = rgb_display if rgb_display is not None else rgb_original
        self.disp_h, self.disp_w, _ = self.rgb_display.shape

        self.channel_name = "Y luminance"
        self.mask_type = "Ellipse/circle"
        self.edge_type = "Gaussian"
        self.zero_pad = False
        self.square_fft = True
        self.square_pad_mode = "edge_block20"
        self.link_dimensions = True
        self.fft_highpass_percent = 0.0
        self.fft_lowpass_percent = 100.0
        self.fft_threshold_value = 0.0
        self.debug_peak_stats = debug_peak_stats

        self.cx0 = (self.w - 1) / 2
        self.cy0 = (self.h - 1) / 2
        # Defaults are specified directly in original-image pixels.
        self.width0 = 200.0
        self.height0 = 200.0
        self.width0 = float(np.clip(self.width0, 2.0, float(self.w)))
        self.height0 = float(np.clip(self.height0, 2.0, float(self.h)))
        self._linking_sliders = False
        self.angle0 = 0.0
        self.softness0 = 0.20
        self.center_dot_scale0 = 1.5

        self.fig = plt.figure(figsize=figure_size)
        self.fig.canvas.manager.set_window_title("2D FFT Image Explorer")

        # Main axes
        self.ax_img = self.fig.add_axes([0.03, 0.28, 0.30, 0.67])
        self.ax_pre = self.fig.add_axes([0.35, 0.62, 0.28, 0.33])
        self.ax_fft = self.fig.add_axes([0.65, 0.62, 0.32, 0.33])
        self.ax_ifft = self.fig.add_axes([0.35, 0.39, 0.20, 0.19])
        self.ax_peaks = self.fig.add_axes([0.58, 0.39, 0.16, 0.19])
        self.ax_radial = self.fig.add_axes([0.85, 0.39, 0.12, 0.19])

        # Controls
        self.ax_channel = self.fig.add_axes([0.02, 0.02, 0.16, 0.19])
        self.ax_mask = self.fig.add_axes([0.20, 0.02, 0.14, 0.19])
        self.ax_edge = self.fig.add_axes([0.36, 0.02, 0.10, 0.19])
        self.ax_checks = self.fig.add_axes([0.48, 0.06, 0.12, 0.12])
        self.ax_reset = self.fig.add_axes([0.48, 0.02, 0.12, 0.035])

        slider_left = 0.67
        slider_width = 0.21
        input_gap = 0.01
        input_width = 0.06
        self.ax_cx = self.fig.add_axes([slider_left, 0.17, slider_width, 0.025])
        self.ax_cy = self.fig.add_axes([slider_left, 0.135, slider_width, 0.025])
        self.ax_width = self.fig.add_axes([slider_left, 0.10, slider_width, 0.025])
        self.ax_height = self.fig.add_axes([slider_left, 0.065, slider_width, 0.025])
        self.ax_angle = self.fig.add_axes([slider_left, 0.03, slider_width, 0.025])
        self.ax_soft = self.fig.add_axes([slider_left, 0.205, slider_width, 0.025])
        self.ax_fft_thresh = self.fig.add_axes([slider_left, 0.245, slider_width, 0.025])
        self.ax_fft_highpass = self.fig.add_axes([slider_left, 0.275, slider_width, 0.025])
        self.ax_fft_lowpass = self.fig.add_axes([slider_left, 0.305, slider_width, 0.025])
        input_left = slider_left + slider_width + input_gap
        self.ax_cx_in = self.fig.add_axes([input_left, 0.17, input_width, 0.025])
        self.ax_cy_in = self.fig.add_axes([input_left, 0.135, input_width, 0.025])
        self.ax_width_in = self.fig.add_axes([input_left, 0.10, input_width, 0.025])
        self.ax_height_in = self.fig.add_axes([input_left, 0.065, input_width, 0.025])
        self.ax_angle_in = self.fig.add_axes([input_left, 0.03, input_width, 0.025])
        self.ax_soft_in = self.fig.add_axes([input_left, 0.205, input_width, 0.025])
        self.ax_fft_thresh_in = self.fig.add_axes([input_left, 0.245, input_width, 0.025])
        self.ax_fft_highpass_in = self.fig.add_axes([input_left, 0.275, input_width, 0.025])
        self.ax_fft_lowpass_in = self.fig.add_axes([input_left, 0.305, input_width, 0.025])

        self.im_img = self.ax_img.imshow(self.rgb_display)
        self.mask_overlay = self.ax_img.imshow(np.zeros((self.disp_h, self.disp_w)), alpha=0.35, cmap="magma", vmin=0, vmax=1)
        self.ax_img.set_title("Image with mask overlay")
        self.ax_img.set_axis_off()
        self.ax_img.format_coord = self._format_img_hover_text
        self._img_overlay_data = self.rgb_display.copy()

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
        self._fft_raw_logmag_data = dummy_fft
        self._fft_complex_data = np.zeros((self.h, self.w), dtype=np.complex64)
        self._fft_recon_meta: dict[str, int | bool] = {
            "src_h": int(self.h),
            "src_w": int(self.w),
            "in_h": int(self.h),
            "in_w": int(self.w),
            "pad_top": 0,
            "pad_left": 0,
            "zero_pad": False,
            "square_fft": False,
        }
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
        self.txt_fft_hover = self.ax_fft.text(
            0.02,
            0.02,
            "",
            transform=self.ax_fft.transAxes,
            ha="left",
            va="bottom",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.45, boxstyle="round,pad=0.2"),
        )

        dummy_ifft = np.zeros((self.h, self.w))
        self.im_ifft = self.ax_ifft.imshow(dummy_ifft, cmap="gray", vmin=0, vmax=1)
        self.ax_ifft.set_title("Inverse FFT (real part)")
        self.ax_ifft.set_axis_off()
        self._ifft_img_data = dummy_ifft
        self._peaks_text_data = ""
        self._peaks_data: list[dict[str, float]] = []
        self._radial_r_data = np.zeros(1, dtype=np.float32)
        self._radial_log_power_data = np.zeros(1, dtype=np.float32)
        self.ax_peaks.set_title("Top FFT peaks")
        self.ax_peaks.set_axis_off()
        self.txt_peaks = self.ax_peaks.text(
            0.0,
            1.0,
            "",
            transform=self.ax_peaks.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            family="monospace",
        )

        self.radial_line, = self.ax_radial.plot([0.0], [0.0], color="tab:orange", linewidth=1.5)
        self.ax_radial.set_title("Radial FFT power")
        self.ax_radial.set_xlabel("radius (px)")
        self.ax_radial.set_ylabel("log1p(sum |F|^2)")
        self.ax_radial.yaxis.set_label_position("left")
        self.ax_radial.yaxis.tick_left()
        self.ax_radial.tick_params(axis="both", labelsize=8)
        self.ax_radial.yaxis.labelpad = 4
        self.ax_radial.grid(True, alpha=0.2)
        self._popup_views: list[dict] = []

        self.radio_channel = RadioButtons(self.ax_channel, self.channels.names, active=self.channels.names.index(self.channel_name))
        self.radio_mask = RadioButtons(self.ax_mask, ["Full image", "Rectangle", "Rotated rectangle", "Ellipse/circle"], active=3)
        self.radio_edge = RadioButtons(self.ax_edge, ["Hard", "Cosine", "Gaussian"], active=2)

        self.checks = CheckButtons(self.ax_checks, ["zero pad", "link w/h"], [self.zero_pad, self.link_dimensions])
        self.btn_reset = Button(self.ax_reset, "Reset")

        self.sl_cx = Slider(self.ax_cx, "center x", 0, self.w - 1, valinit=self.cx0, valstep=1, valfmt="%0.0f")
        self.sl_cy = Slider(self.ax_cy, "center y", 0, self.h - 1, valinit=self.cy0, valstep=1, valfmt="%0.0f")
        self.sl_width = Slider(self.ax_width, "width", 2, self.w, valinit=self.width0)
        self.sl_height = Slider(self.ax_height, "height", 2, self.h, valinit=self.height0)
        self.sl_angle = Slider(self.ax_angle, "angle", -180, 180, valinit=self.angle0)
        self.sl_soft = Slider(self.ax_soft, "softness", 0.02, 1.0, valinit=self.softness0)
        self.sl_fft_thresh = Slider(self.ax_fft_thresh, "FFT thresh %", 0.0, 100.0, valinit=self.fft_threshold_value)
        self.sl_fft_highpass = Slider(self.ax_fft_highpass, "FFT high-pass %", 0.0, 100.0, valinit=self.fft_highpass_percent)
        self.sl_fft_lowpass = Slider(self.ax_fft_lowpass, "FFT low-pass %", 0.0, 100.0, valinit=self.fft_lowpass_percent)
        self.tb_cx = TextBox(self.ax_cx_in, "", initial=f"{int(round(self.cx0))}")
        self.tb_cy = TextBox(self.ax_cy_in, "", initial=f"{int(round(self.cy0))}")
        self.tb_width = TextBox(self.ax_width_in, "", initial=f"{self.width0:.1f}")
        self.tb_height = TextBox(self.ax_height_in, "", initial=f"{self.height0:.1f}")
        self.tb_angle = TextBox(self.ax_angle_in, "", initial=f"{self.angle0:.1f}")
        self.tb_soft = TextBox(self.ax_soft_in, "", initial=f"{self.softness0:.3f}")
        self.tb_fft_thresh = TextBox(self.ax_fft_thresh_in, "", initial=f"{self.fft_threshold_value:.1f}")
        self.tb_fft_highpass = TextBox(self.ax_fft_highpass_in, "", initial=f"{self.fft_highpass_percent:.1f}")
        self.tb_fft_lowpass = TextBox(self.ax_fft_lowpass_in, "", initial=f"{self.fft_lowpass_percent:.1f}")
        self._syncing_inputs = False

        # Hide right-side slider value text; typed entry is shown in the input boxes.
        for sl in [
            self.sl_cx,
            self.sl_cy,
            self.sl_width,
            self.sl_height,
            self.sl_angle,
            self.sl_soft,
            self.sl_fft_thresh,
            self.sl_fft_highpass,
            self.sl_fft_lowpass,
        ]:
            sl.valtext.set_visible(False)

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
        self.sl_cx.on_changed(lambda _val: self._on_center_slider_changed())
        self.sl_cy.on_changed(lambda _val: self._on_center_slider_changed())
        self.sl_angle.on_changed(lambda _val: self._on_angle_changed())
        self.sl_soft.on_changed(lambda _val: self._on_soft_changed())

        self.tb_cx.on_submit(lambda text: self._on_center_text_submitted("x", text))
        self.tb_cy.on_submit(lambda text: self._on_center_text_submitted("y", text))
        self.tb_width.on_submit(lambda text: self._on_size_text_submitted("width", text))
        self.tb_height.on_submit(lambda text: self._on_size_text_submitted("height", text))
        self.tb_angle.on_submit(lambda text: self._on_float_text_submitted(self.sl_angle, self.tb_angle, text, "{:.1f}"))
        self.tb_soft.on_submit(lambda text: self._on_float_text_submitted(self.sl_soft, self.tb_soft, text, "{:.3f}"))
        self.tb_fft_thresh.on_submit(lambda text: self._on_float_text_submitted(self.sl_fft_thresh, self.tb_fft_thresh, text, "{:.1f}"))
        self.tb_fft_highpass.on_submit(lambda text: self._on_float_text_submitted(self.sl_fft_highpass, self.tb_fft_highpass, text, "{:.1f}"))
        self.tb_fft_lowpass.on_submit(lambda text: self._on_float_text_submitted(self.sl_fft_lowpass, self.tb_fft_lowpass, text, "{:.1f}"))

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
        if event.inaxes == self.ax_peaks:
            self.open_popup_peaks()
            return
        if event.inaxes == self.ax_radial:
            self.open_popup_radial()
            return

        if event.inaxes == self.ax_img and event.xdata is not None and event.ydata is not None:
            if getattr(event, "dblclick", False):
                self.open_popup_image("img", "Image with mask overlay", cmap=None)
                return
            self._dragging = True
            cx, cy = self._display_to_original_xy(event.xdata, event.ydata)
            self.set_center(cx, cy)

    def on_mouse_release(self, event):
        self._dragging = False

    def on_mouse_move(self, event):
        if self._dragging and event.inaxes == self.ax_img and event.xdata is not None and event.ydata is not None:
            cx, cy = self._display_to_original_xy(event.xdata, event.ydata)
            self.set_center(cx, cy)
        if event.inaxes == self.ax_fft and event.xdata is not None and event.ydata is not None:
            # Main FFT view uses image pixel axes [0..w, 0..h], so convert to centered freq coords.
            fx = event.xdata - self._fft_raw_logmag_data.shape[1] / 2.0 + 0.5
            fy = self._fft_raw_logmag_data.shape[0] / 2.0 - 0.5 - event.ydata
            self._update_fft_hover_label(fx, fy)
        elif not self._dragging and event.inaxes != self.ax_fft:
            self.txt_fft_hover.set_text("")

    def set_center(self, x: float, y: float):
        self.sl_cx.set_val(np.clip(round(x), 0, self.w - 1))
        self.sl_cy.set_val(np.clip(round(y), 0, self.h - 1))

    def _set_textbox_value(self, textbox: TextBox, text: str) -> None:
        self._syncing_inputs = True
        textbox.set_val(text)
        self._syncing_inputs = False

    def _on_center_slider_changed(self) -> None:
        if not self._syncing_inputs:
            self._set_textbox_value(self.tb_cx, f"{int(round(self.sl_cx.val))}")
            self._set_textbox_value(self.tb_cy, f"{int(round(self.sl_cy.val))}")
        self.update()

    def _on_center_text_submitted(self, axis: str, text: str) -> None:
        if self._syncing_inputs:
            return
        try:
            value = float(text.strip())
        except ValueError:
            if axis == "x":
                self._set_textbox_value(self.tb_cx, f"{int(round(self.sl_cx.val))}")
            else:
                self._set_textbox_value(self.tb_cy, f"{int(round(self.sl_cy.val))}")
            return

        if axis == "x":
            clamped = float(np.clip(round(value), 0, self.w - 1))
            self.sl_cx.set_val(clamped)
        else:
            clamped = float(np.clip(round(value), 0, self.h - 1))
            self.sl_cy.set_val(clamped)

    def _on_size_text_submitted(self, which: str, text: str) -> None:
        if self._syncing_inputs:
            return
        try:
            value = float(text.strip())
        except ValueError:
            if which == "width":
                self._set_textbox_value(self.tb_width, f"{self.sl_width.val:.1f}")
            else:
                self._set_textbox_value(self.tb_height, f"{self.sl_height.val:.1f}")
            return

        if which == "width":
            clamped = float(np.clip(value, 2.0, self.w))
            self.sl_width.set_val(clamped)
        else:
            clamped = float(np.clip(value, 2.0, self.h))
            self.sl_height.set_val(clamped)

    def _on_float_text_submitted(self, slider: Slider, textbox: TextBox, text: str, fmt: str) -> None:
        if self._syncing_inputs:
            return
        try:
            value = float(text.strip())
        except ValueError:
            self._set_textbox_value(textbox, fmt.format(slider.val))
            return

        clamped = float(np.clip(value, slider.valmin, slider.valmax))
        slider.set_val(clamped)

    def _on_angle_changed(self) -> None:
        if not self._syncing_inputs:
            self._set_textbox_value(self.tb_angle, f"{self.sl_angle.val:.1f}")
        self.update()

    def _on_soft_changed(self) -> None:
        if not self._syncing_inputs:
            self._set_textbox_value(self.tb_soft, f"{self.sl_soft.val:.3f}")
        self.update()

    def _format_img_hover_text(self, x: float, y: float) -> str:
        if x < 0 or y < 0 or x > self.disp_w - 1 or y > self.disp_h - 1:
            return f"x={x:.1f}, y={y:.1f}"

        ox, oy = self._display_to_original_xy(x, y)
        ix = int(round(ox))
        iy = int(round(oy))
        ix = int(np.clip(ix, 0, self.w - 1))
        iy = int(np.clip(iy, 0, self.h - 1))

        px = self.rgb_original[iy, ix, :]
        return (
            f"x={ix}, y={iy}, "
            f"R={px[0]:.3f}, G={px[1]:.3f}, B={px[2]:.3f}"
        )

    def _display_to_original_xy(self, x: float, y: float) -> tuple[float, float]:
        if self.disp_w <= 1:
            ox = 0.0
        else:
            ox = float(x) * float(self.w - 1) / float(self.disp_w - 1)

        if self.disp_h <= 1:
            oy = 0.0
        else:
            oy = float(y) * float(self.h - 1) / float(self.disp_h - 1)

        return ox, oy

    def _on_width_changed(self, val: float):
        """Sync height to width if linking is enabled."""
        if self.link_dimensions and not self._linking_sliders:
            self._linking_sliders = True
            self.sl_height.set_val(val)
            self._linking_sliders = False
        if not self._syncing_inputs:
            self._set_textbox_value(self.tb_width, f"{self.sl_width.val:.1f}")
            self._set_textbox_value(self.tb_height, f"{self.sl_height.val:.1f}")
        self.update()

    def _on_height_changed(self, val: float):
        """Sync width to height if linking is enabled."""
        if self.link_dimensions and not self._linking_sliders:
            self._linking_sliders = True
            self.sl_width.set_val(val)
            self._linking_sliders = False
        if not self._syncing_inputs:
            self._set_textbox_value(self.tb_width, f"{self.sl_width.val:.1f}")
            self._set_textbox_value(self.tb_height, f"{self.sl_height.val:.1f}")
        self.update()

    def _on_fft_thresh_changed(self, val: float):
        """Update FFT threshold and refresh."""
        self.fft_threshold_value = float(val)
        if not self._syncing_inputs:
            self._set_textbox_value(self.tb_fft_thresh, f"{self.sl_fft_thresh.val:.1f}")
        self.update()

    def _on_fft_highpass_changed(self, val: float):
        """Update FFT high-pass radius and refresh."""
        self.fft_highpass_percent = float(val)
        if not self._syncing_inputs:
            self._set_textbox_value(self.tb_fft_highpass, f"{self.sl_fft_highpass.val:.1f}")
        self.update()

    def _on_fft_lowpass_changed(self, val: float):
        """Update FFT low-pass radius and refresh."""
        self.fft_lowpass_percent = float(val)
        if not self._syncing_inputs:
            self._set_textbox_value(self.tb_fft_lowpass, f"{self.sl_fft_lowpass.val:.1f}")
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
            txt_hover = ax.text(
                0.02,
                0.02,
                "",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.45, boxstyle="round,pad=0.2"),
            )
        else:
            txt_hover = None

        record = {
            "fig": fig,
            "ax": ax,
            "im": im,
            "source_key": source_key,
            "title": title,
            "txt_hover": txt_hover,
        }
        self._popup_views.append(record)

        if source_key == "fft":
            def on_motion(event):
                if txt_hover is None:
                    return
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    txt_hover.set_text("")
                    fig.canvas.draw_idle()
                    return
                txt_hover.set_text(self._format_fft_hover_text(event.xdata, event.ydata, self._fft_raw_logmag_data))
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", on_motion)

        def on_close(_event):
            self._popup_views = [p for p in self._popup_views if p.get("fig") is not fig]

        fig.canvas.mpl_connect("close_event", on_close)

        fig.tight_layout()
        fig.canvas.manager.set_window_title(f"Detail: {title}")
        fig.show()

    def open_popup_peaks(self) -> None:
        """Open interactive peak-synthesis view from non-zero listed FFT peaks."""
        if self._fft_complex_data.size == 0:
            return

        # User-requested behavior: build synthetic FFT only from non-zero listed peaks.
        nonzero_peaks = [p for p in self._peaks_data if float(p.get("distance", 0.0)) > 1e-9]

        fig = plt.figure(figsize=(16, 9))
        ax_tbl = fig.add_axes([0.03, 0.58, 0.45, 0.36])
        ax_pre = fig.add_axes([0.02, 0.11, 0.18, 0.33])
        ax_fft_orig = fig.add_axes([0.215, 0.11, 0.18, 0.33])
        ax_fft = fig.add_axes([0.41, 0.11, 0.18, 0.33])
        ax_ifft = fig.add_axes([0.605, 0.11, 0.18, 0.33])
        ax_mix = fig.add_axes([0.80, 0.11, 0.18, 0.33])

        ax_tbl.set_axis_off()
        table_text = self._peaks_text_data
        ax_tbl.text(
            0.0,
            1.0,
            table_text,
            transform=ax_tbl.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
        )
        fig.text(
            0.03,
            0.55,
            "Strength meaning: 1.0 = original complex amplitude of that conjugate peak pair.",
            ha="left",
            va="top",
            fontsize=10,
        )

        h, w = self._fft_complex_data.shape
        cy = h // 2
        cx = w // 2

        def _offset_to_index(dx: float, dy: float) -> tuple[int, int] | None:
            ix = int(np.round(dx + (w / 2.0 - 0.5)))
            iy = int(np.round((h / 2.0 - 0.5) - dy))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                return None
            return iy, ix

        peak_models: list[dict[str, int]] = []
        for p in nonzero_peaks:
            idx = _offset_to_index(float(p["x"]), float(p["y"]))
            if idx is None:
                continue
            iy, ix = idx
            my = int(2 * cy - iy)
            mx = int(2 * cx - ix)
            peak_models.append({"iy": iy, "ix": ix, "my": my, "mx": mx})

        strengths = np.ones(len(peak_models), dtype=np.float32)

        def _fft_display_from_complex(arr: np.ndarray) -> np.ndarray:
            mag = np.log1p(np.abs(arr)).astype(np.float32)
            mmax = float(mag.max())
            if mmax <= 1e-12:
                return np.zeros_like(mag, dtype=np.float32)
            return np.power(np.clip(mag / mmax, 0.0, 1.0), 0.6).astype(np.float32)

        def _ifft_signed_display(arr_c: np.ndarray) -> np.ndarray:
            # Preserve sign in the inverse visualization; midpoint gray is 0.
            inv_full = np.real(np.fft.ifft2(np.fft.ifftshift(arr_c))).astype(np.float32)

            # Apply the same inverse cropping path as the main FFT view so zero/square
            # padding corners do not dominate popup interpretation.
            meta = self._fft_recon_meta
            inv = inv_full
            if bool(meta.get("zero_pad", False)):
                in_h = int(meta.get("in_h", inv.shape[0]))
                in_w = int(meta.get("in_w", inv.shape[1]))
                y0 = in_h // 2
                x0 = in_w // 2
                inv = inv[y0:y0 + in_h, x0:x0 + in_w]

            if bool(meta.get("square_fft", False)):
                src_h = int(meta.get("src_h", inv.shape[0]))
                src_w = int(meta.get("src_w", inv.shape[1]))
                in_h2 = int(meta.get("in_h", inv.shape[0]))
                in_w2 = int(meta.get("in_w", inv.shape[1]))
                if src_h != in_h2 or src_w != in_w2:
                    pad_top = int(meta.get("pad_top", 0))
                    pad_left = int(meta.get("pad_left", 0))
                    inv = inv[pad_top:pad_top + src_h, pad_left:pad_left + src_w]

            scale = float(np.percentile(np.abs(inv), 99.5))
            if scale <= 1e-12:
                return np.full_like(inv, 0.5, dtype=np.float32)
            return np.clip(0.5 + 0.5 * (inv / scale), 0.0, 1.0).astype(np.float32)

        def _build_synth_complex() -> np.ndarray:
            out = np.zeros_like(self._fft_complex_data, dtype=np.complex64)
            for k, model in enumerate(peak_models):
                gain = float(strengths[k])
                iy = int(model["iy"])
                ix = int(model["ix"])
                my = int(model["my"])
                mx = int(model["mx"])
                out[iy, ix] = self._fft_complex_data[iy, ix] * gain
                if 0 <= my < h and 0 <= mx < w and not (my == iy and mx == ix):
                    out[my, mx] = self._fft_complex_data[my, mx] * gain
            return out

        synth_c = _build_synth_complex()
        synth_mag = _fft_display_from_complex(synth_c)

        synth_ifft = _ifft_signed_display(synth_c)

        im_pre = ax_pre.imshow(self._pre_img_data, cmap="gray", vmin=0, vmax=1)
        ax_pre.set_title("FFT input (mask/filter applied)")
        ax_pre.set_axis_off()

        orig_fft_disp = _fft_display_from_complex(self._fft_complex_data)
        im_fft_orig = ax_fft_orig.imshow(orig_fft_disp, cmap="gray", vmin=0, vmax=1)
        ax_fft_orig.set_title("Original FFT (current view)")
        ax_fft_orig.set_axis_off()

        im_fft = ax_fft.imshow(synth_mag, cmap="gray", vmin=0, vmax=1)
        ax_fft.set_title("Synthetic FFT")
        ax_fft.set_axis_off()

        im_ifft = ax_ifft.imshow(synth_ifft, cmap="gray", vmin=0, vmax=1)
        ax_ifft.set_title("Inverse FFT")
        ax_ifft.set_axis_off()

        def _build_mix_rgb(pre_img: np.ndarray, ifft_signed_img: np.ndarray) -> np.ndarray:
            pre_g = np.clip(pre_img.astype(np.float32), 0.0, 1.0)
            inv_r = np.clip(np.abs(ifft_signed_img.astype(np.float32) - 0.5) * 2.0, 0.0, 1.0)
            mix = np.zeros((pre_g.shape[0], pre_g.shape[1], 3), dtype=np.float32)
            mix[..., 0] = inv_r
            mix[..., 1] = pre_g
            return mix

        mix_rgb = _build_mix_rgb(self._pre_img_data, synth_ifft)
        im_mix = ax_mix.imshow(mix_rgb, vmin=0, vmax=1)
        ax_mix.set_title("Composite (R=inverse, G=input)")
        ax_mix.set_axis_off()

        slider_axes: list[plt.Axes] = []
        sliders: list[Slider] = []
        y0 = 0.93
        dy = 0.032
        max_rows = min(12, len(peak_models))

        for i in range(max_rows):
            ax_s = fig.add_axes([0.52, y0 - i * dy, 0.45, 0.022])
            p = nonzero_peaks[i]
            label = f"peak {i+1} gain ({p['x']:.0f},{p['y']:.0f})"
            s = Slider(ax_s, label, 0.0, 3.0, valinit=1.0, valfmt="%1.2f")
            slider_axes.append(ax_s)
            sliders.append(s)

        reset_y = max(0.53, y0 - max_rows * dy - 0.02)
        ax_reset = fig.add_axes([0.84, reset_y, 0.13, 0.035])
        btn_reset = Button(ax_reset, "Reset strengths")

        def _refresh_views() -> None:
            synth = _build_synth_complex()
            smag = _fft_display_from_complex(synth)
            iimg = _ifft_signed_display(synth)
            im_fft.set_data(smag)
            im_ifft.set_data(iimg)
            im_pre.set_data(self._pre_img_data)
            im_fft_orig.set_data(_fft_display_from_complex(self._fft_complex_data))
            im_mix.set_data(_build_mix_rgb(self._pre_img_data, iimg))
            fig.canvas.draw_idle()

        def _on_slider(_val: float) -> None:
            for i, s in enumerate(sliders):
                strengths[i] = float(s.val)
            _refresh_views()

        for s in sliders:
            s.on_changed(_on_slider)

        def _on_reset(_event) -> None:
            for s in sliders:
                s.reset()

        btn_reset.on_clicked(_on_reset)
        fig.canvas.manager.set_window_title("Detail: Top FFT peaks synthesis")
        fig.show()

    def open_popup_radial(self) -> None:
        """Open a large radial FFT power graph."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        line, = ax.plot(self._radial_r_data, self._radial_log_power_data, color="tab:orange", linewidth=1.8)
        ax.set_title("Radial FFT power")
        ax.set_xlabel("radius (px)")
        ax.set_ylabel("log1p(sum |F|^2)")
        ax.grid(True, alpha=0.25)
        ax.relim()
        ax.autoscale_view()

        record = {
            "fig": fig,
            "ax": ax,
            "line": line,
            "source_key": "radial",
            "title": "Radial FFT power",
        }
        self._popup_views.append(record)

        def on_close(_event):
            self._popup_views = [p for p in self._popup_views if p.get("fig") is not fig]

        fig.canvas.mpl_connect("close_event", on_close)
        fig.canvas.manager.set_window_title("Detail: Radial FFT power")
        fig.tight_layout()
        fig.show()

    def _get_source_image(self, source_key: str) -> np.ndarray:
        if source_key == "img":
            return self._img_overlay_data
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

    @staticmethod
    def _fft_freq_to_index(x: float, y: float, shape: tuple[int, int]) -> tuple[int, int] | None:
        h, w = shape
        ix = int(np.round(x + (w / 2.0 - 0.5)))
        iy = int(np.round((h / 2.0 - 0.5) - y))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return None
        return iy, ix

    def _format_fft_hover_text(self, x: float, y: float, raw_fft: np.ndarray) -> str:
        idx = self._fft_freq_to_index(x, y, raw_fft.shape)
        if idx is None:
            return ""
        iy, ix = idx
        fx = ix - raw_fft.shape[1] / 2.0 + 0.5
        fy = raw_fft.shape[0] / 2.0 - 0.5 - iy
        raw_val = float(raw_fft[iy, ix])
        return f"fx={fx:.2f}, fy={fy:.2f}, raw log|F|={raw_val:.6f}"

    def _update_fft_hover_label(self, x: float, y: float) -> None:
        if self._fft_raw_logmag_data.size == 0:
            return
        self.txt_fft_hover.set_text(self._format_fft_hover_text(x, y, self._fft_raw_logmag_data))
        self.fig.canvas.draw_idle()

    def _refresh_open_popups(self) -> None:
        active: list[dict] = []
        for popup in self._popup_views:
            fig = popup["fig"]
            if not plt.fignum_exists(fig.number):
                continue

            source_key = popup["source_key"]
            ax = popup["ax"]
            if source_key in {"img", "pre", "fft", "ifft"}:
                img = self._get_source_image(source_key)
                im = popup["im"]
                extent = self._get_source_extent(source_key, img)
                im.set_data(img)
                im.set_extent(extent)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
            elif source_key == "peaks":
                popup["txt"].set_text(self._peaks_text_data)
            elif source_key == "radial":
                popup["line"].set_data(self._radial_r_data, self._radial_log_power_data)
                ax.relim()
                ax.autoscale_view()
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

        if (self.disp_h, self.disp_w) != (self.h, self.w):
            mask_disp = np.asarray(
                Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8)).resize(
                    (self.disp_w, self.disp_h), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        else:
            mask_disp = mask

        # Show scalar channel as background, not RGB, when useful?
        # Keep RGB image visible and overlay mask; title says selected channel.
        self.mask_overlay.set_data(mask_disp)
        self.mask_overlay.set_alpha(0.35)
        # Keep a composed RGB+mask frame for popup zoom inspection.
        overlay_rgb = plt.get_cmap("magma")(mask_disp)[..., :3]
        self._img_overlay_data = np.clip((1.0 - 0.35) * self.rgb_display + 0.35 * overlay_rgb, 0.0, 1.0)

        (
            pre_img,
            fft_img,
            fft_raw_logmag,
            radial_r,
            radial_log_power,
            ifft_img,
            fft_complex,
            fft_recon_meta,
            low_pass_removed_percent,
            high_pass_removed_percent,
            remaining_percent,
            peaks,
        ) = compute_fft_products(
            channel=channel,
            mask=mask,
            zero_pad=self.zero_pad,
            square_fft=self.square_fft,
            square_pad_mode=self.square_pad_mode,
            fft_highpass_percent=self.fft_highpass_percent,
            fft_lowpass_percent=self.fft_lowpass_percent,
            fft_threshold_percent=self.fft_threshold_value,
            debug_peak_stats=self.debug_peak_stats,
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
        self._fft_raw_logmag_data = fft_raw_logmag
        self._fft_complex_data = fft_complex
        self._fft_recon_meta = fft_recon_meta

        self.im_ifft.set_data(ifft_img)
        self.im_ifft.set_extent([0, ifft_img.shape[1], ifft_img.shape[0], 0])
        self.ax_ifft.set_xlim(0, ifft_img.shape[1])
        self.ax_ifft.set_ylim(ifft_img.shape[0], 0)
        self._ifft_img_data = ifft_img

        self.radial_line.set_data(radial_r, radial_log_power)
        self.ax_radial.relim()
        self.ax_radial.autoscale_view()
        self._radial_r_data = radial_r
        self._radial_log_power_data = radial_log_power

        self.ax_img.set_title(f"Image with mask overlay | channel: {self.channel_name}")
        self.ax_pre.set_title("Masked image crop (zeros outside mask removed)")
        self.ax_fft.set_title("2D FFT log magnitude; center = low frequency")
        self.txt_fft_metric.set_text(
            f"low-pass removed: {low_pass_removed_percent:.2f}%\n"
            f"high-pass removed: {high_pass_removed_percent:.2f}%\n"
            f"remaining power: {remaining_percent:.2f}%"
        )
        self.ax_ifft.set_title("Inverse FFT of spectrum (real part)")
        peak_lines = [
            "#      x      y   dist   ang   phase   peak   logsum  broad",
            "----------------------------------------------------------",
        ]

        fft_h, fft_w = self._fft_complex_data.shape
        def _phase_deg_for_peak(px: float, py: float) -> float:
            ix = int(np.round(px + (fft_w / 2.0 - 0.5)))
            iy = int(np.round((fft_h / 2.0 - 0.5) - py))
            if ix < 0 or ix >= fft_w or iy < 0 or iy >= fft_h:
                return 0.0
            z = self._fft_complex_data[iy, ix]
            if abs(z) <= 1e-12:
                return 0.0
            return float(np.degrees(np.angle(z)))

        for idx, peak in enumerate(peaks, start=1):
            phase_deg = _phase_deg_for_peak(float(peak["x"]), float(peak["y"]))
            peak_lines.append(
                f"{idx:>1} {peak['x']:>6.0f} {peak['y']:>6.0f}"
                f" {peak['distance']:>6.1f} {peak['angle_deg']:>6.0f}"
                f" {phase_deg:>7.1f}"
                f" {peak['peak_val']:>6.2f}"
                f" {peak['log_power_sum']:>8.2f} {peak['broadness']:>6.2f}"
            )
        peaks_text = "\n".join(peak_lines)
        self.txt_peaks.set_text(peaks_text)
        self._peaks_text_data = peaks_text
        self._peaks_data = peaks

        self._refresh_open_popups()
        self.fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 2D FFT explorer for color images.")
    parser.add_argument("image", help="Input image path, e.g. JPG, PNG, TIFF; HEIC if pillow-heif is installed.")
    parser.add_argument("--max-side", type=int, default=1200, help="Resize largest dimension for speed. Use 0 for no resize.")
    parser.add_argument("--fig-width", type=float, default=14.0, help="Main window width in inches.")
    parser.add_argument("--fig-height", type=float, default=9.5, help="Main window height in inches.")
    parser.add_argument(
        "--debug-peak-stats",
        action="store_true",
        help="Print per-peak support geometry from raw FFT log-magnitude for broadness verification.",
    )
    args = parser.parse_args()

    max_side = None if args.max_side == 0 else args.max_side
    rgb_original = load_image_rgb(args.image, max_side=None)
    rgb_display = load_image_rgb(args.image, max_side=max_side)

    fig_width = max(6.0, float(args.fig_width))
    fig_height = max(5.0, float(args.fig_height))
    explorer = FFTExplorer(
        rgb_original,
        rgb_display=rgb_display,
        debug_peak_stats=args.debug_peak_stats,
        figure_size=(fig_width, fig_height),
    )
    plt.show()


if __name__ == "__main__":
    main()

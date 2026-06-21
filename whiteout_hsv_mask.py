#!/usr/bin/env python3
"""Set pixels matching a specific HSV box to white and save/display output."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set pixels matching HSV bounds to white and save the result."
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "--out",
        default="whiteout_hsv_mask_output.png",
        help="Output image path (default: whiteout_hsv_mask_output.png)",
    )
    args = parser.parse_args()

    img = Image.open(args.image)
    img = ImageOps.exif_transpose(img).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32) / 255.0

    hsv = rgb_to_hsv(np.clip(rgb, 0.0, 1.0).astype(np.float32))
    hsv_u8 = np.clip(np.round(hsv * 255.0), 0, 255).astype(np.uint8)
    h_u8_full = hsv_u8[..., 0]
    s_u8_full = hsv_u8[..., 1]
    v_u8_full = hsv_u8[..., 2]

    mask = (
        (h_u8_full >= 219) & (h_u8_full <= 242)
        & (s_u8_full >= 100) & (s_u8_full <= 200)
        & (v_u8_full >= 80) & (v_u8_full <= 250)
    )

    out = rgb.copy()
    out[mask] = 1.0

    Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(args.out)
    print(f"Saved: {args.out}")
    print(f"Masked pixels set to white: {int(np.count_nonzero(mask))}")

    plt.figure(figsize=(9, 7), constrained_layout=True)
    plt.imshow(out, interpolation="nearest")
    plt.title("Original image with selected HSV pixels set to white")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()

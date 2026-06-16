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


def _smooth_closed_curve(points: np.ndarray, window: int = 7, passes: int = 2) -> np.ndarray:
    """Light circular smoothing to make a traced polygon appear spline-like."""
    if points.shape[0] < 8:
        return points

    n = points.shape[0]
    win = max(3, int(window))
    if win % 2 == 0:
        win += 1
    half = win // 2

    out = points.astype(np.float64, copy=True)
    kernel = np.ones(win, dtype=np.float64) / float(win)
    for _ in range(max(1, int(passes))):
        px = np.pad(out[:, 0], (half, half), mode="wrap")
        py = np.pad(out[:, 1], (half, half), mode="wrap")
        sx = np.convolve(px, kernel, mode="valid")
        sy = np.convolve(py, kernel, mode="valid")
        out = np.column_stack([sx[:n], sy[:n]])
    return out.astype(np.float32)


def _plot_trace_failure_debug(
    raw_map: np.ndarray,
    coarse: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    curr_y: int,
    curr_x: int,
    ring_offsets: list[tuple[int, int]],
    threshold: float,
    ring_steps: int,
    step: int,
    spline_point_index: int,
    is_transition_fn,
) -> None:
    """Plot a legible diagnostic view for the current failure point and ring candidates."""
    h, w = raw_map.shape
    px = int(xs[curr_x])
    py = int(ys[curr_y])
    radius_px = int(ring_steps * step)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    im0 = ax_full.imshow(raw_map, cmap="magma", vmin=0, vmax=100, interpolation="nearest")
    ax_full.set_title("Failure point context")
    ax_full.scatter([px], [py], s=60, c="cyan", edgecolors="black", linewidths=0.8, zorder=5)
    circ = plt.Circle((px, py), radius_px, fill=False, color="white", linewidth=1.4, alpha=0.9)
    ax_full.add_patch(circ)
    ax_full.set_xlim(max(0, px - 3 * radius_px), min(w - 1, px + 3 * radius_px))
    ax_full.set_ylim(min(h - 1, py + 3 * radius_px), max(0, py - 3 * radius_px))

    ax_zoom.imshow(raw_map, cmap="magma", vmin=0, vmax=100, interpolation="nearest")
    ax_zoom.set_title("Failure-point ring candidates")
    ax_zoom.scatter([px], [py], s=80, c="cyan", edgecolors="black", linewidths=0.9, zorder=6)

    for dy, dx in ring_offsets:
        ny = curr_y + dy
        nx = curr_x + dx
        if ny < 1 or nx < 1 or ny >= coarse.shape[0] - 1 or nx >= coarse.shape[1] - 1:
            continue
        npx = int(xs[nx])
        npy = int(ys[ny])
        val = float(coarse[ny, nx])
        hi = val >= threshold
        trans = bool(is_transition_fn(ny, nx))
        color = "lime" if hi else "red"
        marker = "o" if trans else "x"
        ax_zoom.scatter([npx], [npy], s=50, c=color, marker=marker, linewidths=1.0, zorder=5)
        ax_zoom.text(
            npx + 2,
            npy + 2,
            f"{val:.1f}",
            color="white",
            fontsize=7,
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 0.2, "edgecolor": "none"},
            zorder=6,
        )

    ax_zoom.set_xlim(max(0, px - 2 * radius_px), min(w - 1, px + 2 * radius_px))
    ax_zoom.set_ylim(min(h - 1, py + 2 * radius_px), max(0, py - 2 * radius_px))

    fig.colorbar(im0, ax=[ax_full, ax_zoom], fraction=0.03, pad=0.02, label="Raw value")
    fig.suptitle(
        f"Tracer debug at spline point {spline_point_index}\n"
        f"point=(y={py}, x={px}), threshold={threshold:.1f}, ring={ring_steps} steps ({radius_px} px)"
    )
    plt.show()


def _trace_outer_spline_step_cells(
    raw_map: np.ndarray,
    step: int,
    threshold: float = 99.0,
    ring_steps: int = 4,
    clockwise: bool = True,
    debug_start_point: int = 75,
) -> np.ndarray | None:
    """Trace an outer bracelet spline by stepping along threshold transitions on the step grid."""
    h, w = raw_map.shape
    ys = np.arange(0, h, step, dtype=np.int32)
    xs = np.arange(0, w, step, dtype=np.int32)
    if ys.size < 5 or xs.size < 5:
        return None

    coarse = raw_map[ys[:, None], xs[None, :]]
    hi = coarse >= threshold
    gh, gw = coarse.shape

    cy = gh // 2
    cx = gw // 2

    # Start from one end of the centerline and move inward until value drops below threshold.
    start: tuple[int, int] | None = None
    start_scan_tail: list[tuple[int, int, float]] = []
    endpoint_orders = [
        [(cy, ix) for ix in range(gw - 1, -1, -1)],
        [(cy, ix) for ix in range(0, gw)],
        [(iy, cx) for iy in range(gh - 1, -1, -1)],
        [(iy, cx) for iy in range(0, gh)],
    ]
    for line in endpoint_orders:
        scan_vals: list[tuple[int, int, float]] = []
        for iy, ix in line:
            scan_vals.append((iy, ix, float(coarse[iy, ix])))
            # Use the exact same HIGH/LOW rule as ring evaluation: HIGH iff value >= threshold.
            if not hi[iy, ix]:
                start = (iy, ix)
                start_scan_tail = scan_vals[-5:]
                break
        if start is not None:
            break
    trace_output = False

    if start is None:
        print(
            "Tracer stop: no start point found on center lines where raw<threshold; "
            f"threshold={threshold:.3f}"
        )
        return None

    print("Initial scan tail (last 5 checked from edge):")
    tail_base = max(0, len(start_scan_tail) - 5)
    for j, (iy, ix, val) in enumerate(start_scan_tail, start=tail_base):
        print(
            f"  edge_check[{j}]: grid(y={iy},x={ix}) "
            f"pixel(y={int(ys[iy])},x={int(xs[ix])}) raw={val:.3f}"
        )

    def is_transition(iy: int, ix: int) -> bool:
        # Transition is defined only by state change from current point to candidate.
        # No neighborhood/3x3 test.
        return bool(hi[iy, ix] != hi[curr_y, curr_x])

    # Cells approximately ring_steps away from current cell.
    ring_offsets: list[tuple[int, int]] = []
    r = float(ring_steps)
    for dy in range(-ring_steps - 1, ring_steps + 2):
        for dx in range(-ring_steps - 1, ring_steps + 2):
            if dx == 0 and dy == 0:
                continue
            d = math.hypot(dx, dy)
            if abs(d - r) <= 0.75:
                ring_offsets.append((dy, dx))
    if not ring_offsets:
        print("Tracer stop: no ring offsets generated; check ring_steps value.")
        return None

    curr_y, curr_x = start
    pts: list[tuple[int, int]] = [(curr_y, curr_x)]
    visited_at: dict[tuple[int, int], int] = {(curr_y, curr_x): 0}
    start_y, start_x = curr_y, curr_x
    print(
        f"Spline point 0: grid(y={curr_y},x={curr_x}) "
        f"pixel(y={int(ys[curr_y])},x={int(xs[curr_x])})"
    )
    prev_dir: tuple[float, float] | None = None
    max_turn_rad = math.radians(90.0)
    stop_reason: str | None = None

    while True:
        trace_output = (len(pts) - 1) >= debug_start_point
        curr_px = int(xs[curr_x])
        curr_py = int(ys[curr_y])

        # Lost-state detection: if the local neighborhood is all above or all below
        # threshold, there is no nearby crossing to follow.
        y0 = max(0, curr_y - ring_steps)
        y1 = min(gh, curr_y + ring_steps + 1)
        x0 = max(0, curr_x - ring_steps)
        x1 = min(gw, curr_x + ring_steps + 1)
        local_hi = hi[y0:y1, x0:x1]
        ly, lx = np.ogrid[y0:y1, x0:x1]
        local_disk = (ly - curr_y) ** 2 + (lx - curr_x) ** 2 <= ring_steps ** 2
        disk_vals = local_hi[local_disk]
        disk_n = int(disk_vals.size)
        disk_hi = int(np.count_nonzero(disk_vals))
        if disk_n > 0 and (disk_hi == 0 or disk_hi == disk_n):
            px = int(xs[curr_x])
            py = int(ys[curr_y])
            raw_here = float(coarse[curr_y, curr_x])
            state = "all>=threshold" if disk_hi == disk_n else "all<threshold"
            print(
                "Tracer lost: uniform neighborhood within "
                f"{ring_steps} steps ({ring_steps * step} px); {state}."
            )
            print(
                "Tracer debug: "
                f"grid(y={curr_y}, x={curr_x}), pixel(y={py}, x={px}), "
                f"raw={raw_here:.3f}, threshold={threshold:.3f}, "
                f"disk_hi={disk_hi}/{disk_n}, points_traced={len(pts)}"
            )
            return None

        best: tuple[int, int] | None = None
        best_score = float("inf")
        cand_total = 0
        cand_oob = 0
        cand_not_transition = 0
        cand_too_short = 0
        cand_turn_reject = 0
        cand_scored = 0
        cand_debug_rows: list[dict[str, float | int | str | bool]] = []

        # Process ring candidates by a strict angular sweep around the CURRENT point
        # so debug angle progression matches (dy, dx) ordering intuitively.
        # Debug lines are printed in this exact same sweep order.
        ordered_candidates: list[tuple[float, int, int]] = []
        for dy, dx in ring_offsets:
            # Use a local geometric angle for the offset vector; y-axis inverted so
            # positive angles follow visual CCW on image coordinates.
            theta_local = math.atan2(-dy, dx)
            cw_angle = (2.0 * math.pi - theta_local) % (2.0 * math.pi)
            ccw_angle = (theta_local + 2.0 * math.pi) % (2.0 * math.pi)
            sweep_angle = cw_angle if clockwise else ccw_angle
            ordered_candidates.append((sweep_angle, dy, dx))
        ordered_candidates.sort(key=lambda t: t[0])

        sweep_label = "CW" if clockwise else "CCW"
        ordered_ring_states: list[bool] = []
        ordered_ring_meta: list[tuple[int, float, int, int, int, int, float]] = []
        in_bounds_entries: list[dict[str, float | int | str | bool]] = []

        for ord_idx, (sweep_angle, dy, dx) in enumerate(ordered_candidates):
            cand_total += 1
            ny = curr_y + dy
            nx = curr_x + dx
            if ny < 1 or nx < 1 or ny >= gh - 1 or nx >= gw - 1:
                cand_oob += 1
                cand_debug_rows.append(
                    {
                        "ord_idx": ord_idx,
                        "sweep_angle": sweep_angle,
                        "dy": dy,
                        "dx": dx,
                        "ny": ny,
                        "nx": nx,
                        "status": "out_of_bounds",
                    }
                )
                if trace_output:
                    print(
                        f"  ring[{ord_idx:02d}] {sweep_label} {math.degrees(sweep_angle):6.2f}deg "
                        f"(dy={dy:+d},dx={dx:+d}): out-of-bounds"
                    )
                continue

            val = float(coarse[ny, nx])
            hi_bool = bool(val >= threshold)
            hi_state = "HIGH" if hi_bool else "LOW"
            row: dict[str, float | int | str | bool] = {
                "ord_idx": ord_idx,
                "sweep_angle": sweep_angle,
                "dy": dy,
                "dx": dx,
                "ny": ny,
                "nx": nx,
                "pixel_y": int(ys[ny]),
                "pixel_x": int(xs[nx]),
                "raw": val,
                "state": hi_state,
                "hi": hi_bool,
            }
            ordered_ring_states.append(hi_bool)
            ordered_ring_meta.append(
                (ord_idx, sweep_angle, dy, dx, ny, nx, val)
            )
            in_bounds_entries.append(row)

        # Transition along sweep means a state change relative to the previous
        # in-bounds sample in angular order. Only these changes count.
        ring_n = len(ordered_ring_states)
        if ring_n < 2:
            print(
                "Tracer stop: insufficient in-bounds ring samples to validate transitions; "
                f"point grid(y={curr_y},x={curr_x})"
            )
            return None

        for i, row in enumerate(in_bounds_entries):
            prev_state = ordered_ring_states[(i - 1) % ring_n]
            curr_state = ordered_ring_states[i]
            trans = bool(curr_state != prev_state)
            row["transition"] = trans

        for row in in_bounds_entries:
            ord_idx = int(row["ord_idx"])
            sweep_angle = float(row["sweep_angle"])
            dy = int(row["dy"])
            dx = int(row["dx"])
            ny = int(row["ny"])
            nx = int(row["nx"])
            val = float(row["raw"])
            hi_state = str(row["state"])
            trans = bool(row["transition"])

            if trace_output:
                print(
                    f"  ring[{ord_idx:02d}] {sweep_label} {math.degrees(sweep_angle):6.2f}deg "
                    f"(dy={dy:+d},dx={dx:+d}): "
                    f"grid(y={ny},x={nx}) pixel(y={int(ys[ny])},x={int(xs[nx])}) "
                    f"raw={val:.3f} state={hi_state} transition={'Y' if trans else 'N'}"
                )

            if not trans:
                cand_not_transition += 1
                row["status"] = "reject_non_transition"
                cand_debug_rows.append(row)
                continue

            move = math.hypot(ny - curr_y, nx - curr_x)
            if move < 1.0:
                cand_too_short += 1
                row["move"] = move
                row["status"] = "reject_too_short"
                cand_debug_rows.append(row)
                continue
            row["move"] = move

            # Purely local geometry: keep the turn angle between successive
            # segments small; avoid any global-angle reference.
            if prev_dir is None:
                turn_angle = 0.0
                orientation_penalty = 0.0
                row["turn_deg"] = 0.0
                row["turn_limit_deg"] = math.degrees(max_turn_rad)
                row["cross"] = 0.0
                row["wrong_turn"] = False
            else:
                mvx = (nx - curr_x) / move
                mvy = (ny - curr_y) / move
                dot = prev_dir[0] * mvx + prev_dir[1] * mvy
                dot = float(np.clip(dot, -1.0, 1.0))
                turn_angle = math.acos(dot)
                turn_deg = math.degrees(turn_angle)
                row["dot"] = dot
                row["turn_deg"] = turn_deg
                row["turn_limit_deg"] = math.degrees(max_turn_rad)
                if turn_angle > max_turn_rad:
                    cand_turn_reject += 1
                    row["status"] = "reject_turn"
                    cand_debug_rows.append(row)
                    continue

                # Direction preference is a tie-breaker only.
                cross = prev_dir[0] * mvy - prev_dir[1] * mvx
                wrong_turn = cross > 0.0 if clockwise else cross < 0.0
                orientation_penalty = 0.20 * abs(cross) if wrong_turn else 0.0
                row["cross"] = cross
                row["wrong_turn"] = wrong_turn

            score = turn_angle + 0.03 * abs(move - r) + orientation_penalty
            cand_scored += 1
            row["orientation_penalty"] = orientation_penalty
            row["score"] = score
            row["status"] = "accepted_scored"
            cand_debug_rows.append(row)

            if score < best_score:
                best_score = score
                best = (ny, nx)

        low_to_high = 0
        high_to_low = 0
        for i in range(ring_n):
            a = ordered_ring_states[i]
            b = ordered_ring_states[(i + 1) % ring_n]
            if (not a) and b:
                low_to_high += 1
            elif a and (not b):
                high_to_low += 1

        if low_to_high != 1 or high_to_low != 1:
            ring_raw_vals = [val for _, _, _, _, _, _, val in ordered_ring_meta]
            ring_raw_vals_sorted = sorted(ring_raw_vals)
            ring_n_vals = len(ring_raw_vals_sorted)
            median_msg = "median=unavailable"
            if ring_n_vals > 0:
                mid = ring_n_vals // 2
                if ring_n_vals % 2 == 1:
                    median_msg = f"median={ring_raw_vals_sorted[mid]:.3f}"
                else:
                    lo = ring_raw_vals_sorted[mid - 1]
                    hi_v = ring_raw_vals_sorted[mid]
                    median_msg = (
                        f"median-neighbors={lo:.3f},{hi_v:.3f}"
                    )

            print(
                "Tracer stop: ring transition count mismatch "
                f"at grid(y={curr_y},x={curr_x}) pixel(y={curr_py},x={curr_px}); "
                f"LOW->HIGH={low_to_high}, HIGH->LOW={high_to_low}, expected 1 each; "
                f"{median_msg}."
            )
            for ord_idx, sweep_angle, dy, dx, ny, nx, val in ordered_ring_meta:
                state_txt = "HIGH" if val >= threshold else "LOW"
                print(
                    f"  ring[{ord_idx:02d}] {sweep_label} {math.degrees(sweep_angle):6.2f}deg "
                    f"(dy={dy:+d},dx={dx:+d}) grid(y={ny},x={nx}) "
                    f"pixel(y={int(ys[ny])},x={int(xs[nx])}) raw={val:.3f} state={state_txt}"
                )

            # Show a visual diagnostic snapshot at the mismatch point.
            _plot_trace_failure_debug(
                raw_map=raw_map,
                coarse=coarse,
                ys=ys,
                xs=xs,
                curr_y=curr_y,
                curr_x=curr_x,
                ring_offsets=[(dy, dx) for _, dy, dx in ordered_candidates],
                threshold=threshold,
                ring_steps=ring_steps,
                step=step,
                spline_point_index=len(pts) - 1,
                is_transition_fn=is_transition,
            )

            if len(pts) >= 2:
                prev_y, prev_x = pts[-2]
                _plot_trace_failure_debug(
                    raw_map=raw_map,
                    coarse=coarse,
                    ys=ys,
                    xs=xs,
                    curr_y=prev_y,
                    curr_x=prev_x,
                    ring_offsets=[(dy, dx) for _, dy, dx in ordered_candidates],
                    threshold=threshold,
                    ring_steps=ring_steps,
                    step=step,
                    spline_point_index=len(pts) - 2,
                    is_transition_fn=is_transition,
                )
            return None

        if best is None:
            print(
                "Tracer stopped: no valid transition candidate at this point."
            )
            cand_in_bounds = cand_total - cand_oob
            cand_transition = cand_in_bounds - cand_not_transition
            print(
                "Tracer debug(no_candidate): "
                f"grid(y={curr_y},x={curr_x}) pixel(y={curr_py},x={curr_px}) "
                f"raw={float(coarse[curr_y, curr_x]):.3f} threshold={threshold:.3f}"
            )
            print(
                "  candidate counts: "
                f"total={cand_total}, out_of_bounds={cand_oob}, in_bounds={cand_in_bounds}, "
                f"transition={cand_transition}, non_transition={cand_not_transition}, "
                f"too_short={cand_too_short}, turn_reject={cand_turn_reject}, "
                f"scored={cand_scored}"
            )
            if prev_dir is not None:
                print(
                    "  local-turn config: "
                    f"max_turn_deg={math.degrees(max_turn_rad):.2f}, "
                    f"prev_dir=({prev_dir[0]:.3f},{prev_dir[1]:.3f})"
                )
            print("  checked ring candidates (in sweep order):")
            for row in cand_debug_rows:
                ord_idx = int(row["ord_idx"])
                sweep_angle = float(row["sweep_angle"])
                dy = int(row["dy"])
                dx = int(row["dx"])
                status = str(row.get("status", "unknown"))
                if status == "out_of_bounds":
                    print(
                        f"    ring[{ord_idx:02d}] {sweep_label} {math.degrees(sweep_angle):6.2f}deg "
                        f"(dy={dy:+d},dx={dx:+d}) -> out_of_bounds"
                    )
                    continue

                ny = int(row["ny"])
                nx = int(row["nx"])
                py = int(row["pixel_y"])
                px = int(row["pixel_x"])
                raw_val = float(row["raw"])
                state = str(row["state"])
                trans = bool(row["transition"])
                base = (
                    f"    ring[{ord_idx:02d}] {sweep_label} {math.degrees(sweep_angle):6.2f}deg "
                    f"(dy={dy:+d},dx={dx:+d}) grid(y={ny},x={nx}) pixel(y={py},x={px}) "
                    f"raw={raw_val:.3f} state={state} transition={'Y' if trans else 'N'}"
                )

                if status == "reject_non_transition":
                    print(base + " -> reject_non_transition")
                elif status == "reject_too_short":
                    print(base + f" -> reject_too_short move={float(row['move']):.3f}")
                elif status == "reject_turn":
                    print(
                        base
                        + " -> reject_turn "
                        + f"turn_deg={float(row['turn_deg']):.2f} > limit={float(row['turn_limit_deg']):.2f}"
                    )
                else:
                    print(
                        base
                        + " -> accepted_scored "
                        + f"turn_deg={float(row['turn_deg']):.2f}, "
                        + f"cross={float(row['cross']):.3f}, "
                        + f"wrong_turn={bool(row['wrong_turn'])}, "
                        + f"score={float(row['score']):.4f}"
                    )
            stop_reason = "no_candidate"
            break

        next_y, next_x = best
        step_dx = float(next_x - curr_x)
        step_dy = float(next_y - curr_y)
        step_norm = math.hypot(step_dx, step_dy)
        if step_norm > 1e-9:
            prev_dir = (step_dx / step_norm, step_dy / step_norm)
        curr_y, curr_x = next_y, next_x
        pts.append((curr_y, curr_x))
        print(
            f"Spline point {len(pts) - 1}: grid(y={curr_y},x={curr_x}) "
            f"pixel(y={int(ys[curr_y])},x={int(xs[curr_x])})"
        )
        if len(pts) >= 3:
            y0, x0 = pts[-3]
            y1, x1 = pts[-2]
            y2, x2 = pts[-1]
            v1x = float(x1 - x0)
            v1y = float(y1 - y0)
            v2x = float(x2 - x1)
            v2y = float(y2 - y1)
            n1 = math.hypot(v1x, v1y)
            n2 = math.hypot(v2x, v2y)
            if n1 > 1e-9 and n2 > 1e-9:
                dot = float(np.clip((v1x * v2x + v1y * v2y) / (n1 * n2), -1.0, 1.0))
                turn_deg = math.degrees(math.acos(dot))
                print(
                    f"  turn angle at point {len(pts) - 1}: {turn_deg:.2f} deg "
                    f"(segments {len(pts)-3}->{len(pts)-2} and {len(pts)-2}->{len(pts)-1})"
                )
            else:
                print(
                    f"  turn angle at point {len(pts) - 1}: unavailable (zero-length segment)"
                )
        else:
            print(
                f"  turn angle at point {len(pts) - 1}: unavailable (need >= 3 points)"
            )

        if len(pts) > 30 and math.hypot(curr_y - start_y, curr_x - start_x) <= max(1.5, 0.5 * r):
            stop_reason = "closed_loop"
            break

        key = (curr_y, curr_x)
        if key in visited_at:
            first_idx = visited_at[key]
            print(
                "Tracer stop: detected repeating cycle away from start; "
                f"revisited grid(y={curr_y},x={curr_x}) at spline point {len(pts) - 1}, "
                f"first seen at point {first_idx}."
            )
            print(
                "Tracer debug(cycle): "
                f"pixel(y={int(ys[curr_y])},x={int(xs[curr_x])}), "
                f"raw={float(coarse[curr_y, curr_x]):.3f}, threshold={threshold:.3f}"
            )
            return None
        visited_at[key] = len(pts) - 1

    if stop_reason is None:
        return None

    if len(pts) < 16:
        return None

    # Convert from step-grid indices to image pixel coordinates.
    pix = np.array([[float(xs[ix]), float(ys[iy])] for iy, ix in pts], dtype=np.float32)
    if np.hypot(*(pix[0] - pix[-1])) > 1e-6:
        pix = np.vstack([pix, pix[0]])
    pix = _smooth_closed_curve(pix, window=7, passes=2)
    if np.hypot(*(pix[0] - pix[-1])) > 1e-6:
        pix = np.vstack([pix, pix[0]])
    return pix


def load_image_and_luminance(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    rgb = np.asarray(img).astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return rgb, y.astype(np.float32)


def make_gaussian_window(size: int = 265, softness: float = 0.2) -> np.ndarray:
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
        description="Map local FFT metric using 200x200 Gaussian-windowed FFT."
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
        default=8,
        help="Stride in pixels for sampling (default: 8)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Gaussian FFT window size in pixels (default: 200)",
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
    window_size = max(8, int(args.window_size))
    display_scale = args.display_scale
    near100_alpha = max(0.05, float(args.near100_alpha))
    rgb, y = load_image_and_luminance(args.image)
    h, w = y.shape

    win_size = window_size
    softness = 0.2
    window = make_gaussian_window(size=win_size, softness=softness)

    # Edge padding so every image pixel can be used as a window center.
    pad = win_size // 2
    y_pad = np.pad(y, ((pad, pad), (pad, pad)), mode="edge")

    # High-pass mask for a fixed local FFT grid.
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
        window_size=win_size,
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
    window_size: int,
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

    if metric == "hp_removed":
        traced = _trace_outer_spline_step_cells(
            out,
            step=max(1, int(step)),
            threshold=99.0,
            ring_steps=4,
            clockwise=True,
            debug_start_point=75,
        )
        if traced is not None:
            ax_src.plot(traced[:, 0], traced[:, 1], color="white", linewidth=2.0, alpha=0.95)

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
        f"window={window_size}x{window_size}, softness=0.2, stride sampling, "
        f"step={step}, hp={hp_percent:.2f}%, metric={metric}, "
        f"processed={processed}/{total}"
    )

    ax_src.axis("off")
    ax_map.axis("off")
    plt.show()


if __name__ == "__main__":
    main()

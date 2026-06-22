#!/usr/bin/env python3
"""Load a saved map and compute spline overlays.

This script reads the original image, a precomputed map (.npy), and metadata
(.json), computes outer/inner/centerline splines, writes spline geometry to
JSON, and displays the original image + map with splines overlaid on both.
"""

# Suggested commands:
# python map_from_fft.py beads-photo-2.jpg
# python find_splines.py beads-photo-2.jpg
# python find_splines.py beads-photo-2_map.npy beads-photo-2_map_metadata.json

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image, ImageOps


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


def _cross2d(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
    eps: float = 1e-9,
) -> bool:
    p1x, p1y = p1
    p2x, p2y = p2
    q1x, q1y = q1
    q2x, q2y = q2

    r_x = p2x - p1x
    r_y = p2y - p1y
    s_x = q2x - q1x
    s_y = q2y - q1y
    rxs = _cross2d(r_x, r_y, s_x, s_y)
    qmp_x = q1x - p1x
    qmp_y = q1y - p1y
    qmpxr = _cross2d(qmp_x, qmp_y, r_x, r_y)

    if abs(rxs) <= eps and abs(qmpxr) <= eps:
        rr = r_x * r_x + r_y * r_y
        if rr <= eps:
            return math.hypot(q1x - p1x, q1y - p1y) <= eps
        t0 = ((q1x - p1x) * r_x + (q1y - p1y) * r_y) / rr
        t1 = ((q2x - p1x) * r_x + (q2y - p1y) * r_y) / rr
        tmin = min(t0, t1)
        tmax = max(t0, t1)
        return tmax >= -eps and tmin <= 1.0 + eps

    if abs(rxs) <= eps:
        return False

    t = _cross2d(qmp_x, qmp_y, s_x, s_y) / rxs
    u = _cross2d(qmp_x, qmp_y, r_x, r_y) / rxs
    return (-eps <= t <= 1.0 + eps) and (-eps <= u <= 1.0 + eps)


def _spline_intersects(outer: np.ndarray, inner: np.ndarray, eps: float = 1e-6) -> bool:
    if outer.shape[0] < 2 or inner.shape[0] < 2:
        return False

    for i in range(outer.shape[0] - 1):
        p1 = (float(outer[i, 0]), float(outer[i, 1]))
        p2 = (float(outer[i + 1, 0]), float(outer[i + 1, 1]))
        for j in range(inner.shape[0] - 1):
            q1 = (float(inner[j, 0]), float(inner[j, 1]))
            q2 = (float(inner[j + 1, 0]), float(inner[j + 1, 1]))
            if _segments_intersect(p1, p2, q1, q2, eps=eps):
                return True
    return False


def _ensure_closed_curve(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    if np.hypot(*(points[0] - points[-1])) <= eps:
        return points
    return np.vstack([points, points[0]])


def _resample_closed_curve(points: np.ndarray, n_samples: int) -> np.ndarray:
    pts = _ensure_closed_curve(points.astype(np.float64, copy=False))
    if pts.shape[0] < 2:
        return pts.astype(np.float32)

    seg = pts[1:] - pts[:-1]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    total = float(seg_len.sum())
    if total <= 1e-9:
        out = np.repeat(pts[:1], max(2, n_samples), axis=0)
        return out.astype(np.float32)

    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    targets = np.linspace(0.0, total, num=max(2, n_samples), endpoint=False)
    out = np.zeros((targets.shape[0], 2), dtype=np.float64)

    for i, d in enumerate(targets):
        k = int(np.searchsorted(cum, d, side="right") - 1)
        k = int(np.clip(k, 0, seg_len.shape[0] - 1))
        d0 = cum[k]
        d1 = cum[k + 1]
        t = 0.0 if d1 <= d0 else (d - d0) / (d1 - d0)
        out[i] = pts[k] + t * (pts[k + 1] - pts[k])

    out = np.vstack([out, out[0]])
    return out.astype(np.float32)


def _ray_segment_intersection(
    p: np.ndarray,
    d: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-9,
) -> tuple[float, float] | None:
    """Solve p + t*d = a + u*(b-a), with ray t>=0 and segment u in [0,1]."""
    s = b - a
    denom = _cross2d(float(d[0]), float(d[1]), float(s[0]), float(s[1]))
    if abs(denom) <= eps:
        return None

    ap = a - p
    t = _cross2d(float(ap[0]), float(ap[1]), float(s[0]), float(s[1])) / denom
    u = _cross2d(float(ap[0]), float(ap[1]), float(d[0]), float(d[1])) / denom
    if t < eps:
        return None
    if u < -eps or u > 1.0 + eps:
        return None
    return float(t), float(u)


def _inner_hits_along_normal(
    outer_pt: np.ndarray,
    normal: np.ndarray,
    inner_closed: np.ndarray,
) -> list[tuple[float, np.ndarray]]:
    """Collect all positive ray hits of the bidirectional normal against inner spline segments."""
    hits: list[tuple[float, np.ndarray]] = []
    for sign in (1.0, -1.0):
        d = normal * sign
        for i in range(inner_closed.shape[0] - 1):
            a = inner_closed[i]
            b = inner_closed[i + 1]
            hit = _ray_segment_intersection(outer_pt, d, a, b)
            if hit is None:
                continue
            t, _u = hit
            pt = outer_pt + t * d
            hits.append((float(t), pt))

    if not hits:
        return []

    hits.sort(key=lambda row: row[0])
    deduped: list[tuple[float, np.ndarray]] = []
    for t, pt in hits:
        if not deduped:
            deduped.append((t, pt))
            continue
        t_prev, pt_prev = deduped[-1]
        if abs(t - t_prev) <= 1e-4 and float(np.hypot(*(pt - pt_prev))) <= 1e-4:
            continue
        deduped.append((t, pt))
    return deduped


def _nearest_inner_hit_along_normal(
    outer_pt: np.ndarray,
    normal: np.ndarray,
    inner_closed: np.ndarray,
) -> np.ndarray | None:
    """Find nearest intersection between a bidirectional normal ray and inner spline segments."""
    hits = _inner_hits_along_normal(outer_pt=outer_pt, normal=normal, inner_closed=inner_closed)
    if not hits:
        return None
    return hits[0][1]


def _ray_hits_with_curve(
    origin: np.ndarray,
    direction: np.ndarray,
    curve_closed: np.ndarray,
) -> list[tuple[float, np.ndarray]]:
    """Collect all forward ray intersections with a closed polyline, sorted by distance."""
    hits: list[tuple[float, np.ndarray]] = []
    for i in range(curve_closed.shape[0] - 1):
        a = curve_closed[i]
        b = curve_closed[i + 1]
        hit = _ray_segment_intersection(origin, direction, a, b)
        if hit is None:
            continue
        t, _u = hit
        if t <= 1e-8:
            continue
        pt = origin + t * direction
        hits.append((float(t), pt))

    if not hits:
        return []

    hits.sort(key=lambda row: row[0])
    deduped: list[tuple[float, np.ndarray]] = []
    for t, pt in hits:
        if not deduped:
            deduped.append((t, pt))
            continue
        t_prev, pt_prev = deduped[-1]
        if abs(t - t_prev) <= 1e-4 and float(np.hypot(*(pt - pt_prev))) <= 1e-4:
            continue
        deduped.append((t, pt))
    return deduped


def _nearest_point_on_closed_polyline(pt: np.ndarray, curve_closed: np.ndarray) -> tuple[np.ndarray, float]:
    """Return nearest point on a closed polyline to pt and the distance."""
    best_d2 = float("inf")
    best_pt = curve_closed[0].astype(np.float64)

    for i in range(curve_closed.shape[0] - 1):
        a = curve_closed[i].astype(np.float64)
        b = curve_closed[i + 1].astype(np.float64)
        ab = b - a
        ab2 = float(ab[0] * ab[0] + ab[1] * ab[1])
        if ab2 <= 1e-12:
            cand = a
        else:
            t = float(np.dot(pt - a, ab) / ab2)
            t = min(1.0, max(0.0, t))
            cand = a + t * ab

        d = pt - cand
        d2 = float(d[0] * d[0] + d[1] * d[1])
        if d2 < best_d2:
            best_d2 = d2
            best_pt = cand

    return best_pt, float(math.sqrt(best_d2))


def _build_centerline_spline(outer: np.ndarray, inner: np.ndarray) -> np.ndarray | None:
    if outer is None or inner is None:
        return None
    if outer.shape[0] < 3 or inner.shape[0] < 3:
        return None

    n_samples = max(180, min(1440, 2 * (max(outer.shape[0], inner.shape[0]) - 1)))
    outer_rs = _resample_closed_curve(outer, n_samples=n_samples).astype(np.float64)
    inner_rs = _resample_closed_curve(inner, n_samples=n_samples).astype(np.float64)

    inner_open = inner_rs[:-1]

    center_pts: list[np.ndarray] = []
    for i_pt in inner_open:
        nearest_outer, dist = _nearest_point_on_closed_polyline(i_pt, outer_rs)
        if dist <= 1e-9:
            center_pts.append(i_pt.copy())
            continue
        direction = nearest_outer - i_pt
        center_pts.append(i_pt + 0.5 * direction)

    print(f"Centerline builder: midpoint samples from inner->outer nearest map={len(center_pts)}")

    if len(center_pts) < max(48, n_samples // 3):
        return None

    center = np.asarray(center_pts, dtype=np.float32)
    center = _ensure_closed_curve(center)
    return center.astype(np.float32)


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
    spline_label: str,
    is_transition_fn,
    wait_for_close: bool = False,
) -> None:
    """Plot a legible diagnostic view for the current failure point and ring candidates."""
    h, w = raw_map.shape
    px = int(xs[curr_x])
    py = int(ys[curr_y])
    radius_px = int(ring_steps * step)

    fig, (ax_full, ax_zoom, ax_ring) = plt.subplots(1, 3, figsize=(17, 6))

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

    ring_vals: list[float] = []
    ring_trans: list[bool] = []
    ring_states_hi: list[bool] = []
    ring_angles_deg: list[float] = []

    for idx, (dy, dx) in enumerate(ring_offsets):
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
        ring_vals.append(val)
        ring_trans.append(trans)
        ring_states_hi.append(bool(hi))
        ring_angles_deg.append(float(idx) * 360.0 / float(max(1, len(ring_offsets))))

    ax_zoom.set_xlim(max(0, px - 2 * radius_px), min(w - 1, px + 2 * radius_px))
    ax_zoom.set_ylim(min(h - 1, py + 2 * radius_px), max(0, py - 2 * radius_px))

    if ring_vals:
        idxs = np.arange(len(ring_vals), dtype=np.int32)
        vals = np.asarray(ring_vals, dtype=np.float32)
        ring_hi_arr = np.asarray(ring_states_hi, dtype=bool)
        trans_arr = np.asarray(ring_trans, dtype=bool)
        ax_ring.plot(idxs, vals, color="white", linewidth=1.2, alpha=0.9)
        ax_ring.scatter(idxs[ring_hi_arr], vals[ring_hi_arr], s=22, c="lime", label="HIGH")
        ax_ring.scatter(idxs[~ring_hi_arr], vals[~ring_hi_arr], s=22, c="red", label="LOW")
        if np.any(trans_arr):
            ax_ring.scatter(
                idxs[trans_arr],
                vals[trans_arr],
                s=70,
                facecolors="none",
                edgecolors="cyan",
                linewidths=1.2,
                label="transition",
            )
        ax_ring.axhline(float(threshold), color="yellow", linestyle="--", linewidth=1.0, label="threshold")
        ax_ring.set_xlabel("ring sample index")
        ax_ring.set_ylabel("raw value")
        ax_ring.set_title("Ring values around failure point")
        ax_ring.legend(loc="best", fontsize=7)
        if ring_angles_deg:
            tick_idx = np.linspace(0, len(ring_angles_deg) - 1, num=min(8, len(ring_angles_deg)), dtype=int)
            ax_ring.set_xticks(tick_idx)
            ax_ring.set_xticklabels([f"{ring_angles_deg[i]:.0f}deg" for i in tick_idx], rotation=25)
    else:
        ax_ring.set_title("Ring values: unavailable")
        ax_ring.text(0.5, 0.5, "No in-bounds ring samples", ha="center", va="center", transform=ax_ring.transAxes)

    fig.colorbar(im0, ax=[ax_full, ax_zoom], fraction=0.03, pad=0.02, label="Raw value")
    fig.suptitle(
        f"Tracer debug at {spline_label} spline point {spline_point_index}\n"
        f"{spline_label} spline point=(y={py}, x={px}), threshold={threshold:.1f}, ring={ring_steps} steps ({radius_px} px)"
    )
    backend = str(plt.get_backend()).lower()
    noninteractive_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    is_noninteractive = backend in noninteractive_backends or backend.startswith("module://matplotlib_inline")
    if is_noninteractive:
        plt.close(fig)
        return
    # Use tight_layout instead of constrained_layout to avoid backend warnings
    # about collapsed axes in some interactive Tk sessions.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if wait_for_close:
        plt.show(block=True)
    else:
        plt.show(block=False)
        plt.pause(0.001)


def _build_step_grid(
    raw_map: np.ndarray,
    step: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys = np.arange(0, raw_map.shape[0], step, dtype=np.int32)
    xs = np.arange(0, raw_map.shape[1], step, dtype=np.int32)
    coarse = raw_map[ys[:, None], xs[None, :]]
    return ys, xs, coarse, coarse >= threshold


def _find_centerline_start(
    coarse: np.ndarray,
    hi: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> tuple[tuple[int, int] | None, list[tuple[int, int, float]]]:
    gh, gw = coarse.shape
    cy = gh // 2
    cx = gw // 2

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
            if not hi[iy, ix]:
                start = (iy, ix)
                start_scan_tail = scan_vals[-5:]
                break
        if start is not None:
            break

    if start is not None:
        print("Initial scan tail (last 5 checked from edge):")
        tail_base = max(0, len(start_scan_tail) - 5)
        for j, (iy, ix, val) in enumerate(start_scan_tail, start=tail_base):
            print(
                f"  edge_check[{j}]: grid(y={iy},x={ix}) "
                f"pixel(y={int(ys[iy])},x={int(xs[ix])}) raw={val:.3f}"
            )
    return start, start_scan_tail


def _collect_ring_transitions(
    coarse: np.ndarray,
    threshold: float,
    center_y: int,
    center_x: int,
    radius_steps: int,
) -> list[dict[str, float | int | str | bool]]:
    gh, gw = coarse.shape
    ordered_candidates: list[tuple[float, int, int]] = []
    r = float(radius_steps)
    for dy in range(-radius_steps - 1, radius_steps + 2):
        for dx in range(-radius_steps - 1, radius_steps + 2):
            if dx == 0 and dy == 0:
                continue
            d = math.hypot(dx, dy)
            if abs(d - r) <= 0.75:
                theta_local = math.atan2(-dy, dx)
                sweep_angle = (2.0 * math.pi - theta_local) % (2.0 * math.pi)
                ordered_candidates.append((sweep_angle, dy, dx))
    ordered_candidates.sort(key=lambda t: t[0])

    in_bounds: list[dict[str, float | int | str | bool]] = []
    states: list[bool] = []
    for ord_idx, (sweep_angle, dy, dx) in enumerate(ordered_candidates):
        ny = center_y + dy
        nx = center_x + dx
        if ny < 1 or nx < 1 or ny >= gh - 1 or nx >= gw - 1:
            continue
        val = float(coarse[ny, nx])
        state = bool(val >= threshold)
        in_bounds.append(
            {
                "ord_idx": ord_idx,
                "angle": sweep_angle,
                "dy": dy,
                "dx": dx,
                "ny": ny,
                "nx": nx,
                "raw": val,
                "state": state,
            }
        )
        states.append(state)

    trans: list[dict[str, float | int | str | bool]] = []
    n = len(in_bounds)
    if n < 2:
        return trans

    # Remove isolated one-bin state flips on the circular ring sequence:
    # LOW,HIGH,LOW -> LOW,LOW,LOW and HIGH,LOW,HIGH -> HIGH,HIGH,HIGH.
    # This suppresses singular spikes before counting transitions.
    warned_isolated: set[int] = set()
    cleaned_states = states.copy()
    for _ in range(max(1, n)):
        changed = False
        next_states = cleaned_states.copy()
        for i in range(n):
            prev_s = cleaned_states[(i - 1) % n]
            curr_s = cleaned_states[i]
            next_s = cleaned_states[(i + 1) % n]
            if prev_s == next_s and curr_s != prev_s:
                if i not in warned_isolated:
                    prev_row = in_bounds[(i - 1) % n]
                    curr_row = in_bounds[i]
                    next_row = in_bounds[(i + 1) % n]
                    print(
                        "Warning: removed isolated ring state spike "
                        f"at grid(y={int(curr_row['ny'])},x={int(curr_row['nx'])}) "
                        f"values(prev,mid,next)=({float(prev_row['raw']):.3f},"
                        f"{float(curr_row['raw']):.3f},{float(next_row['raw']):.3f})"
                    )
                    warned_isolated.add(i)
                next_states[i] = prev_s
                changed = True
        cleaned_states = next_states
        if not changed:
            break

    # Ignore short HIGH islands when they are bracketed by long LOW runs.
    # Rule requested by user: if HIGH run <= 4 and both neighboring LOW runs
    # are > 10, convert that HIGH run to LOW.
    for _ in range(max(1, n)):
        changed = False
        runs = _circular_runs(np.asarray(cleaned_states, dtype=bool))
        if not runs:
            break
        m = len(runs)
        for i, (state, start, run_len) in enumerate(runs):
            if not bool(state):
                continue
            left_state, _left_start, left_len = runs[(i - 1) % m]
            right_state, _right_start, right_len = runs[(i + 1) % m]
            if bool(left_state) or bool(right_state):
                continue
            if int(run_len) <= 4 and int(left_len) > 10 and int(right_len) > 10:
                print(
                    "Warning: removed short HIGH island "
                    f"len={int(run_len)} between LOW runs len={int(left_len)} and len={int(right_len)}"
                )
                for k in range(int(run_len)):
                    cleaned_states[(int(start) + k) % n] = False
                changed = True
        if not changed:
            break

    for i, row in enumerate(in_bounds):
        prev_state = cleaned_states[(i - 1) % n]
        curr_state = cleaned_states[i]
        if curr_state != prev_state:
            row_out = dict(row)
            row_out["ring_idx"] = i
            row_out["prev_state"] = prev_state
            row_out["curr_state"] = curr_state
            row_out["transition_type"] = "LOW->HIGH" if (not prev_state and curr_state) else "HIGH->LOW"
            trans.append(row_out)

    if trans:
        transition_positions = [int(t["ring_idx"]) for t in trans]
        m = len(transition_positions)
        for k, t in enumerate(trans):
            i = transition_positions[k]
            prev_i = transition_positions[(k - 1) % m]
            run_len = (i - prev_i) % n
            if run_len == 0:
                run_len = n
            t["run_len_prev"] = int(run_len)
    return trans


def _find_inner_start_from_outer_point0(
    raw_map: np.ndarray,
    step: int,
    threshold: float,
    outer_start: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
    coarse: np.ndarray,
    start_radius: int = 64,
) -> tuple[int, int] | None:
    oy, ox = outer_start
    print("Inner-start search: begin from outer spline point 0")
    print(
        f"  outer p0 grid(y={oy},x={ox}) pixel(y={int(ys[oy])},x={int(xs[ox])}) "
        f"raw={float(coarse[oy, ox]):.3f}, threshold={threshold:.3f}"
    )

    baseline_r = max(1, int(start_radius))
    base_trans = _collect_ring_transitions(coarse, threshold, oy, ox, baseline_r)
    print(f"  radius={baseline_r}: transitions={len(base_trans)}")
    for i, t in enumerate(base_trans):
        print(
            f"    base[{i}] angle={math.degrees(float(t['angle'])):.2f}deg "
            f"{str(t['transition_type'])} grid(y={int(t['ny'])},x={int(t['nx'])}) "
            f"raw={float(t['raw']):.3f}"
        )

    max_r = max(baseline_r, max(coarse.shape) // 2)
    for radius in range(baseline_r + 1, max_r + 1):
        trans = _collect_ring_transitions(coarse, threshold, oy, ox, radius)
        print(f"  radius={radius}: transitions={len(trans)}")
        for i, t in enumerate(trans):
            run_len = int(t.get("run_len_prev", 0))
            prev_state_txt = "HIGH" if bool(t.get("prev_state", False)) else "LOW"
            print(
                f"    trans[{i}] angle={math.degrees(float(t['angle'])):.2f}deg "
                f"{str(t['transition_type'])} grid(y={int(t['ny'])},x={int(t['nx'])}) "
                f"raw={float(t['raw']):.3f} after {run_len} consecutive {prev_state_txt} samples"
            )

        # Keep center fixed at outer spline point 0 and expand until we have
        # enough transition structure to isolate the inner pair.
        if len(trans) < 4:
            continue

        if len(base_trans) < 2:
            print(
                "    baseline has fewer than 2 transitions; cannot identify which transitions belong to outer boundary."
            )
            continue

        base_angles = [float(t["angle"]) for t in base_trans]

        def circ_dist(a: float, b: float) -> float:
            d = abs(a - b)
            return min(d, 2.0 * math.pi - d)

        scored: list[tuple[float, int]] = []
        for i, t in enumerate(trans):
            a = float(t["angle"])
            dmin = min(circ_dist(a, b) for b in base_angles)
            scored.append((dmin, i))

        scored.sort(key=lambda x: x[0])
        # Two transitions near baseline correspond to outer boundary crossings.
        keep_existing = {idx for _, idx in scored[:2]}
        new_idxs = [i for i in range(len(trans)) if i not in keep_existing]
        if len(new_idxs) < 2:
            print("    unable to isolate two new transitions at this radius; continue.")
            continue

        t1 = trans[new_idxs[0]]
        t2 = trans[new_idxs[1]]

        run1 = int(t1.get("run_len_prev", 0))
        run2 = int(t2.get("run_len_prev", 0))
        if run1 < 12 or run2 < 12:
            print(
                "    rejected new transitions: minimum consecutive run length not met "
                f"(new[1]={run1}, new[2]={run2}, required>=12); continue."
            )
            continue

        print("    identified new transitions:")
        for j, t in enumerate([t1, t2], start=1):
            run_len = int(t.get("run_len_prev", 0))
            prev_state_txt = "HIGH" if bool(t.get("prev_state", False)) else "LOW"
            print(
                f"      new[{j}] angle={math.degrees(float(t['angle'])):.2f}deg "
                f"{str(t['transition_type'])} grid(y={int(t['ny'])},x={int(t['nx'])}) "
                f"pixel(y={int(ys[int(t['ny'])])},x={int(xs[int(t['nx'])])}) "
                f"after {run_len} consecutive {prev_state_txt} samples"
            )

        mid_y = int(round((int(t1["ny"]) + int(t2["ny"])) / 2.0))
        mid_x = int(round((int(t1["nx"]) + int(t2["nx"])) / 2.0))
        mid_y = int(np.clip(mid_y, 1, coarse.shape[0] - 2))
        mid_x = int(np.clip(mid_x, 1, coarse.shape[1] - 2))

        # Prefer a foreground-valued start point so the inner tracer begins on
        # the bracelet side of the boundary when possible.
        if float(coarse[mid_y, mid_x]) < threshold:
            fg_candidate: tuple[int, int] | None = None
            for t in (t1, t2):
                if str(t["transition_type"]) == "LOW->HIGH":
                    ty = int(np.clip(int(t["ny"]), 1, coarse.shape[0] - 2))
                    tx = int(np.clip(int(t["nx"]), 1, coarse.shape[1] - 2))
                    if float(coarse[ty, tx]) >= threshold:
                        fg_candidate = (ty, tx)
                        break
            if fg_candidate is not None:
                mid_y, mid_x = fg_candidate
            else:
                for t in (t1, t2):
                    ty = int(np.clip(int(t["ny"]), 1, coarse.shape[0] - 2))
                    tx = int(np.clip(int(t["nx"]), 1, coarse.shape[1] - 2))
                    if float(coarse[ty, tx]) >= threshold:
                        mid_y, mid_x = ty, tx
                        break
        print(
            f"    inner spline point 0 midpoint grid(y={mid_y},x={mid_x}) "
            f"pixel(y={int(ys[mid_y])},x={int(xs[mid_x])}) raw={float(coarse[mid_y, mid_x]):.3f}"
        )
        return mid_y, mid_x

    print("  inner-start search stop: no expanded radius produced at least 4 transitions.")
    return None


def _trace_outer_spline_step_cells(
    raw_map: np.ndarray,
    step: int,
    threshold: float = 99.0,
    ring_steps: int = 4,
    max_bg_gap: int = 0,
    clockwise: bool = True,
    debug_start_point: int | None = None,
    start_point: tuple[int, int] | None = None,
    spline_label: str = "outer",
    return_partial_on_stop: bool = False,
    verbose: bool = True,
    smooth_output: bool = True,
) -> np.ndarray | None:
    """Trace an outer bracelet spline by stepping along threshold transitions on the step grid."""
    h, w = raw_map.shape
    ys, xs, coarse, hi = _build_step_grid(raw_map, step=step, threshold=threshold)
    if ys.size < 5 or xs.size < 5:
        return None
    gh, gw = coarse.shape

    if start_point is not None:
        sy, sx = start_point
        if sy < 1 or sx < 1 or sy >= gh - 1 or sx >= gw - 1:
            print(
                f"Tracer stop ({spline_label} spline): provided {spline_label} spline point is out of valid tracing bounds; "
                f"grid(y={sy},x={sx})"
            )
            return None
        start = (int(sy), int(sx))
        if verbose:
            print(
                f"Tracer start ({spline_label} spline): using provided {spline_label} spline point "
                f"grid(y={start[0]},x={start[1]}) "
                f"pixel(y={int(ys[start[0]])},x={int(xs[start[1]])}) "
                f"raw={float(coarse[start[0], start[1]]):.3f}"
            )
    else:
        # Start from one end of the centerline and move inward until value drops below threshold.
        start, _ = _find_centerline_start(coarse=coarse, hi=hi, ys=ys, xs=xs)
    trace_output = False

    if start is None:
        print(
            f"Tracer stop ({spline_label} spline): no start {spline_label} spline point found on center lines where raw<threshold; "
            f"threshold={threshold:.3f}"
        )
        return None

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
    if verbose:
        print(
            f"{spline_label.capitalize()} spline point 0: grid(y={curr_y},x={curr_x}) "
            f"pixel(y={int(ys[curr_y])},x={int(xs[curr_x])})"
        )
    prev_dir: tuple[float, float] | None = None
    max_turn_rad = math.radians(90.0)
    stop_reason: str | None = None

    def _find_near_revisit_index(
        y: int,
        x: int,
        points: list[tuple[int, int]],
        exclude_recent: int = 12,
        near_dist: float = 1.6,
    ) -> int | None:
        end = max(0, len(points) - exclude_recent)
        for i in range(end):
            py, px = points[i]
            if math.hypot(y - py, x - px) <= near_dist:
                return i
        return None

    def _partial_trace_pixels() -> np.ndarray | None:
        if len(pts) < 2:
            return None
        return np.array([[float(xs[ix]), float(ys[iy])] for iy, ix in pts], dtype=np.float32)

    while True:
        trace_output = (
            debug_start_point is not None
            and (len(pts) - 1) >= debug_start_point
        )
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
                f"Tracer lost ({spline_label} spline): uniform neighborhood within "
                f"{ring_steps} steps ({ring_steps * step} px); {state}."
            )
            print(
                f"Tracer debug ({spline_label} spline): "
                f"grid(y={curr_y}, x={curr_x}), pixel(y={py}, x={px}), "
                f"raw={raw_here:.3f}, threshold={threshold:.3f}, "
                f"disk_hi={disk_hi}/{disk_n}, points_traced={len(pts)}"
            )
            if return_partial_on_stop:
                return _partial_trace_pixels()
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
                f"Tracer stop ({spline_label} spline): insufficient in-bounds ring samples to validate transitions; "
                f"{spline_label} spline point grid(y={curr_y},x={curr_x})"
            )
            if return_partial_on_stop:
                return _partial_trace_pixels()
            return None

        # Suppress isolated one-bin flips in the circular ring sequence so
        # patterns like LOW,HIGH,LOW or HIGH,LOW,HIGH do not create spurious
        # extra transitions.
        cleaned_ring_states = ordered_ring_states.copy()
        warned_isolated: set[int] = set()
        for _ in range(max(1, ring_n)):
            changed = False
            next_states = cleaned_ring_states.copy()
            for i in range(ring_n):
                prev_s = cleaned_ring_states[(i - 1) % ring_n]
                curr_s = cleaned_ring_states[i]
                next_s = cleaned_ring_states[(i + 1) % ring_n]
                if prev_s == next_s and curr_s != prev_s:
                    if i not in warned_isolated and verbose:
                        prev_meta = ordered_ring_meta[(i - 1) % ring_n]
                        curr_meta = ordered_ring_meta[i]
                        next_meta = ordered_ring_meta[(i + 1) % ring_n]
                        print(
                            "Warning: removed isolated ring state spike "
                            f"at grid(y={curr_meta[4]},x={curr_meta[5]}) "
                            f"values(prev,mid,next)=({prev_meta[6]:.3f},{curr_meta[6]:.3f},{next_meta[6]:.3f})"
                        )
                        warned_isolated.add(i)
                    next_states[i] = prev_s
                    changed = True
            cleaned_ring_states = next_states
            if not changed:
                break

        # Ignore short HIGH islands bracketed by long LOW runs.
        # This mirrors the inner-start ring cleanup rule.
        for _ in range(max(1, ring_n)):
            changed = False
            runs = _circular_runs(np.asarray(cleaned_ring_states, dtype=bool))
            if not runs:
                break
            m = len(runs)
            for i, (state, start, run_len) in enumerate(runs):
                if not bool(state):
                    continue
                left_state, _left_start, left_len = runs[(i - 1) % m]
                right_state, _right_start, right_len = runs[(i + 1) % m]
                if bool(left_state) or bool(right_state):
                    continue
                if int(run_len) <= 4 and int(left_len) > 10 and int(right_len) > 10:
                    if verbose:
                        print(
                            "Warning: removed short HIGH island in tracer "
                            f"at {spline_label} spline point {len(pts) - 1}; "
                            f"len={int(run_len)} between LOW runs len={int(left_len)} and len={int(right_len)}"
                        )
                    for k in range(int(run_len)):
                        cleaned_ring_states[(int(start) + k) % ring_n] = False
                    changed = True
            if not changed:
                break

        # Optionally bridge short LOW runs enclosed by HIGH runs.
        # In this tracer, HIGH corresponds to bracelet/foreground and LOW to background.
        # Bridging a short LOW run should remove two transitions.
        if int(max_bg_gap) > 0:
            before_bridge = np.asarray(cleaned_ring_states, dtype=bool)
            # Pass 1: merge short LOW gaps into HIGH.
            cleaned_ring_states = _bridge_small_background_gaps(
                before_bridge,
                max_bg_gap=int(max_bg_gap),
            )
            # Pass 2 (temporarily disabled): merge short HIGH islands into LOW.
            enable_high_island_merge = False
            if enable_high_island_merge:
                cleaned_ring_states = np.logical_not(
                    _bridge_small_background_gaps(
                        np.logical_not(np.asarray(cleaned_ring_states, dtype=bool)),
                        max_bg_gap=int(max_bg_gap),
                    )
                )
            if verbose and not np.array_equal(before_bridge, cleaned_ring_states):
                b_lh = 0
                b_hl = 0
                a_lh = 0
                a_hl = 0
                for i in range(ring_n):
                    b0 = before_bridge[i]
                    b1 = before_bridge[(i + 1) % ring_n]
                    if (not b0) and b1:
                        b_lh += 1
                    elif b0 and (not b1):
                        b_hl += 1
                    a0 = cleaned_ring_states[i]
                    a1 = cleaned_ring_states[(i + 1) % ring_n]
                    if (not a0) and a1:
                        a_lh += 1
                    elif a0 and (not a1):
                        a_hl += 1
                print(
                    f"Ring gap-bridge ({spline_label} spline point {len(pts) - 1}): "
                    f"LOW->HIGH {b_lh}->{a_lh}, HIGH->LOW {b_hl}->{a_hl}, max_bg_gap={int(max_bg_gap)}"
                )

        for i, row in enumerate(in_bounds_entries):
            prev_state = cleaned_ring_states[(i - 1) % ring_n]
            curr_state = cleaned_ring_states[i]
            trans = bool(curr_state != prev_state)
            row["transition"] = trans

        ring_runs = _circular_runs(np.asarray(cleaned_ring_states, dtype=bool))
        transition_positions = sorted({int(start) for _state, start, _len in ring_runs})
        if transition_positions:
            m = len(transition_positions)
            for k, i in enumerate(transition_positions):
                prev_i = transition_positions[(k - 1) % m]
                run_len = (i - prev_i) % ring_n
                if run_len == 0:
                    run_len = ring_n
                prev_state = bool(cleaned_ring_states[(i - 1) % ring_n])
                curr_state = bool(cleaned_ring_states[i])
                trans_type = "LOW->HIGH" if ((not prev_state) and curr_state) else "HIGH->LOW"
                print(
                    f"Transition scan ({spline_label} spline point {len(pts) - 1}): {trans_type} at ring[{i:02d}] "
                    f"after {run_len} consecutive {'HIGH' if prev_state else 'LOW'} samples."
                )

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
        for i in transition_positions:
            prev_state = bool(cleaned_ring_states[(i - 1) % ring_n])
            curr_state = bool(cleaned_ring_states[i])
            if (not prev_state) and curr_state:
                low_to_high += 1
            elif prev_state and (not curr_state):
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
                f"Tracer stop ({spline_label} spline): ring transition count mismatch "
                f"at grid(y={curr_y},x={curr_x}) pixel(y={curr_py},x={curr_px}); "
                f"LOW->HIGH={low_to_high}, HIGH->LOW={high_to_low}, expected 1 each; "
                f"{median_msg}."
            )
            for ord_idx, sweep_angle, dy, dx, ny, nx, val in ordered_ring_meta:
                if not verbose:
                    break
                state_txt = "HIGH" if val >= threshold else "LOW"
                print(
                    f"  ring[{ord_idx:02d}] {sweep_label} {math.degrees(sweep_angle):6.2f}deg "
                    f"(dy={dy:+d},dx={dx:+d}) grid(y={ny},x={nx}) "
                    f"pixel(y={int(ys[ny])},x={int(xs[nx])}) raw={val:.3f} state={state_txt}"
                )

            # Always show ring diagnostic plot at mismatch so the user can inspect failure.
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
                spline_label=spline_label,
                is_transition_fn=is_transition,
                wait_for_close=True,
            )

            if verbose and len(pts) >= 2:
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
                    spline_label=spline_label,
                    is_transition_fn=is_transition,
                    wait_for_close=False,
                )
            if return_partial_on_stop:
                return _partial_trace_pixels()
            return None

        if best is None:
            print(
                f"Tracer stopped ({spline_label} spline): no valid transition candidate at this {spline_label} spline point."
            )
            cand_in_bounds = cand_total - cand_oob
            cand_transition = cand_in_bounds - cand_not_transition
            if verbose:
                print(
                    f"Tracer debug(no_candidate, {spline_label} spline): "
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
        expected_step_px = float(ring_steps * step)
        actual_step_px = float(
            math.hypot(
                float(ys[next_y]) - float(ys[curr_y]),
                float(xs[next_x]) - float(xs[curr_x]),
            )
        )
        if not math.isclose(actual_step_px, expected_step_px, rel_tol=0.0, abs_tol=0.75):
            raise RuntimeError(
                "Tracer bug: consecutive spline points are not ring-radius apart; "
                f"expected={expected_step_px:.6f}px, actual={actual_step_px:.6f}px, "
                f"from grid(y={curr_y},x={curr_x}) to grid(y={next_y},x={next_x}), "
                f"spline_point_index={len(pts)} ({spline_label} spline)."
            )

        step_dx = float(next_x - curr_x)
        step_dy = float(next_y - curr_y)
        step_norm = math.hypot(step_dx, step_dy)
        if step_norm > 1e-9:
            prev_dir = (step_dx / step_norm, step_dy / step_norm)
        curr_y, curr_x = next_y, next_x
        pts.append((curr_y, curr_x))
        if verbose:
            print(
                f"{spline_label.capitalize()} spline point {len(pts) - 1}: grid(y={curr_y},x={curr_x}) "
                f"pixel(y={int(ys[curr_y])},x={int(xs[curr_x])})"
            )
        if len(pts) >= 3 and verbose:
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
                    f"  turn angle at {spline_label} spline point {len(pts) - 1}: {turn_deg:.2f} deg "
                    f"(segments {len(pts)-3}->{len(pts)-2} and {len(pts)-2}->{len(pts)-1})"
                )
            else:
                print(
                    f"  turn angle at {spline_label} spline point {len(pts) - 1}: unavailable (zero-length segment)"
                )
        elif verbose:
            print(
                f"  turn angle at {spline_label} spline point {len(pts) - 1}: unavailable (need >= 3 points)"
            )

        # Allow a looser near-start closure radius so completed laps are detected
        # even when the path does not pass exactly through point 0.
        if len(pts) > 30 and math.hypot(curr_y - start_y, curr_x - start_x) <= max(3.0, 0.9 * r):
            stop_reason = "closed_loop"
            break

        key = (curr_y, curr_x)
        if key in visited_at:
            first_idx = visited_at[key]
            print(
                f"Tracer stop ({spline_label} spline): detected repeating cycle away from start; "
                f"revisited grid(y={curr_y},x={curr_x}) at {spline_label} spline point {len(pts) - 1}, "
                f"first seen at {spline_label} spline point {first_idx}."
            )
            if verbose:
                print(
                    f"Tracer debug(cycle, {spline_label} spline): "
                    f"pixel(y={int(ys[curr_y])},x={int(xs[curr_x])}), "
                    f"raw={float(coarse[curr_y, curr_x]):.3f}, threshold={threshold:.3f}"
                )
            stop_reason = "cycle_loop"
            break

        near_idx = _find_near_revisit_index(curr_y, curr_x, pts)
        if near_idx is not None and len(pts) > 30:
            print(
                f"Tracer stop ({spline_label} spline): detected near-cycle away from start; "
                f"current grid(y={curr_y},x={curr_x}) at {spline_label} spline point {len(pts) - 1} "
                f"is within 1.6 steps of {spline_label} spline point {near_idx}."
            )
            if verbose:
                print(
                    f"Tracer debug(near-cycle, {spline_label} spline): "
                    f"pixel(y={int(ys[curr_y])},x={int(xs[curr_x])}), "
                    f"raw={float(coarse[curr_y, curr_x]):.3f}, threshold={threshold:.3f}"
                )
            stop_reason = "near_cycle_loop"
            break
        visited_at[key] = len(pts) - 1

    if stop_reason is None:
        if return_partial_on_stop:
            return _partial_trace_pixels()
        return None

    if len(pts) < 16:
        if return_partial_on_stop:
            return _partial_trace_pixels()
        return None

    # Convert from step-grid indices to image pixel coordinates.
    pix = np.array([[float(xs[ix]), float(ys[iy])] for iy, ix in pts], dtype=np.float32)
    if np.hypot(*(pix[0] - pix[-1])) > 1e-6:
        pix = np.vstack([pix, pix[0]])
    if bool(smooth_output):
        pix = _smooth_closed_curve(pix, window=7, passes=2)
    if np.hypot(*(pix[0] - pix[-1])) > 1e-6:
        pix = np.vstack([pix, pix[0]])
    return pix


def find_top_fft_peaks(logmag: np.ndarray, top_k: int = 7, min_distance: int = 6) -> list[dict[str, float]]:
    """Find strongest local maxima in 2D FFT log-magnitude and merge conjugate pairs."""
    h, w = logmag.shape
    cy, cx = h // 2, w // 2

    def canonical_offset(yy: int, xx: int) -> tuple[int, int]:
        dx = int(xx - cx)
        dy = int(cy - yy)
        if dy < 0 or (dy == 0 and dx < 0):
            dx = -dx
            dy = -dy
        return dx, dy

    # 3x3 local-maximum test without additional dependencies.
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

    peaks: list[dict[str, float]] = []
    for yy, xx in selected[:top_k]:
        peak_val = float(logmag[yy, xx])
        dx = float(xx - cx)
        dy = float(cy - yy)
        angle_deg = float(np.degrees(np.arctan2(dy, dx))) if dx != 0.0 or dy != 0.0 else 0.0
        distance = float(np.hypot(dx, dy))
        peaks.append(
            {
                "x": dx,
                "y": dy,
                "distance": distance,
                "angle_deg": angle_deg,
                "peak_val": peak_val,
            }
        )

    return peaks


def load_image_and_luminance(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    rgb = np.asarray(img).astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return rgb, y.astype(np.float32)


def _coerce_spline_xy(spline_xy: np.ndarray | list | tuple) -> np.ndarray | None:
    """Best-effort conversion to an Nx2 float array for spline geometry."""
    arr = np.asarray(spline_xy, dtype=np.float64)
    if arr.size == 0:
        return None

    if arr.ndim == 1:
        if arr.size % 2 != 0:
            return None
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[1] == 2:
            pass
        elif arr.shape[0] == 2:
            arr = arr.T
        else:
            return None
    else:
        return None

    if arr.shape[0] < 3:
        return None
    return arr.astype(np.float32, copy=False)


def _normalize_closed_spline_xy(
    spline_xy: np.ndarray | list | tuple,
    min_open_points: int = 300,
) -> np.ndarray | None:
    """Coerce a spline to closed Nx2 and upsample to a consistent point density."""
    arr = _coerce_spline_xy(spline_xy)
    if arr is None:
        return None

    arr = _ensure_closed_curve(arr.astype(np.float32, copy=False))
    if arr.shape[0] < 4:
        return None

    open_n = arr.shape[0] - 1
    target_open = max(int(min_open_points), int(open_n))
    if target_open != open_n:
        arr = _resample_closed_curve(arr, n_samples=target_open)
    return arr.astype(np.float32, copy=False)


def _catmull_rom_closed_interpolating(
    spline_xy: np.ndarray | list | tuple,
    samples_per_segment: int = 6,
) -> np.ndarray | None:
    """Closed Catmull-Rom interpolation that passes through every control point."""
    arr = _coerce_spline_xy(spline_xy)
    if arr is None:
        return None

    arr = _ensure_closed_curve(arr.astype(np.float32, copy=False))
    if arr.shape[0] < 4:
        return arr

    pts = arr[:-1].astype(np.float64, copy=False)
    n = int(pts.shape[0])
    if n < 3:
        return arr

    sps = max(2, int(samples_per_segment))
    t_values = np.linspace(0.0, 1.0, num=sps, endpoint=False, dtype=np.float64)
    out: list[np.ndarray] = []

    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i % n]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]

        for t in t_values:
            t2 = t * t
            t3 = t2 * t
            # Uniform Catmull-Rom basis, interpolating p1 at t=0 and p2 at t=1.
            q = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(q)

    out_arr = np.asarray(out, dtype=np.float32)
    if out_arr.shape[0] < 2:
        return _ensure_closed_curve(pts.astype(np.float32, copy=False))
    out_arr = np.vstack([out_arr, out_arr[0]])
    return out_arr.astype(np.float32, copy=False)


def _anchor_closed_spline_points(
    spline_xy: np.ndarray,
    first_point_xy: tuple[float, float],
    second_point_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """Rotate/orient a closed spline and pin first/second points to requested anchors."""
    arr = _ensure_closed_curve(np.asarray(spline_xy, dtype=np.float32))
    if arr.shape[0] < 4:
        return arr.astype(np.float32, copy=False)

    open_pts = arr[:-1].astype(np.float64, copy=False)
    n = open_pts.shape[0]

    first = np.asarray(first_point_xy, dtype=np.float64)
    d2 = np.sum((open_pts - first[None, :]) ** 2, axis=1)
    i0 = int(np.argmin(d2))

    forward = np.roll(open_pts, -i0, axis=0)

    reverse_base = open_pts[::-1]
    i0_rev = (n - 1) - i0
    reverse = np.roll(reverse_base, -i0_rev, axis=0)

    chosen = forward
    if second_point_xy is not None and n > 1:
        second = np.asarray(second_point_xy, dtype=np.float64)
        fwd_d = float(np.hypot(*(forward[1] - second)))
        rev_d = float(np.hypot(*(reverse[1] - second)))
        if rev_d < fwd_d:
            chosen = reverse

    chosen = chosen.copy()
    chosen[0] = first
    if second_point_xy is not None and n > 1:
        chosen[1] = np.asarray(second_point_xy, dtype=np.float64)

    closed = np.vstack([chosen, chosen[0]])
    return closed.astype(np.float32, copy=False)


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


def _hsv_background_predicate_mask(rgb: np.ndarray) -> np.ndarray:
    """Background predicate from user-selected HSV bounds (u8 space)."""
    hsv = rgb_to_hsv(np.clip(rgb, 0.0, 1.0).astype(np.float32))
    hsv_u8 = np.clip(np.round(hsv * 255.0), 0, 255).astype(np.uint8)
    h_u8_full = hsv_u8[..., 0]
    s_u8_full = hsv_u8[..., 1]
    v_u8_full = hsv_u8[..., 2]
    return (
        (h_u8_full >= 219) & (h_u8_full <= 242)
        & (s_u8_full >= 100) & (s_u8_full <= 200)
        & (v_u8_full >= 80) & (v_u8_full <= 250)
    )


def _sample_ring_states(
    predicate_mask: np.ndarray,
    center_y: int,
    center_x: int,
    radius_px: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, num=max(16, int(n_samples)), endpoint=False, dtype=np.float64)
    ys = np.rint(center_y + radius_px * np.sin(angles)).astype(np.int32)
    xs = np.rint(center_x + radius_px * np.cos(angles)).astype(np.int32)
    ys = np.clip(ys, 0, predicate_mask.shape[0] - 1)
    xs = np.clip(xs, 0, predicate_mask.shape[1] - 1)
    states = predicate_mask[ys, xs].astype(bool)
    return ys, xs, states


def _find_initial_point_from_center_line(predicate_mask: np.ndarray) -> tuple[int, int, str]:
    """Find initial point where a center-line scan transitions background->foreground.

    Starts from image edges on lines passing through the center and scans inward.
    Background is predicate=True, foreground is predicate=False.
    """
    h, w = predicate_mask.shape
    cy = h // 2
    cx = w // 2

    scans: list[tuple[str, list[tuple[int, int]]]] = [
        ("center-row right->left", [(cy, x) for x in range(w - 1, -1, -1)]),
        ("center-row left->right", [(cy, x) for x in range(0, w)]),
        ("center-col bottom->top", [(y, cx) for y in range(h - 1, -1, -1)]),
        ("center-col top->bottom", [(y, cx) for y in range(0, h)]),
    ]

    for label, line in scans:
        if len(line) < 2:
            continue
        y0, x0 = line[0]
        prev_bg = bool(predicate_mask[y0, x0])
        for y, x in line[1:]:
            curr_bg = bool(predicate_mask[y, x])
            if prev_bg and (not curr_bg):
                return int(y), int(x), label
            prev_bg = curr_bg

    # Fallback if no transition is found.
    return int(cy), int(cx), "fallback-center"


def _bridge_small_background_gaps(states: np.ndarray, max_bg_gap: int) -> np.ndarray:
    """Treat short background runs between foreground runs as foreground."""
    n = int(states.size)
    if n == 0:
        return states

    out = np.asarray(states, dtype=bool).copy()
    max_gap = max(0, int(max_bg_gap))
    for _ in range(max(1, n)):
        changed = False
        runs = _circular_runs(out)
        if not runs:
            break
        m = len(runs)
        for i, (state, start, run_len) in enumerate(runs):
            if state:
                continue
            left_fg = bool(runs[(i - 1) % m][0])
            right_fg = bool(runs[(i + 1) % m][0])
            if left_fg and right_fg and int(run_len) <= max_gap:
                for k in range(int(run_len)):
                    out[(int(start) + k) % n] = True
                changed = True
        if not changed:
            break
    return out


def _circular_runs(states: np.ndarray) -> list[tuple[bool, int, int]]:
    """Return circular runs as (state, start_index, length)."""
    n = int(states.size)
    if n == 0:
        return []

    runs: list[tuple[bool, int, int]] = []
    start = 0
    curr = bool(states[0])
    for i in range(1, n):
        v = bool(states[i])
        if v != curr:
            runs.append((curr, start, i - start))
            start = i
            curr = v
    runs.append((curr, start, n - start))

    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        first = runs[0]
        last = runs[-1]
        runs[0] = (first[0], last[1], first[2] + last[2])
        runs.pop()
    return runs


def _connected_component_from_seed(mask: np.ndarray, seed_y: int, seed_x: int) -> np.ndarray:
    """Return 4-connected component mask grown from seed on a boolean mask."""
    h, w = mask.shape
    if seed_y < 0 or seed_x < 0 or seed_y >= h or seed_x >= w:
        return np.zeros_like(mask, dtype=bool)
    if not bool(mask[seed_y, seed_x]):
        return np.zeros_like(mask, dtype=bool)

    out = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    q.append((int(seed_y), int(seed_x)))
    out[seed_y, seed_x] = True

    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or nx < 0 or ny >= h or nx >= w:
                continue
            if out[ny, nx] or (not bool(mask[ny, nx])):
                continue
            out[ny, nx] = True
            q.append((ny, nx))
    return out


def _extract_outer_contour_xy(component_mask: np.ndarray) -> np.ndarray | None:
    """Extract largest closed contour (x,y) from a boolean component mask."""
    if not np.any(component_mask):
        return None

    fig = plt.figure(figsize=(1, 1))
    try:
        cs = plt.contour(component_mask.astype(np.float32), levels=[0.5])
        segs = cs.allsegs[0] if len(cs.allsegs) > 0 else []
    finally:
        plt.close(fig)

    if not segs:
        return None

    best = max(segs, key=lambda s: s.shape[0])
    xy = np.asarray(best, dtype=np.float32)
    if xy.shape[0] < 3:
        return None
    xy = _ensure_closed_curve(xy)
    xy = _smooth_closed_curve(xy, window=7, passes=2)
    xy = _ensure_closed_curve(xy)
    return xy.astype(np.float32)


def _run_image_only_hsv_mode(
    rgb: np.ndarray,
    image_path: Path,
    spline_out_path: Path,
    ring_radius_px: int,
    ring_samples: int,
    max_bg_gap: int,
    spline_style: str,
) -> dict[str, object]:
    predicate_mask = _hsv_background_predicate_mask(rgb)
    h, w = predicate_mask.shape
    init_y, init_x, init_scan_label = _find_initial_point_from_center_line(predicate_mask)

    ys, xs, states_raw = _sample_ring_states(
        predicate_mask=predicate_mask,
        center_y=init_y,
        center_x=init_x,
        radius_px=max(1, int(ring_radius_px)),
        n_samples=max(16, int(ring_samples)),
    )
    states = _bridge_small_background_gaps(states_raw, max_bg_gap=max(0, int(max_bg_gap)))
    runs = _circular_runs(states)

    bg_runs = [r for r in runs if r[0]]
    fg_runs = [r for r in runs if not r[0]]
    best_bg = max(bg_runs, key=lambda r: r[2]) if bg_runs else None
    best_fg = max(fg_runs, key=lambda r: r[2]) if fg_runs else None

    def _midpoint_from_run(run: tuple[bool, int, int] | None) -> tuple[int, int] | None:
        if run is None:
            return None
        _state, st, ln = run
        mid_idx = (st + ln // 2) % states.size
        return int(ys[mid_idx]), int(xs[mid_idx])

    bg_mid = _midpoint_from_run(best_bg)
    fg_mid = _midpoint_from_run(best_fg)

    transition_pt: tuple[int, int] | None = None
    transition_idx: int | None = None
    if states.size > 1:
        for i in range(states.size):
            prev_i = (i - 1) % states.size
            if bool(states[i]) != bool(states[prev_i]):
                transition_idx = int(i)
                transition_pt = (int(ys[i]), int(xs[i]))
                break

    # Build outer spline using the same circular ring-transition sampling at each point.
    # Bracelet foreground is predicate=False, map it to high values for threshold tracing.
    seed_fg: tuple[int, int] | None = None
    if not bool(predicate_mask[init_y, init_x]):
        seed_fg = (int(init_y), int(init_x))
    elif transition_idx is not None:
        i = int(transition_idx)
        prev_i = (i - 1) % states.size
        fg_i: int | None = None
        if not bool(states[i]):
            fg_i = i
        elif not bool(states[prev_i]):
            fg_i = prev_i
        if fg_i is not None:
            seed_fg = (int(ys[fg_i]), int(xs[fg_i]))
    elif fg_mid is not None:
        seed_fg = (int(fg_mid[0]), int(fg_mid[1]))

    use_catmull = str(spline_style).lower() in {"smooth", "catmull-rom", "catmull_rom", "catmullrom"}
    catmull_samples_per_segment = 6

    outer_control_xy: np.ndarray | None = None
    outer_spline_xy: np.ndarray | None = None
    inner_control_xy: np.ndarray | None = None
    inner_spline_xy: np.ndarray | None = None
    centerline_xy: np.ndarray | None = None
    inner_start: tuple[int, int] | None = None
    inner_intersects_outer = False
    if seed_fg is not None:
        foreground_map = np.where(predicate_mask, 0.0, 100.0).astype(np.float32)
        traced = _trace_outer_spline_step_cells(
            raw_map=foreground_map,
            step=1,
            threshold=50.0,
            ring_steps=max(1, int(ring_radius_px)),
            max_bg_gap=max(0, int(max_bg_gap)),
            clockwise=True,
            start_point=(int(seed_fg[0]), int(seed_fg[1])),
            spline_label="outer",
            return_partial_on_stop=True,
            verbose=False,
            smooth_output=False,
        )
        outer_control_xy = _normalize_closed_spline_xy(traced, min_open_points=0)
        if outer_control_xy is not None:
            first_xy = (float(init_x), float(init_y))
            second_xy = (
                (float(transition_pt[1]), float(transition_pt[0]))
                if transition_pt is not None
                else None
            )
            outer_control_xy = _anchor_closed_spline_points(
                outer_control_xy,
                first_point_xy=first_xy,
                second_point_xy=second_xy,
            )
            outer_spline_xy = outer_control_xy
            if use_catmull:
                smooth_xy = _catmull_rom_closed_interpolating(
                    outer_control_xy,
                    samples_per_segment=catmull_samples_per_segment,
                )
                if smooth_xy is not None:
                    outer_spline_xy = smooth_xy

        ys_step, xs_step, coarse_step, _hi_step = _build_step_grid(
            foreground_map,
            step=1,
            threshold=50.0,
        )

        outer_start_for_inner = (int(seed_fg[0]), int(seed_fg[1]))
        if outer_control_xy is not None and outer_control_xy.shape[0] > 0:
            # Point 0 of outer spline is stored in (x, y) pixel coordinates.
            oy = int(round(float(outer_control_xy[0, 1])))
            ox = int(round(float(outer_control_xy[0, 0])))
            oy = int(np.clip(oy, 1, coarse_step.shape[0] - 2))
            ox = int(np.clip(ox, 1, coarse_step.shape[1] - 2))
            outer_start_for_inner = (oy, ox)

        inner_start = _find_inner_start_from_outer_point0(
            raw_map=foreground_map,
            step=1,
            threshold=50.0,
            outer_start=outer_start_for_inner,
            ys=ys_step,
            xs=xs_step,
            coarse=coarse_step,
            start_radius=max(1, int(ring_radius_px)),
        )
        if inner_start is not None:
            traced_inner = _trace_outer_spline_step_cells(
                raw_map=foreground_map,
                step=1,
                threshold=50.0,
                ring_steps=max(1, int(ring_radius_px)),
                max_bg_gap=max(0, int(max_bg_gap)),
                clockwise=True,
                start_point=(int(inner_start[0]), int(inner_start[1])),
                spline_label="inner",
                return_partial_on_stop=True,
                verbose=False,
                smooth_output=False,
            )
            inner_control_xy = _normalize_closed_spline_xy(traced_inner, min_open_points=0)
            inner_spline_xy = inner_control_xy
            if inner_control_xy is not None and use_catmull:
                smooth_inner = _catmull_rom_closed_interpolating(
                    inner_control_xy,
                    samples_per_segment=catmull_samples_per_segment,
                )
                if smooth_inner is not None:
                    inner_spline_xy = smooth_inner

            if outer_control_xy is not None and inner_control_xy is not None:
                centerline_xy = _build_centerline_spline(outer_control_xy, inner_control_xy)
                inner_intersects_outer = bool(_spline_intersects(outer_spline_xy, inner_spline_xy))

    print("Image-only HSV mode")
    print(f"  image={image_path}")
    print(
        f"  initial_point=(y={init_y}, x={init_x}) from {init_scan_label}, "
        f"ring_radius_px={ring_radius_px}, samples={ring_samples}, spline_style={'catmull-rom' if use_catmull else 'polyline'}"
    )
    print(f"  background_true_pixels={int(np.count_nonzero(predicate_mask))}/{predicate_mask.size}")
    if best_bg is not None:
        print(f"  largest background run length={best_bg[2]} samples")
    if best_fg is not None:
        print(f"  largest bracelet run length={best_fg[2]} samples")
    if outer_spline_xy is not None:
        print(f"  outer spline points={outer_spline_xy.shape[0]}")
    else:
        print("  outer spline: unavailable (tracer did not produce a valid loop)")
    if inner_start is not None:
        print(
            "  inner spline start "
            f"grid(y={int(inner_start[0])}, x={int(inner_start[1])})"
        )
    else:
        print("  inner spline start: unavailable")
    if inner_spline_xy is not None:
        print(f"  inner spline points={inner_spline_xy.shape[0]}")
    else:
        print("  inner spline: unavailable")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title("Original image only: HSV ring classification")

    bg_pts = states
    fg_pts = ~states
    ax.scatter(xs[bg_pts], ys[bg_pts], s=8, c="lime", alpha=0.75, label="background (predicate true)")
    ax.scatter(xs[fg_pts], ys[fg_pts], s=8, c="red", alpha=0.75, label="bracelet (predicate false)")
    ax.scatter([init_x], [init_y], s=55, c="cyan", edgecolors="black", linewidths=0.8, label="initial point")
    if transition_pt is not None:
        ax.scatter(
            [transition_pt[1]],
            [transition_pt[0]],
            s=70,
            c="blue",
            edgecolors="white",
            linewidths=0.9,
            label="selected transition (2nd spline point)",
        )
    if bg_mid is not None:
        ax.scatter([bg_mid[1]], [bg_mid[0]], s=80, c="yellow", edgecolors="black", linewidths=0.8, label="largest background")
    if fg_mid is not None:
        ax.scatter([fg_mid[1]], [fg_mid[0]], s=80, c="white", edgecolors="black", linewidths=0.8, label="largest bracelet")
    if outer_spline_xy is not None:
        ax.plot(outer_spline_xy[:, 0], outer_spline_xy[:, 1], color="white", linewidth=2.0, alpha=0.95, label="outer spline")
        spline_pts = np.asarray(outer_control_xy if outer_control_xy is not None else outer_spline_xy, dtype=np.float64)
        if spline_pts.shape[0] >= 2 and float(np.hypot(*(spline_pts[0] - spline_pts[-1]))) <= 1e-6:
            spline_pts = spline_pts[:-1]
        if spline_pts.shape[0] > 0:
            spline_scatter = ax.scatter(
                spline_pts[:, 0],
                spline_pts[:, 1],
                marker="o",
                s=28,
                facecolors="none",
                edgecolors="white",
                linewidths=0.9,
                alpha=0.95,
                label="spline points",
            )
            for i in range(0, spline_pts.shape[0], 10):
                ax.text(
                    float(spline_pts[i, 0]) + 2.0,
                    float(spline_pts[i, 1]) + 2.0,
                    str(i),
                    color="white",
                    fontsize=7,
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 0.2, "edgecolor": "none"},
                )

            hover_annot = ax.annotate(
                "",
                xy=(0.0, 0.0),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.75, "pad": 0.3, "edgecolor": "white"},
            )
            hover_annot.set_visible(False)

            def _on_spline_hover(event) -> None:
                if event.inaxes != ax:
                    if hover_annot.get_visible():
                        hover_annot.set_visible(False)
                        fig.canvas.draw_idle()
                    return
                contains, info = spline_scatter.contains(event)
                if (not contains) or ("ind" not in info) or (len(info["ind"]) == 0):
                    if hover_annot.get_visible():
                        hover_annot.set_visible(False)
                        fig.canvas.draw_idle()
                    return
                idx = int(info["ind"][0])
                px = float(spline_pts[idx, 0])
                py = float(spline_pts[idx, 1])
                hover_annot.xy = (px, py)
                hover_annot.set_text(f"point {idx}\\n(x={px:.1f}, y={py:.1f})")
                hover_annot.set_visible(True)
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", _on_spline_hover)
    if inner_spline_xy is not None:
        ax.plot(inner_spline_xy[:, 0], inner_spline_xy[:, 1], color="cyan", linewidth=1.6, alpha=0.95, label="inner spline")
        inner_pts = np.asarray(inner_control_xy if inner_control_xy is not None else inner_spline_xy, dtype=np.float64)
        if inner_pts.shape[0] >= 2 and float(np.hypot(*(inner_pts[0] - inner_pts[-1]))) <= 1e-6:
            inner_pts = inner_pts[:-1]
        if inner_pts.shape[0] > 0:
            ax.scatter(
                inner_pts[:, 0],
                inner_pts[:, 1],
                marker="o",
                s=28,
                facecolors="none",
                edgecolors="cyan",
                linewidths=0.9,
                alpha=0.95,
                label="inner spline points",
            )
    if centerline_xy is not None:
        ax.plot(centerline_xy[:, 0], centerline_xy[:, 1], color="yellow", linewidth=1.4, alpha=0.9, label="centerline")
        center_pts = np.asarray(centerline_xy, dtype=np.float64)
        if center_pts.shape[0] >= 2 and float(np.hypot(*(center_pts[0] - center_pts[-1]))) <= 1e-6:
            center_pts = center_pts[:-1]
        if center_pts.shape[0] > 0:
            ax.scatter(
                center_pts[:, 0],
                center_pts[:, 1],
                marker="o",
                s=28,
                facecolors="none",
                edgecolors="yellow",
                linewidths=0.9,
                alpha=0.95,
                label="centerline points",
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    backend = str(plt.get_backend()).lower()
    noninteractive_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    is_noninteractive = backend in noninteractive_backends or backend.startswith("module://matplotlib_inline")
    if is_noninteractive:
        plt.close(fig)
    else:
        plt.show()

    output = {
        "version": 1,
        "mode": "image_only_hsv",
        "image_path": str(image_path),
        "ring_radius_px": int(ring_radius_px),
        "ring_samples": int(ring_samples),
        "max_bg_gap": int(max_bg_gap),
        "spline_style": "catmull-rom" if use_catmull else "polyline",
        "initial_point": [int(init_x), int(init_y)],
        "initial_point_source": str(init_scan_label),
        "largest_background_run_samples": int(best_bg[2]) if best_bg is not None else 0,
        "largest_bracelet_run_samples": int(best_fg[2]) if best_fg is not None else 0,
        "largest_background_midpoint": [int(bg_mid[1]), int(bg_mid[0])] if bg_mid is not None else None,
        "largest_bracelet_midpoint": [int(fg_mid[1]), int(fg_mid[0])] if fg_mid is not None else None,
        "outer_spline": (
            [[float(p[0]), float(p[1])] for p in np.asarray(outer_spline_xy, dtype=np.float64)]
            if outer_spline_xy is not None
            else None
        ),
        "outer_spline_control": (
            [[float(p[0]), float(p[1])] for p in np.asarray(outer_control_xy, dtype=np.float64)]
            if outer_control_xy is not None
            else None
        ),
        "inner_spline": (
            [[float(p[0]), float(p[1])] for p in np.asarray(inner_spline_xy, dtype=np.float64)]
            if inner_spline_xy is not None
            else None
        ),
        "inner_spline_control": (
            [[float(p[0]), float(p[1])] for p in np.asarray(inner_control_xy, dtype=np.float64)]
            if inner_control_xy is not None
            else None
        ),
        "centerline_spline": (
            [[float(p[0]), float(p[1])] for p in np.asarray(centerline_xy, dtype=np.float64)]
            if centerline_xy is not None
            else None
        ),
        "inner_intersects_outer": bool(inner_intersects_outer),
    }
    spline_out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote spline file: {spline_out_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a saved map+metadata, compute splines, save spline JSON, and render overlays."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Either: <image_path> to auto-resolve map/metadata names, "
            "or: <map.npy> <metadata.json>"
        ),
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional override path for the original image (defaults to metadata image_path).",
    )
    parser.add_argument(
        "--spline-out",
        default=None,
        help="Path for writing computed spline geometry (.json). Default: <image_name>_splines.json",
    )
    parser.add_argument(
        "--trace-threshold",
        type=float,
        default=None,
        help="Override trace threshold. Default uses metadata value.",
    )
    parser.add_argument(
        "--trace-debug-start-outer",
        type=int,
        default=None,
        help="Optional debug start index for outer tracer.",
    )
    parser.add_argument(
        "--trace-debug-start-inner",
        type=int,
        default=None,
        help="Optional debug start index for inner tracer.",
    )
    parser.add_argument(
        "--image-only-hsv",
        action="store_true",
        help=(
            "Operate on original image only using HSV predicate instead of FFT-derived map. "
            "Samples a ring around the initial point and finds contiguous background/bracelet regions."
        ),
    )
    parser.add_argument(
        "--ring-radius-px",
        type=int,
        default=64,
        help="Ring radius in pixels for image-only HSV mode (default: 64).",
    )
    parser.add_argument(
        "--ring-samples",
        type=int,
        default=720,
        help="Number of evenly spaced samples on the ring in image-only HSV mode (default: 720).",
    )
    parser.add_argument(
        "--max-bg-gap",
        type=int,
        default=14,
        help=(
            "Maximum short background run length to reclassify as foreground when it lies "
            "between foreground runs (image-only HSV mode, default: 14)."
        ),
    )
    parser.add_argument(
        "--image-only-spline-style",
        choices=["smooth", "polyline", "catmull-rom"],
        default="smooth",
        help="Spline style for image-only mode: smooth interpolating or polyline (default).",
    )
    args = parser.parse_args()

    if args.image_only_hsv:
        if len(args.inputs) != 1:
            parser.error("--image-only-hsv requires exactly one positional input: <image_path>.")
        image_path = Path(args.inputs[0]).expanduser().resolve()
        image_stem = image_path.stem
        spline_out_path = (
            Path(args.spline_out).expanduser().resolve()
            if args.spline_out
            else (Path.cwd() / f"{image_stem}_splines.json").resolve()
        )
        rgb, _ = load_image_and_luminance(str(image_path))
        _run_image_only_hsv_mode(
            rgb=rgb,
            image_path=image_path,
            spline_out_path=spline_out_path,
            ring_radius_px=max(1, int(args.ring_radius_px)),
            ring_samples=max(16, int(args.ring_samples)),
            max_bg_gap=max(0, int(args.max_bg_gap)),
            spline_style=str(args.image_only_spline_style),
        )
        return

    if len(args.inputs) == 1:
        image_input_path = Path(args.inputs[0]).expanduser().resolve()
        image_stem = image_input_path.stem
        map_path = (Path.cwd() / f"{image_stem}_map.npy").resolve()
        meta_path = (Path.cwd() / f"{image_stem}_map_metadata.json").resolve()
    elif len(args.inputs) == 2:
        map_path = Path(args.inputs[0]).expanduser().resolve()
        meta_path = Path(args.inputs[1]).expanduser().resolve()
    else:
        parser.error("Provide either 1 argument (<image_path>) or 2 arguments (<map.npy> <metadata.json>).")

    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    image_path = Path(args.image).expanduser().resolve() if args.image else Path(meta["image_path"]).expanduser().resolve()
    image_stem = image_path.stem
    spline_out_path = (
        Path(args.spline_out).expanduser().resolve()
        if args.spline_out
        else (Path.cwd() / f"{image_stem}_splines.json").resolve()
    )
    rgb, _ = load_image_and_luminance(str(image_path))
    out = np.load(map_path).astype(np.float32)

    if rgb.shape[:2] != out.shape:
        raise RuntimeError(
            f"Shape mismatch: image shape={rgb.shape[:2]} vs map shape={out.shape}. "
            "Use --image to point to the matching source image."
        )

    hp_percent = float(meta.get("highpass_percent", 8.0))
    metric = str(meta.get("metric", "hp_removed"))
    step = int(meta.get("step", 1))
    window_size = int(meta.get("window_size", 0))
    display_scale = str(meta.get("display_scale", "near100"))
    near100_alpha = float(meta.get("near100_alpha", 0.25))
    processed = int(meta.get("processed", int(out.size)))
    total = int(meta.get("total", int(out.size)))

    trace_threshold = (
        float(np.clip(args.trace_threshold, 0.0, 100.0))
        if args.trace_threshold is not None
        else float(np.clip(float(meta.get("trace_threshold", 99.0)), 0.0, 100.0))
    )
    trace_debug_start_outer: int | None = (
        int(args.trace_debug_start_outer)
        if args.trace_debug_start_outer is not None and int(args.trace_debug_start_outer) >= 0
        else meta.get("trace_debug_start_outer")
    )
    trace_debug_start_inner: int | None = (
        int(args.trace_debug_start_inner)
        if args.trace_debug_start_inner is not None and int(args.trace_debug_start_inner) >= 0
        else meta.get("trace_debug_start_inner")
    )

    spline_data = _show_map(
        rgb,
        out,
        hp_percent,
        metric=metric,
        processed=processed,
        total=total,
        step=step,
        window_size=window_size,
        display_scale=display_scale,
        near100_alpha=near100_alpha,
        trace_threshold=trace_threshold,
        trace_debug_start_outer=trace_debug_start_outer,
        trace_debug_start_inner=trace_debug_start_inner,
    )

    output = {
        "version": 1,
        "image_path": str(image_path),
        "map_path": str(map_path),
        "metadata_path": str(meta_path),
        "trace_threshold": float(trace_threshold),
        "outer_spline": spline_data.get("outer"),
        "inner_spline": spline_data.get("inner"),
        "centerline_spline": spline_data.get("centerline"),
        "inner_intersects_outer": bool(spline_data.get("intersects", False)),
    }
    spline_out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote spline file: {spline_out_path}")


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
    trace_threshold: float,
    trace_debug_start_outer: int | None,
    trace_debug_start_inner: int | None,
) -> dict[str, object]:
    deferred_error: str | None = None
    t0 = time.perf_counter()
    # Keep a pristine copy for HSV counting so plotted spline overlays are never sampled.
    rgb_source = np.asarray(rgb, dtype=np.float32).copy()

    def _dbg(msg: str) -> None:
        dt = time.perf_counter() - t0
        print(f"[_show_map +{dt:8.3f}s] {msg}", flush=True)

    _dbg(
        "enter: "
        f"metric={metric}, display_scale={display_scale}, out_shape={tuple(out.shape)}, rgb_shape={tuple(rgb_source.shape)}"
    )

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
    )

    ax_src.imshow(rgb_source, interpolation="nearest")
    ax_src.set_title("Original image")
    h_img, w_img = out.shape
    fixed_xlim = (-0.5, float(w_img) - 0.5)
    fixed_ylim = (float(h_img) - 0.5, -0.5)
    ax_src.set_xlim(*fixed_xlim)
    ax_src.set_ylim(*fixed_ylim)
    # Keep shared axes pinned to image bounds so line plots cannot autoscale
    # the view and wash out the map panel.
    ax_src.set_autoscale_on(False)
    ax_map.set_autoscale_on(False)

    traced_outer = None
    traced_inner = None
    centerline = None
    intersects = False

    if metric == "hp_removed":
        _dbg("begin hp_removed tracing block")
        trace_step = max(1, int(step))
        ys_step, xs_step, coarse_step, hi_step = _build_step_grid(
            out,
            step=trace_step,
            threshold=trace_threshold,
        )
        _dbg(f"step-grid built: coarse_shape={tuple(coarse_step.shape)}, trace_step={trace_step}")
        outer_start, _ = _find_centerline_start(
            coarse=coarse_step,
            hi=hi_step,
            ys=ys_step,
            xs=xs_step,
        )
        _dbg(f"outer_start={outer_start}")

        if outer_start is not None:
            _dbg("calling outer tracer")
            traced_outer = _trace_outer_spline_step_cells(
                out,
                step=trace_step,
                threshold=trace_threshold,
                ring_steps=4,
                clockwise=True,
                debug_start_point=trace_debug_start_outer,
                start_point=outer_start,
                spline_label="outer",
                return_partial_on_stop=True,
            )
            _dbg(f"outer tracer returned shape={np.asarray(traced_outer).shape}")
        traced_outer = _normalize_closed_spline_xy(traced_outer, min_open_points=300)
        if traced_outer is not None:
            print(f"Outer spline points: {traced_outer.shape[0]}")
            for ax in (ax_src, ax_map):
                ax.plot(traced_outer[:, 0], traced_outer[:, 1], color="white", linewidth=2.0, alpha=0.95)

        if outer_start is not None:
            _dbg("searching inner_start from outer point 0")
            inner_start = _find_inner_start_from_outer_point0(
                raw_map=out,
                step=trace_step,
                threshold=trace_threshold,
                outer_start=outer_start,
                ys=ys_step,
                xs=xs_step,
                coarse=coarse_step,
            )
            _dbg(f"inner_start={inner_start}")
            if inner_start is not None:
                _dbg("calling inner tracer")
                traced_inner = _trace_outer_spline_step_cells(
                    out,
                    step=trace_step,
                    threshold=trace_threshold,
                    ring_steps=4,
                    clockwise=True,
                    debug_start_point=trace_debug_start_inner,
                    start_point=inner_start,
                    spline_label="inner",
                    return_partial_on_stop=True,
                )
                _dbg(f"inner tracer returned shape={np.asarray(traced_inner).shape}")
                traced_inner = _normalize_closed_spline_xy(traced_inner, min_open_points=300)
                if traced_inner is not None:
                    print(f"Inner spline points: {traced_inner.shape[0]}")
                    for ax in (ax_src, ax_map):
                        ax.plot(traced_inner[:, 0], traced_inner[:, 1], color="cyan", linewidth=1.8, alpha=0.95)
                    centerline = _build_centerline_spline(traced_outer, traced_inner)
                    if centerline is not None:
                        print(
                            "Centerline spline: built midpoint curve from outer/inner splines; "
                            f"samples={centerline.shape[0]}"
                        )
                        for ax in (ax_src, ax_map):
                            ax.plot(
                                centerline[:, 0],
                                centerline[:, 1],
                                color="yellow",
                                linewidth=1.6,
                                alpha=0.95,
                            )

                    if traced_outer is not None and _spline_intersects(traced_outer, traced_inner):
                        intersects = True
                        deferred_error = (
                            "Error: inner spline intersects outer spline. "
                            "Inner/outer spline intersection is not allowed."
                        )
                        print(deferred_error)
                _dbg("end hp_removed tracing block")

    disp64 = np.asarray(disp, dtype=np.float64)
    finite_mask = np.isfinite(disp64)
    finite_count = int(np.count_nonzero(finite_mask))
    total_count = int(disp64.size)
    if finite_count == 0:
        raise RuntimeError("disp validation failed: no finite values available for map display")

    disp_finite = disp64[finite_mask]
    dmin = float(np.min(disp_finite))
    dmax = float(np.max(disp_finite))
    dmean = float(np.mean(disp_finite))
    p01, p50, p99 = np.percentile(disp_finite, [1.0, 50.0, 99.0]).astype(float)
    in_vrange = float(np.mean((disp_finite >= float(vmin)) & (disp_finite <= float(vmax))))
    print(
        "disp validation: "
        f"finite={finite_count}/{total_count}, "
        f"min={dmin:.6f}, p01={p01:.6f}, p50={p50:.6f}, p99={p99:.6f}, max={dmax:.6f}, "
        f"mean={dmean:.6f}, in_vrange={100.0 * in_vrange:.2f}%"
    )

    _dbg("entering ax_map.imshow")
    im = ax_map.imshow(disp, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
    _dbg("ax_map.imshow returned")
    ax_map.set_title("FFT-derived map")
    ax_map.set_xlim(*fixed_xlim)
    ax_map.set_ylim(*fixed_ylim)

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
    # Explicit cursor-data formatting helps newer Matplotlib toolbars include
    # value readouts on image artists.
    ax_map.format_cursor_data = lambda data: f"{float(data):.3f}" if np.isscalar(data) else str(data)

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
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    # Avoid axis("off") because some backends drop rich cursor readouts on hidden axes.
    for ax in (ax_src, ax_map):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Re-assert after axis styling to avoid backend-specific formatter resets.
    ax_map.format_coord = _fmt_xy_values
    ax_src.format_coord = _fmt_xy_values
    _dbg("calling final plt.show (blocking until figure closes)")
    plt.show()
    _dbg("returned from final plt.show")
    if deferred_error is not None:
        _dbg("raising deferred intersection RuntimeError")
        raise RuntimeError(deferred_error)

    def _curve_to_json(curve: np.ndarray | None) -> list[list[float]] | None:
        if curve is None:
            return None
        return [[float(p[0]), float(p[1])] for p in np.asarray(curve, dtype=np.float64)]

    return {
        "outer": _curve_to_json(traced_outer),
        "inner": _curve_to_json(traced_inner),
        "centerline": _curve_to_json(centerline),
        "intersects": bool(intersects),
    }


if __name__ == "__main__":
    main()

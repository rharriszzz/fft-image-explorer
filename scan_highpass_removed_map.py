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
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.path import Path
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
    center = _smooth_closed_curve(center, window=9, passes=2)
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
        f"Tracer debug at {spline_label} spline point {spline_point_index}\n"
        f"{spline_label} spline point=(y={py}, x={px}), threshold={threshold:.1f}, ring={ring_steps} steps ({radius_px} px)"
    )
    # Render this diagnostic window without blocking; final blocking show occurs in _show_map.
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

    for i, row in enumerate(in_bounds):
        prev_state = cleaned_states[(i - 1) % n]
        curr_state = cleaned_states[i]
        if curr_state != prev_state:
            row_out = dict(row)
            row_out["prev_state"] = prev_state
            row_out["curr_state"] = curr_state
            row_out["transition_type"] = "LOW->HIGH" if (not prev_state and curr_state) else "HIGH->LOW"
            trans.append(row_out)
    return trans


def _find_inner_start_from_outer_point0(
    raw_map: np.ndarray,
    step: int,
    threshold: float,
    outer_start: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
    coarse: np.ndarray,
) -> tuple[int, int] | None:
    oy, ox = outer_start
    print("Inner-start search: begin from outer spline point 0")
    print(
        f"  outer p0 grid(y={oy},x={ox}) pixel(y={int(ys[oy])},x={int(xs[ox])}) "
        f"raw={float(coarse[oy, ox]):.3f}, threshold={threshold:.3f}"
    )

    baseline_r = 4
    base_trans = _collect_ring_transitions(coarse, threshold, oy, ox, baseline_r)
    print(f"  radius={baseline_r}: transitions={len(base_trans)}")
    for i, t in enumerate(base_trans):
        print(
            f"    base[{i}] angle={math.degrees(float(t['angle'])):.2f}deg "
            f"{str(t['transition_type'])} grid(y={int(t['ny'])},x={int(t['nx'])}) "
            f"raw={float(t['raw']):.3f}"
        )

    if len(base_trans) < 2:
        print("  inner-start search stop: baseline ring does not contain 2 transitions.")
        return None

    max_r = min(max(coarse.shape) // 2, 60)
    for radius in range(baseline_r + 1, max_r + 1):
        trans = _collect_ring_transitions(coarse, threshold, oy, ox, radius)
        print(f"  radius={radius}: transitions={len(trans)}")
        for i, t in enumerate(trans):
            print(
                f"    trans[{i}] angle={math.degrees(float(t['angle'])):.2f}deg "
                f"{str(t['transition_type'])} grid(y={int(t['ny'])},x={int(t['nx'])}) "
                f"raw={float(t['raw']):.3f}"
            )

        if len(trans) < 4:
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
        keep_existing = {idx for _, idx in scored[:2]}
        new_idxs = [i for i in range(len(trans)) if i not in keep_existing]
        if len(new_idxs) < 2:
            print("    unable to isolate two new transitions at this radius; continue.")
            continue

        t1 = trans[new_idxs[0]]
        t2 = trans[new_idxs[1]]
        print("    identified new transitions:")
        for j, t in enumerate([t1, t2], start=1):
            print(
                f"      new[{j}] angle={math.degrees(float(t['angle'])):.2f}deg "
                f"{str(t['transition_type'])} grid(y={int(t['ny'])},x={int(t['nx'])}) "
                f"pixel(y={int(ys[int(t['ny'])])},x={int(xs[int(t['nx'])])})"
            )

        mid_y = int(round((int(t1["ny"]) + int(t2["ny"])) / 2.0))
        mid_x = int(round((int(t1["nx"]) + int(t2["nx"])) / 2.0))
        mid_y = int(np.clip(mid_y, 1, coarse.shape[0] - 2))
        mid_x = int(np.clip(mid_x, 1, coarse.shape[1] - 2))
        print(
            f"    inner spline point 0 midpoint grid(y={mid_y},x={mid_x}) "
            f"pixel(y={int(ys[mid_y])},x={int(xs[mid_x])}) raw={float(coarse[mid_y, mid_x]):.3f}"
        )
        return mid_y, mid_x

    print("  inner-start search stop: no radius produced >=4 transitions.")
    return None


def _trace_outer_spline_step_cells(
    raw_map: np.ndarray,
    step: int,
    threshold: float = 99.0,
    ring_steps: int = 4,
    clockwise: bool = True,
    debug_start_point: int | None = None,
    start_point: tuple[int, int] | None = None,
    spline_label: str = "outer",
    return_partial_on_stop: bool = False,
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
                    if i not in warned_isolated:
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

        for i, row in enumerate(in_bounds_entries):
            prev_state = cleaned_ring_states[(i - 1) % ring_n]
            curr_state = cleaned_ring_states[i]
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
            a = cleaned_ring_states[i]
            b = cleaned_ring_states[(i + 1) % ring_n]
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
                f"Tracer stop ({spline_label} spline): ring transition count mismatch "
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
                spline_label=spline_label,
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
                    spline_label=spline_label,
                    is_transition_fn=is_transition,
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
        step_dx = float(next_x - curr_x)
        step_dy = float(next_y - curr_y)
        step_norm = math.hypot(step_dx, step_dy)
        if step_norm > 1e-9:
            prev_dir = (step_dx / step_norm, step_dy / step_norm)
        curr_y, curr_x = next_y, next_x
        pts.append((curr_y, curr_x))
        print(
            f"{spline_label.capitalize()} spline point {len(pts) - 1}: grid(y={curr_y},x={curr_x}) "
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
                    f"  turn angle at {spline_label} spline point {len(pts) - 1}: {turn_deg:.2f} deg "
                    f"(segments {len(pts)-3}->{len(pts)-2} and {len(pts)-2}->{len(pts)-1})"
                )
            else:
                print(
                    f"  turn angle at {spline_label} spline point {len(pts) - 1}: unavailable (zero-length segment)"
                )
        else:
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


def _mask_inside_closed_spline(shape: tuple[int, int], spline_xy: np.ndarray) -> np.ndarray:
    """Return boolean mask for pixels whose centers are inside a closed spline polygon."""
    h, w = shape
    poly = np.asarray(spline_xy, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        return np.zeros((h, w), dtype=bool)
    if np.hypot(*(poly[0] - poly[-1])) > 1e-6:
        poly = np.vstack([poly, poly[0]])

    path = Path(poly)
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = path.contains_points(pts, radius=1e-9)
    return inside.reshape(h, w)


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


def _packed_hsv_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] image to packed 24-bit HSV (8 bits per channel)."""
    hsv = rgb_to_hsv(np.clip(rgb, 0.0, 1.0).astype(np.float32))
    hsv_u8 = np.clip(np.round(hsv * 255.0), 0, 255).astype(np.uint8)
    h = hsv_u8[..., 0].astype(np.uint32)
    s = hsv_u8[..., 1].astype(np.uint32)
    v = hsv_u8[..., 2].astype(np.uint32)
    return (h << 16) | (s << 8) | v


def _top_exact_hsv_counts(packed_hsv: np.ndarray, mask: np.ndarray, top_k: int = 20) -> tuple[int, list[tuple[int, int, int, int]]]:
    vals = packed_hsv[mask]
    if vals.size == 0:
        return 0, []
    uniq, cnt = np.unique(vals, return_counts=True)
    order = np.argsort(cnt)[::-1]
    top = []
    for idx in order[:top_k]:
        p = int(uniq[idx])
        c = int(cnt[idx])
        h = (p >> 16) & 0xFF
        s = (p >> 8) & 0xFF
        v = p & 0xFF
        top.append((h, s, v, c))
    return int(uniq.size), top


def _top_hsv_volume_counts(
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

    # Full H/S bins for near-black and near-white V ranges.
    frac = float(np.clip(extreme_v_fraction, 0.0, 0.49))
    edge_span = int(math.ceil(frac * 256.0))
    if edge_span <= 0:
        # Explicitly disable extreme-V bins.
        low_v0, low_v1 = 0, -1
        high_v0, high_v1 = 256, 255
    else:
        low_v0, low_v1 = 0, min(255, edge_span - 1)
        # Use an inclusive high-side edge that starts at 255-edge_span so edgeV=0.05
        # yields 242..255 while low side is 0..12.
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

    # Partition the remaining V range into the requested number of middle groups.
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

    chosen_meta: dict[str, float | int] = {
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

    return chosen_rows, chosen_meta


def _plot_hsv_region_count_summary(
    rgb_source: np.ndarray,
    packed_hsv: np.ndarray,
    region_union_outside_plus_inner: np.ndarray,
    region_between_inner_outer: np.ndarray,
    extreme_v_fraction: float = 0.10,
    h_bins: int = 24,
    s_bins: int = 3,
    v_bins: int = 4,
) -> None:
    d1, _top1 = _top_exact_hsv_counts(packed_hsv, region_union_outside_plus_inner, top_k=20)
    d2, _top2 = _top_exact_hsv_counts(packed_hsv, region_between_inner_outer, top_k=20)
    vol1, meta1 = _top_hsv_volume_counts(
        packed_hsv,
        region_union_outside_plus_inner,
        coverage_target=0.995,
        max_bars=144,
        h_bins=h_bins,
        s_bins=s_bins,
        v_bins=v_bins,
        extreme_v_fraction=extreme_v_fraction,
    )
    vol2, meta2 = _top_hsv_volume_counts(
        packed_hsv,
        region_between_inner_outer,
        coverage_target=0.995,
        max_bars=144,
        h_bins=h_bins,
        s_bins=s_bins,
        v_bins=v_bins,
        extreme_v_fraction=extreme_v_fraction,
    )

    print(
        "HSV distinct values: "
        f"regionA(outside outer U inside inner)={d1}, "
        f"regionB(between inner/outer)={d2}"
    )

    print(
        "HSV volume coverage: "
        f"regionA={100.0 * float(meta1['coverage']):.2f}% "
        f"with fixed bins size=({int(meta1['h_bins'])},{int(meta1['s_bins'])},{int(meta1['v_bins'])}) "
        f"bars={int(meta1['bars'])}; "
        f"regionB={100.0 * float(meta2['coverage']):.2f}% "
        f"with fixed bins size=({int(meta2['h_bins'])},{int(meta2['s_bins'])},{int(meta2['v_bins'])}) "
        f"bars={int(meta2['bars'])}; "
        f"edgeV={float(meta1.get('extreme_v_fraction', 0.0)):.3f}."
    )

    # Keep this figure screen-friendly while showing charts side by side.
    fig, axs = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    ax1, ax2 = axs[0], axs[1]

    h_u8 = ((packed_hsv >> 16) & 0xFF).astype(np.uint8)
    s_u8 = ((packed_hsv >> 8) & 0xFF).astype(np.uint8)
    v_u8 = (packed_hsv & 0xFF).astype(np.uint8)

    def _show_hsv_match_window(row: dict[str, float | int], region_mask: np.ndarray, chart_title: str) -> None:
        h0, h1 = int(row["h0"]), int(row["h1"])
        s0, s1 = int(row["s0"]), int(row["s1"])
        v0, v1 = int(row["v0"]), int(row["v1"])
        hsv_match = (
            (h_u8 >= h0) & (h_u8 <= h1)
            & (s_u8 >= s0) & (s_u8 <= s1)
            & (v_u8 >= v0) & (v_u8 <= v1)
        )
        show_mask = hsv_match & region_mask
        out = np.ones_like(rgb_source, dtype=np.float32)
        out[show_mask] = rgb_source[show_mask]

        fig_match, ax_match = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        ax_match.imshow(out, interpolation="nearest")
        ax_match.set_title(
            f"{chart_title}\n"
            f"H[{h0}-{h1}] S[{s0}-{s1}] V[{v0}-{v1}]  matched={int(np.count_nonzero(show_mask))}"
        )
        ax_match.set_xticks([])
        ax_match.set_yticks([])
        plt.show(block=False)
        plt.pause(0.001)

    def _attach_bar_hover_tooltips(
        ax,
        bars,
        labels: list[str],
        rows: list[dict[str, float | int]],
        region_mask: np.ndarray,
        chart_title: str,
    ) -> None:
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

        def _on_move(event):
            if event.inaxes != ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                return

            for i, bar in enumerate(bars):
                hit, _ = bar.contains(event)
                if hit:
                    x = event.xdata if event.xdata is not None else float(bar.get_width())
                    y = event.ydata if event.ydata is not None else float(bar.get_y() + 0.5 * bar.get_height())
                    annot.xy = (x, y)
                    annot.set_text(labels[i])
                    if not annot.get_visible():
                        annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return

            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

        def _on_click(event):
            if event.inaxes != ax or getattr(event, "button", None) != 1:
                return
            for i, bar in enumerate(bars):
                hit, _ = bar.contains(event)
                if hit:
                    _show_hsv_match_window(rows[i], region_mask, chart_title)
                    return

        fig.canvas.mpl_connect("motion_notify_event", _on_move)
        fig.canvas.mpl_connect("button_press_event", _on_click)

    def _plot_vol(
        ax,
        title: str,
        rows: list[dict[str, float | int]],
        meta: dict[str, float | int],
        region_mask: np.ndarray,
    ):
        if not rows:
            ax.set_title(f"{title}\n(no pixels)")
            ax.axis("off")
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
            f"{title}\n"
            f"Fixed HSV bins: coverage={100.0 * float(meta['coverage']):.2f}% "
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
        _attach_bar_hover_tooltips(ax, bars, labels, rows, region_mask, title)

    _plot_vol(ax1, "Region A: outside outer U inside inner", vol1, meta1, region_union_outside_plus_inner)
    _plot_vol(ax2, "Region B: between inner and outer", vol2, meta2, region_between_inner_outer)
    fig.suptitle(
        "HSV volume-count summaries by spline-defined regions\n"
        f"exact distinct counts: regionA={d1}, regionB={d2}"
    )
    # Render summary window without blocking; final blocking show occurs in _show_map.
    plt.show(block=False)
    plt.pause(0.001)


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
        default=8.0,
        help="High-pass radius percentage (0..100), same meaning as in fft_image_explorer.py (default: 8)",
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
        default="near100",
        help="Color scaling for output map (default: near100)",
    )
    parser.add_argument(
        "--near100-alpha",
        type=float,
        default=0.25,
        help="Exponent for near100 scaling; smaller values increase detail near 100 (default: 0.25)",
    )
    parser.add_argument(
        "--trace-threshold",
        type=float,
        default=99.0,
        help="Threshold used for tracer HIGH/LOW state criterion (default: 99.0)",
    )
    parser.add_argument(
        "--extreme-v-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of V range near black/white treated as full H/S coverage "
            "in fixed HSV bins (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--hsv-h-bins",
        type=int,
        default=24,
        help="Number of H bins for fixed HSV volumes (default: 24).",
    )
    parser.add_argument(
        "--hsv-s-bins",
        type=int,
        default=3,
        help="Number of S bins for fixed HSV volumes (default: 3).",
    )
    parser.add_argument(
        "--hsv-v-bins",
        type=int,
        default=4,
        help="Number of V bins for fixed HSV volumes (default: 4).",
    )
    parser.add_argument(
        "--trace-debug-start-outer",
        type=int,
        default=-1,
        help=(
            "Emit full per-candidate trace debug from this outer-spline-point index onward; "
            "default: off."
        ),
    )
    parser.add_argument(
        "--trace-debug-start-inner",
        type=int,
        default=-1,
        help=(
            "Emit full per-candidate trace debug from this inner-spline-point index onward; "
            "default: off."
        ),
    )
    args = parser.parse_args()

    hp_percent = float(np.clip(args.highpass_percent, 0.0, 100.0))
    metric = args.metric
    step = max(1, int(args.step))
    window_size = max(8, int(args.window_size))
    display_scale = args.display_scale
    near100_alpha = max(0.05, float(args.near100_alpha))
    extreme_v_fraction = float(np.clip(args.extreme_v_fraction, 0.0, 0.49))
    hsv_h_bins = max(1, int(args.hsv_h_bins))
    hsv_s_bins = max(1, int(args.hsv_s_bins))
    hsv_v_bins = max(1, int(args.hsv_v_bins))
    trace_threshold = float(np.clip(args.trace_threshold, 0.0, 100.0))
    trace_debug_start_outer: int | None = (
        int(args.trace_debug_start_outer) if int(args.trace_debug_start_outer) >= 0 else None
    )
    trace_debug_start_inner: int | None = (
        int(args.trace_debug_start_inner) if int(args.trace_debug_start_inner) >= 0 else None
    )
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
        extreme_v_fraction=extreme_v_fraction,
        hsv_h_bins=hsv_h_bins,
        hsv_s_bins=hsv_s_bins,
        hsv_v_bins=hsv_v_bins,
        trace_threshold=trace_threshold,
        trace_debug_start_outer=trace_debug_start_outer,
        trace_debug_start_inner=trace_debug_start_inner,
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
    extreme_v_fraction: float,
    hsv_h_bins: int,
    hsv_s_bins: int,
    hsv_v_bins: int,
    trace_threshold: float,
    trace_debug_start_outer: int | None,
    trace_debug_start_inner: int | None,
) -> None:
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
        constrained_layout=True,
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

        traced_outer = None
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
            ax_src.plot(traced_outer[:, 0], traced_outer[:, 1], color="white", linewidth=2.0, alpha=0.95)

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
                    ax_src.plot(traced_inner[:, 0], traced_inner[:, 1], color="cyan", linewidth=1.8, alpha=0.95)
                    centerline = _build_centerline_spline(traced_outer, traced_inner)
                    if centerline is not None:
                        print(
                            "Centerline spline: built midpoint curve from outer/inner splines; "
                            f"samples={centerline.shape[0]}"
                        )
                        ax_src.plot(
                            centerline[:, 0],
                            centerline[:, 1],
                            color="yellow",
                            linewidth=1.6,
                            alpha=0.95,
                        )

                    # HSV counting over requested spline-defined regions.
                    outer_xy = _normalize_closed_spline_xy(traced_outer, min_open_points=300)
                    inner_xy = _normalize_closed_spline_xy(traced_inner, min_open_points=300)
                    if outer_xy is not None and inner_xy is not None:
                        inside_outer = _mask_inside_closed_spline((out.shape[0], out.shape[1]), outer_xy)
                        inside_inner = _mask_inside_closed_spline((out.shape[0], out.shape[1]), inner_xy)
                        region_between_inner_outer = inside_outer & (~inside_inner)
                        region_union_outside_plus_inner = (~inside_outer) | inside_inner
                        packed_hsv = _packed_hsv_from_rgb(rgb_source)
                        _dbg("calling _plot_hsv_region_count_summary (this opens a blocking figure)")
                        _plot_hsv_region_count_summary(
                            rgb_source=rgb_source,
                            packed_hsv=packed_hsv,
                            region_union_outside_plus_inner=region_union_outside_plus_inner,
                            region_between_inner_outer=region_between_inner_outer,
                            extreme_v_fraction=extreme_v_fraction,
                            h_bins=hsv_h_bins,
                            s_bins=hsv_s_bins,
                            v_bins=hsv_v_bins,
                        )
                        _dbg("returned from _plot_hsv_region_count_summary")
                    else:
                        print(
                            "HSV count summary skipped: spline geometry could not be coerced to Nx2 "
                            f"(outer shape={np.asarray(traced_outer).shape}, inner shape={np.asarray(traced_inner).shape})."
                        )

                    if traced_outer is not None and _spline_intersects(traced_outer, traced_inner):
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


if __name__ == "__main__":
    main()

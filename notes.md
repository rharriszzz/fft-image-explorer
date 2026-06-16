# FFT Notes: Background vs Bracelet Classification

## Useful Background Context

These notes summarize practical lessons from exploring FFT behavior in this project.

- Hard rectangular windows inject strong boundary effects (sinc-like leakage and axis-aligned structure).
- Gaussian windows reduce hard-edge artifacts and usually produce smoother radial decay.
- Zero padding can introduce boundary discontinuity artifacts (axis lines, ripples) if edge values differ strongly from zero.
- Edge-propagated padding is usually gentler than zero padding for texture analysis.
- Radial FFT plots in this app are shell-summed: each radius bin plots `log1p(sum |F|^2)`.
- Shell sums are not the same as per-frequency PSD; interpretation should account for ring circumference growth.

## Interpreting Typical Patterns

- Construction-paper background often appears relatively isotropic in frequency space.
- Bracelet regions can show directional structure (for example a diagonal ridge with bumps).
- A strong center point indicates DC/mean luminance dominance.
- Broad mid-band energy often indicates texture-like content.

## Proposed Test for Background Identification

To separate background vs bracelet with good physical resolution, use both radial and directional features from a fixed local window.

### 1) Keep Windowing Consistent

- Use a fixed Gaussian window size tied to target spatial resolution.
- Keep softness fixed during comparisons.
- Keep channel fixed (for example Y luminance).

### 2) Compute Two Spectral Summaries

- Radial profile `P(r)`: energy vs radius.
- Angular profile `A(theta)`: energy vs angle over a selected radius band.

### 3) Feature Set

- Anisotropy index:
  - `AI = std(A(theta)) / mean(A(theta))`
  - Low for isotropic background, higher for bracelet structure.

- Directional peak ratio:
  - `DPR = max(A(theta)) / median(A(theta))`
  - Higher when one direction dominates (for example diagonal line).

- Radial bumpiness:
  - Peak count or peak prominence on a smoothed `P(r)`.
  - Bracelet-like regions often have stronger structured bumps.

- Mid/high-band energy fraction:
  - `E_mid = sum(P(r), r in [r1, r2]) / sum(P(r), r in [0, rmax])`
  - Helps distinguish smooth background from more detailed foreground texture.

### 4) Practical Decision Rule

Classify as background when most of the following are true:

- `AI` is low
- `DPR` is low
- radial bumpiness is low
- energy decays smoothly with radius

Classify as bracelet/foreground when one or more are strong:

- `AI` high
- `DPR` high
- pronounced bumpiness in radial profile

## Why This Is Better Than Radial-Only

A radial-only curve can miss directional structure. The bracelet signature you observed (diagonal line with bumps) is strongly directional, so combining radial and angular features is much more reliable than radial power alone.

## Recommended Next Step in App

Add a simple live "background score" panel that reports:

- `AI`
- `DPR`
- radial bumpiness
- combined score

This gives an immediate, testable metric while preserving the current visual workflow.

## Planned Next Steps (User Roadmap)

### 0) Determine Gaussian Parameters for Later Local FFT Sampling

Choose a method to estimate the desired Gaussian softness and radius:

- Option A: run a full FFT on the entire rectangular image and extract the relevant scale information.
- Option B: draw lines through the image that intersect the bracelet and estimate bracelet width directly from line-profile pixels.

### 1) Build a High-Resolution Bracelet Boundary Contour

Use the background detector to produce a high-resolution contour of the two bracelet boundaries.

### 2) Fit Boundary and Centerline Splines

- Fit splines to both bracelet boundaries.
- Compute a centerline spline halfway between those boundaries.

### 3) Run Local Gaussian 2D FFT Along the Centerline

At closely spaced points along the centerline:

- run Gaussian-windowed 2D FFT,
- record angle, distance, and power of the top 2 to 4 non-center peaks,
- treat conjugate peak pairs as one peak.

### 4) Continue With Additional Downstream Steps

Additional processing and analysis steps will follow after the centerline FFT characterization phase.

## Additional Advice for Future Sessions

### Reproducibility Checklist

When comparing regions, keep these fixed unless intentionally testing sensitivity:

- channel (for example Y luminance)
- window type and parameters (mask type, width, height, softness)
- FFT filtering settings (high-pass, low-pass, threshold)
- padding mode and FFT-size strategy

Log these with every experiment so results remain comparable.

### Prefer Relative Features Over Absolute Magnitudes

Absolute FFT magnitudes can drift with exposure, gain, and local brightness. Use ratios and normalized descriptors where possible:

- anisotropy ratio (`AI`)
- directional peak ratio (`DPR`)
- normalized band energy fractions
- relative peak power vs local baseline

### Add Confidence Flags to Avoid False Decisions

Before trusting a classification, check:

- minimum total power in analysis band
- SNR threshold for selected FFT peaks
- boundary quality score (for bracelet contours)

If confidence is low, return "uncertain" instead of forcing a label.

### Use Two Radial Metrics, Not One

Keep both:

- shell-summed radial power (current view)
- mean power per ring (`sum / ring_count`)

This prevents misinterpretation from ring-circumference effects.

### Directional Analysis Suggestion

For angular structure, compute `A(theta)` only in a radius band that excludes:

- very low radius (DC and illumination shading)
- very high radius (sensor/compression noise floor)

This usually stabilizes bracelet-vs-background contrast.

### Boundary/Centerline Robustness Tips

- Smooth contours before spline fitting.
- Enforce approximately parallel inner/outer boundary progression.
- Sample centerline at near-constant arclength spacing.
- Store local normal/tangent at each sample point for consistent angle interpretation.

### Peak Tracking Along the Bracelet

For per-point local FFT peak tracking:

- merge conjugate pairs (already done conceptually)
- track peaks frame-to-frame along centerline by nearest angle/radius continuity
- reject sudden one-step jumps unless power rises strongly

This prevents peak ID swapping from dominating downstream statistics.

### Practical Data Logging Format

For each sampled centerline point, save a compact row containing:

- point index, arclength
- centerline x/y
- local tangent angle
- window params (radius/softness)
- top peak tuples: (radius, angle, power)
- radial summary stats
- anisotropy metrics

A CSV or parquet table here will make later modeling much easier.

### Fast Validation Experiments Worth Running

- background-only patches at multiple image locations
- bracelet-only patches at multiple orientations
- same patch with small brightness shifts
- same patch with slight scale changes

Goal: confirm feature stability under nuisance changes while preserving bracelet/background separation.

### Good Immediate Engineering Next Step

Add a lightweight "analysis export" action that writes one row per current ROI click.
This creates a training/evaluation dataset while you continue tuning the physics-informed features.

## Session Handoff Template

Use this block at the end of each focused work session.

### Date/Commit

- Date:
- Branch:
- Latest commit:

### Current Objective

-

### Settings Used for Main Results

- Image:
- Channel:
- Mask type:
- Edge type:
- Width/Height:
- Softness:
- FFT high-pass / low-pass / threshold:
- Padding mode:

### Key Observations

-

### Metrics Snapshot

- AI:
- DPR:
- Radial bumpiness:
- Mid/high-band fraction:
- Confidence flag:

### What Changed This Session

-

### What Broke / Caveats

-

### Next 1-3 Actions

1.
2.
3.

### Open Questions

## Current Spline Implementation Notes

This section captures the currently implemented method in `scan_highpass_removed_map.py`.

### How Spline Start Points Are Found

Outer spline start point:

- Build a coarse grid from the FFT metric map using the scanner step size.
- Use the thresholded coarse map (`HIGH`/`LOW`).
- Scan from image-center lines toward edges in this order:
  - center row: right to left,
  - center row: left to right,
  - center column: bottom to top,
  - center column: top to bottom.
- Choose the first `LOW` cell encountered as outer start.

Inner spline start point:

- Begin from outer spline point 0.
- Examine ring transitions around that point at an initial ring radius.
- Expand ring radius until a new transition pair is found relative to the previous ring.
- Use the midpoint between that new `HIGH->LOW` and `LOW->HIGH` transition pair as inner start.

### How Outer and Inner Splines Are Built

Outer spline:

- Trace on the coarse step grid.
- At each current point, evaluate ring candidates in clockwise order.
- Expect one `LOW->HIGH` and one `HIGH->LOW` transition around the ring.
- Use local transition geometry and turn control to choose the next point.
- Stop on transition mismatch, cycle closure, or guard conditions.
- Smooth and close the resulting curve.

Inner spline:

- Use the same tracer logic, seeded from the inner start point.
- Apply the same ring-transition and stopping logic.
- Smooth and close the curve.
- Enforce non-intersection with outer spline; raise an error if inner crosses outer.

### Centerline Strategy (Current Method)

- Resample inner and outer curves to comparable dense closed polylines.
- For each inner sample point:
  - find the nearest point on the outer polyline (nearest point on segments, not only vertices),
  - compute vector from inner to that outer point,
  - move halfway along that vector.
- Collect all halfway points as the raw centerline samples.
- Smooth and close the final centerline spline.

This centerline method was chosen because it remains usable when bracelet outlines are non-convex.

-

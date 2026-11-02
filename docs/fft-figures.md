# Reproducing the local FFT figures

These figures are computed composites of the committed `beads-photo-2.jpg`, not application screenshots. The saved `centerline_spline` supplies the coordinates. No reconstructed or generated necklace imagery is used.

## Regenerate

From a checkout with the dependencies in `requirements.txt` installed:

```bash
python3 scripts/generate_fft_figures.py
python3 -m unittest discover -s tests -v
```

The generator resolves inputs relative to its own repository, never the historical absolute paths in the JSON. Missing inputs fail. `--output /tmp/fft-figures` writes elsewhere; `--indices 5 80 280` explicitly selects the three zero-based spline indices, with the first becoming the hero. Outputs include three individual candidates, a contact sheet, hero, comparison, a dedicated 1600 × 1000 project card, its 400 × 250 preview and `measurements.json` with source SHA256 and peak coordinates/amplitudes. Reference environment: Python 3.14.4, NumPy 2.3.5, Matplotlib 3.10.7, Pillow 12.1.1. PNG rendering may vary with library/font versions; the numerical recipe is fixed.

## Project card

[Full-resolution card](figures/local-fft-card.png) · [400-pixel preview](figures/local-fft-card-preview.png)

The owner selected Candidate 5. The card is a separate two-panel composition of its original photograph and actual Gaussian-windowed luminance FFT, generated directly from the same calculation as the three-panel hero. It is always fixed to index 5, even when `--indices` selects different candidates for the supporting figures. Run the regeneration command above to recreate both card files.

The 1600 × 1000 PNG has a white background, 24-pixel outer margins, a 20-pixel panel gap and just two labels. The photograph takes 48% of the available panel width and the FFT 52%. The photograph is centered on the exact saved point, cropped to ±90 source pixels horizontally; its vertical span follows the panel aspect ratio. Both 1σ (solid) and 2σ (dashed) contours fit, with a white cross at the selected point and a subtle centerline. The same faint Gaussian weighting overlay as the hero is retained. All Gaussian and FFT parameters are unchanged.

The FFT view is cropped to ±0.08 cycles/pixel horizontally; its vertical range is ±(0.08 × panel height / panel width). Equal data aspect preserves geometry in both panels. Frequency y still points down; the FFT is neither rotated nor stretched. The log display and saturation limits remain `ln(1 + |FFT|)` in [1,6], exactly as in the hero. No axes, direction arrows, peak markers or colorbar are included in this introductory card; the full figure supplies scientific context. DC remains present and display-saturated. Visible square FFT bins and limited photographic sharpness at full resolution reflect the original data rather than added smoothing or invented detail.

Four layouts were inspected at 400 × 250: equal panels, a larger FFT, a larger photograph, and the selected near-equal layout with a tighter photo crop. The selected layout keeps beads recognizable and both contours visible while giving the spectrum slightly more room. The final preview is a uniform Lanczos reduction of the dedicated card, not of the hero; the labels remain readable at this size. Layout alternatives are not published in the README. This is a visual introduction, not an illustration for precise angle measurement or geometric reconstruction.

## Selection and interpretation

An initial visual survey examined indices 5, 20, 45, 80, 105, 145, 160, 190, 210, 225, 250 and 280 with identical windows. The retained candidates show recognizable beads, manageable curvature and different visible color mixtures. The contact sheet retains the three complete triptychs.

| Index | Saved x, y in original pixels | Interpretation |
|---|---|---|
| **5 — hero** | **2293.274658, 1746.540039** | Right side, yellow/red/black transition; nearly straight axis and clearly separated off-center peak pairs. |
| 80 | 955.460449, 2606.994141 | Lower left, yellow and red around the center with black separators; more curvature and stronger leading peak, but less clean separation. |
| 280 | 2151.359619, 973.794067 | Upper right, predominantly red/black near the center with yellow farther out; extra low-frequency structure. |

Candidate 5 is selected for clarity and balance, not maximum peak magnitude: candidate 80's leading peak is stronger. The three-panel figure is intended for the README at larger sizes; the dedicated card below replaces its former 400-pixel thumbnail. The photo panel spans 270 pixels while the scalar panel shows all 401 input pixels; their display magnifications deliberately differ.

In tangent/normal coordinates, a roughly near-tangent feature has `(ft, fn)` of `(0.0478, −0.0044)`, `(0.0434, −0.0101)` and `(0.0450, −0.0087)` cycles/pixel respectively. Its amplitudes are 262, 225 and 278, and its angles relative to the tangent are −5.3°, −13.1° and −10.9°. Two oblique families lie approximately at +42° to +49° and −49° to −61°. These are descriptive correspondences, not a fitted lattice assignment. Their ranks change. Candidate 280 also has a strong lower-frequency pair near `(0.0102, 0.0123)` without an equally strong counterpart in the other two top-four lists. Thus orientation correction reveals approximate similarity, not identical spectra.

Repeated bead spacings contribute directional reciprocal-space peaks. A Fourier vector describes variation perpendicular to constant-phase stripes; its direction is not automatically a bead-to-bead direction. Colors, specular highlights, shadows, background texture, projection and the finite window also contribute. Greater peak strength does not establish greater geometric regularity, and these three samples do not establish stability around the entire necklace.

## Numerical and display conventions

- Original 2540 × 3182 photograph, EXIF orientation honored, no resize or color correction. Pixel centers use x right, y down. The saved centerline is an estimated photographic guide, not a validated physical axis.
- Scalar channel: **Y = 0.299 R + 0.587 G + 0.114 B**, with encoded RGB values in [0,1], using `map_from_fft.load_image_and_luminance`. This is the project's luminance convention, not linear-light radiometry and not three bead-color classes. Y preserves useful bright/dark structure with a fixed definition. Hue is unsuitable as an ordinary scalar across its wrap discontinuity.
- A same-window numerical check of HSV Value (`max(R,G,B)`) retained several peaks but changed others and their ranks. For example the hero's leading Y pair `(ft,fn)=(0.0233,0.0265)` shifted to `(0.0161,0.0210)` as the leading Value pair, while its near-tangent pair stayed in the same bin. Y is a clear demonstration channel here, not a color-invariant geometry estimator.
- **401 × 401 native-pixel patch**, centered on the nearest integer pixel for array extraction. The Gaussian itself remains centered on the **exact floating-point spline point**. No spatial resampling.
- Reuse `fft_image_explorer.make_mask`, `Ellipse/circle`, `Gaussian`, width = height = **401 px**, softness **0.2**, angle 0. Thus **σ = 40.1 px**, 1σ radius 40.1 and 2σ radius 80.2. Weight is `exp(−r²/(2σ²))`, truncated at circular radius 200.5 (5σ), weight ≈ 3.73e−6 at that boundary. The finite square also bounds storage. This agrees with the scanner Gaussian inside the circle, while the scanner retains square corners. The old map's 128-pixel window and 8% high-pass setting are not used for these figures.
- FFT input is exactly `Y_patch * mask` (float32). NumPy `fft2` uses default backward normalization: the forward transform is an unnormalized sum. Then `fftshift`; **no padding, filtering, mean subtraction, resizing or per-location amplitude normalization**. The scalar panel uses fixed [0,1] grayscale, without gamma or contrast adjustment.
- Spectrum display: **natural `ln(1+abs(F))`**, fixed color limits **[1,6]** across every candidate. DC remains in the data and saturates the display. Only the central ±0.10 cycles/pixel is shown; Nyquist is ±0.5 and bin spacing is 1/401 ≈ 0.002494 cycles/pixel. Pixel-edge extents derive from `fftfreq`. Positive fy points down, matching image coordinates. No FFT image is rotated.
- Peak extraction starts with the project's local-maxima finder. The generator excludes a six-bin DC neighborhood and suppresses maxima within six bins of either member of an already selected conjugate pair. This is a measurement selection, not filtering of the FFT. Positions are bin centers, without sub-bin fitting; broad peaks and finite windows limit precision. `measurements.json` keeps six pairs, while the comparison plots four.
- Tangent `t` is the normalized secant from spline index i−2 to i+2, with periodic indexing; `n=(-ty,tx)`. The comparison projects peak vectors as `(f·t,f·n)`, choosing the conjugate with nonnegative ft. This is an explicitly labeled coordinate transformation of points only. Dot size uses the same magnitude mapping for all rows; numbering/color indicates local amplitude rank, not matched physical identity.

The related [beads FFT notes](https://github.com/rharriszzz/beads/blob/photo-2-strategy-v3/photo2/FFT_EXPLORER_NOTES.md) record broader paired V/Y experiments, width sensitivity and the need to preserve competing peak interpretations. These figures are a small independent illustration, not a rerun of that reconstruction campaign.

## Validation

The numerical tests check source/scalar identity, subpixel Gaussian centering and the analytic sigma, agreement with the scanner inside circular support, FFT inverse/DC/Parseval/conjugacy, measured peak bins, orthonormal coordinate projection, and positive frequency orientation using a known sinusoid. The sinusoid is test data only; all published image panels use the photograph. Full-size candidates, comparison and card preview were visually inspected. Clean-tree regeneration and artifact comparison are recorded in the completion report.

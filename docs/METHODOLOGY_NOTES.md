# Methodology Notes: Rotation-Period Detection Algorithm

**Purpose**: notes-to-self capturing the current state of the algorithm in
enough mathematical and procedural detail to write a methods section from,
once the procedure has stabilized. This is **not** paper prose and should
not be copied into a manuscript directly — it's a technical reference,
written to be precise rather than polished, so nothing has to be
reconstructed from code later. Sections are organized the way a methods
section likely would be, so migrating content later should be
mostly-mechanical.

**Status flag used throughout**:
- 🟢 STABLE — unlikely to change before publication; safe to start
  drafting prose from.
- 🟡 LIKELY TO EVOLVE — current approach, but actively being iterated on
  (e.g. short-period recovery). Capture the current version, but expect
  to revisit.
- 🔴 UNVALIDATED AT SCALE — implemented and tested on small hand-picked
  samples only; not yet run against anything close to the full ~1M-star
  corpus. Treat any specific numbers below as illustrative, not final.

---

## 1. Overview

The algorithm identifies stellar rotation periods from an autocorrelation
function (ACF) by jointly fitting a comb of evenly-spaced, downward-
opening parabolae to its repeated peaks. It generalizes the original
SpinSpotter (Holcomb et al. 2022) single-parabola approach, which required
a reasonably accurate a priori period estimate to select the correct ACF
region to fit. The redesign removes that requirement by (1) generating a
pool of candidate periods from several independent, cheap, method-native
heuristics, and (2) fitting the *same* joint comb model to every
candidate and using fit quality itself — plus several physically-motivated
reliability checks — to arbitrate between them. 🟢

This two-stage separation (propose candidates cheaply; evaluate them
properly and uniformly) is itself a methodological choice worth stating
explicitly in the eventual methods section: an earlier design had each
candidate-generation method cross-validate its own proposals against the
ACF, which turned out to be exploitable (see Section 6.1) and made each
method's behavior harder to characterize in isolation.

---

## 2. Light curve preparation 🟡

### 2.1 Source data

Development and testing use SMARTS-simulated light curves: real TESS
photometric backgrounds with synthetic starspot-modulation signals of
known period injected on top, at 30-minute cadence. Because the intended
downstream application is real TESS data, and because training a
selection model on periods *measured* from real TESS data by an existing
pipeline would be circular (the model would partly learn to reproduce that
pipeline's own biases), all training labels come from the known injected
period rather than any measured value. This is the reasoning that
justifies using SMARTS at all, and is worth stating explicitly in a Data
section.

### 2.2 Gap re-identification and orbit segmentation

The distributed SMARTS light curves have their per-orbit data gaps
already linearly interpolated over, which makes them look artificially
more complete than real TESS data. Since realistic gap structure (data
downlink losses, momentum-dump cuts, missing sectors) materially affects
ACF/periodogram behavior (see Section 6.4), gaps are algorithmically
re-identified rather than left interpolated:

1. A rolling standard deviation of the flux is computed (window ~9
   points). Linearly-interpolated stretches are locally flat, so this
   statistic drops sharply and briefly wherever a gap was filled in.
2. That rolling std is compared against a much more slowly-varying local
   baseline (a rolling median of the rolling std over ~10 days), so a
   genuinely low-scatter orbit isn't mistaken for a gap and a
   high-scatter orbit's real gaps aren't missed. A point is flagged as
   "inside a gap" if its rolling std falls below a fixed fraction
   (default 0.2) of this local baseline, for at least a minimum run
   length (default 10 points) to reject momentary noise dips.
3. The light curve is split into individual-orbit segments at these
   identified gap centers (falling back to a nominal evenly-spaced
   boundary, flagged low-confidence, if no convincing gap is found near
   an expected boundary — e.g. because a flare masks the interpolation
   signature there).
4. Each orbit's flux scatter (gap points excluded) and median rolling std
   (gap points included) are compared across all orbits; orbits with
   anomalously high or low scatter, or anomalously low median rolling std
   (indicating a mostly-interpolated segment the gap search itself
   missed), are flagged and can be dropped.
5. Orbits are recombined into one timeseries, optionally restricted to a
   hand-picked subset of sectors — used to deliberately degrade a long,
   mostly-complete SMARTS baseline down to a sparser, more realistic
   TESS-like observing pattern for testing.
6. The result is regridded onto a fixed-cadence (1800 s) grid, with
   missing cadences represented as NaN (not dropped, not interpolated) —
   see Section 3 for why this representation matters.

This gap-recovery step is itself somewhat delicate (thresholds for what
counts as a "gap" vs. genuine low-scatter data) and has been visually
validated (gap detections overlaid on real light curves) but not
systematically characterized (e.g. false-positive/negative gap-detection
rate across a large sample). Worth a validation figure in the paper if
this preprocessing step is described in any detail.

---

## 3. Autocorrelation function computation 🟢

Real TESS-like light curves are evenly cadenced but gappy (NaN at missing
timestamps). A standard FFT-based autocorrelation
(`ifft(fft(x) * conj(fft(x)))`) is undefined in the presence of NaNs (a
single one propagates through the whole transform), and the two common
workarounds both have real drawbacks for this application: linear
interpolation across gaps injects a fabricated trend that can bias the
ACF, particularly for the multi-week dropouts common in real TESS data;
restricting to the single longest gap-free segment is often shorter than
even one stellar rotation for realistically gappy targets, especially
short-period ones.

**Masked-FFT autocorrelation.** Let $f_i$ be the flux at cadence $i$,
valid (finite) or not, and let $\bar{f}$ be the mean of the valid points
only. Define

$$x_i = \begin{cases} f_i - \bar{f} & \text{if } f_i \text{ valid} \\ 0 & \text{otherwise} \end{cases}, \qquad m_i = \begin{cases} 1 & \text{if } f_i \text{ valid} \\ 0 & \text{otherwise} \end{cases}$$

Zeroing (rather than dropping) invalid points means any pair $(i, i+k)$
where either point is invalid contributes exactly zero to a raw
autocorrelation sum, rather than `NaN`. Compute, via FFT (zero-padded to
avoid circular wraparound):

$$\text{raw\_ACF}(k) = \sum_i x_i \, x_{i+k}, \qquad N_\text{valid}(k) = \sum_i m_i \, m_{i+k}$$

$N_\text{valid}(k)$ — obtained by autocorrelating the validity mask
itself, the same way — counts exactly how many valid $(i, i+k)$ pairs
contributed to the raw sum at each lag. The gap-corrected ACF is then

$$\text{ACF}(k) = \frac{\text{raw\_ACF}(k)}{N_\text{valid}(k)}$$

normalized so that $\text{ACF}(0) = 1$. Both sums are computed via FFT in
$O(N \log N)$; the whole procedure adds no meaningful cost over the naive
(gap-incapable) version.

**Truncation.** The returned ACF is truncated at whichever comes first:
(a) a fixed fraction of the full baseline (default 1/3), or (b) the first
lag at which $N_\text{valid}(k)$ falls below a fraction (default 0.3) of
$N_\text{valid}(0)$ and does not recover — beyond that point the ACF value
is an average over too few surviving pairs to trust, which matters far
more for gappy data (a single large dropout can leave almost no valid
pairs at moderate-to-large lags well before the nominal baseline-fraction
cutoff would have stopped) than for the clean case that fraction-of-
baseline cutoff alone was designed for.

---

## 4. Candidate period generation 🟡

Seven methods, each proposing a short ranked list of candidate periods
from a different lens on the data. **None of them checks a candidate
against the ACF's actual shape, and none does any curve fitting** — that
distinction (candidate proposal vs. evaluation) is central to the design;
see Section 1 and Section 6.1.

### 4.1 Pairwise ACF-peak-spacing histogram 🟢

Local maxima of the ACF are found (`scipy.signal.find_peaks`), excluding
a small buffer around lag zero (always a trivial peak) and using an
adaptive prominence threshold ($5\times$ the standard deviation of the
ACF's second difference — see Section 6.2 for why this needed to be
adaptive rather than fixed). For $m$ found peaks at lags $x_1 < \dots <
x_m$, every pairwise positive difference $x_j - x_i$ ($j>i$) is computed
and histogrammed. Each local maximum ("bump") in that histogram is a
candidate period, ranked by histogram count.

This ranking is structurally resistant to harmonic ambiguity: if the
peaks are (approximately) evenly spaced at the true period $P$, spacing
$P$ is supported by up to $m-1$ pairs (every adjacent pair), spacing $2P$
by only $m-2$ pairs, spacing $kP$ by $m-k$ pairs — support strictly
decreases with harmonic order, so simple count-ranking favors the
fundamental without needing any further cross-validation. This replaced
an earlier, more complex candidate-scoring scheme (see Section 6.1) and
is one of the more defensible individual design choices to describe in
detail in a methods section, since the resistance-to-harmonics property
follows directly and provably from the counting argument rather than
being empirically tuned.

### 4.2 Lomb-Scargle periodogram 🟢

Standard Lomb-Scargle periodogram of the light curve itself (`astropy`,
standard normalization), independent of the ACF entirely. Local maxima
ranked by power. Deliberately coarse frequency sampling
(`samples_per_peak=1`, down from `astropy`'s usual finer default) since
this stage only needs to localize a candidate to be refined later by the
joint fit — see Section 6.3 for the empirical justification.

### 4.3 FFT of the ACF 🟢

The ACF's own quasi-periodicity is treated as a signal: the ACF is
Hann-windowed, zero-padded (`oversample=4` by default, purely to
interpolate the power spectrum onto a finer grid for more precise peak
localization — it adds no new information), and Fourier-transformed.
Local maxima of the resulting power spectrum, in period space, are ranked
by power.

### 4.4 Global Wavelet Power Spectrum (opt-in) 🟡

Not run by default (requires gap-free flux; more expensive than the three
methods above). The light curve is cross-correlated with a complex Morlet
wavelet across a log-spaced grid of trial periods, producing a
time-resolved period-power surface; time-averaging that surface gives the
Global Wavelet Power Spectrum (GWPS). Distinct candidate peaks are
extracted by iteratively fitting a single Gaussian (in log-period space)
to the tallest remaining GWPS feature and subtracting it, following the
approach used in the wavelet stage of the Santos/ROOSTER pipeline
(Breton et al. 2021; see also García et al. 2014). Unlike the other
methods, this one retains time information (candidates whose periodicity
is present only during part of the baseline could in principle be
distinguished from persistent ones via the full 2D surface), though this
capability is not yet used downstream — only the time-averaged GWPS feeds
candidate generation currently.

### 4.5 Short-period-focused variants (opt-in) 🟡

Added specifically to address weak recovery of periods below ~10 days
(Section 6.5). Three variants:

- **`guess_lombscargle_short` / `guess_acf_fft_short`**: the same two
  methods above, restricted to a short-period search band (default
  $\leq 15$–50 days, evolving) *before* ranking, so a genuine short-period
  peak only has to out-compete other short-period candidates rather than
  every longer-period feature in the full spectrum.
- **`guess_acf_fft_highpass`**: the flux is high-pass filtered (a
  centered rolling-mean trend is computed at a given window scale and
  subtracted) *before* the ACF and its FFT are computed, at several
  window scales in parallel (current defaults span 2–40 days). Removing
  slow variability keeps a weak short-period signal from being dominated
  by a stronger longer-timescale trend in either the ACF or its
  spectrum. Shorter windows filter more aggressively and can measurably
  attenuate real signal at periods comparable to or longer than the
  window itself — accepted deliberately, since this method's output is
  only ever one contributor to a larger candidate pool evaluated
  independently by several other methods (see Section 1); a period this
  method smooths away is exactly the kind of signal a method that never
  touches the light curve's trend is left to supply.

Empirically (small test batch, 🔴 not yet validated at scale): neither
addition flips a hard case from a clean miss to a clean hit at a strict
tolerance, but both measurably improve *how close* candidates land to the
true period, and `guess_acf_fft_highpass` is currently the strongest
single contributor among the four short-period-focused/optional methods.
One tested case (period comparable to ~5-6 cadence samples) appears
unrecoverable by any periodogram-based method regardless of range
restriction or filtering — a fundamental sampling-resolution limit, not a
search-strategy problem.

---

## 5. Joint comb-of-parabolae fit 🟢

This is the core methodological contribution and the part most worth
describing carefully and precisely in the eventual paper.

### 5.1 Model

For a candidate period $P$ and phase $t_0$ (the lag of the first, $n{=}0$,
expected peak), a window is built around each expected peak location
$t_0 + nP$ for harmonic index $n = 0, 1, \dots, N_\text{peaks}-1$, each
spanning $\pm(\text{window\_frac}) \times P$ (default window\_frac $=
0.25$, i.e. a half-width of one quarter-period) around its center, clipped
to the ACF's available lag range. **These windows — which points are
included in the fit — are frozen before optimization begins** and are not
re-derived as $P$/$t_0$ are refined during the fit; only the parabola
parameters and the shared $P$/$t_0$ vary.

Within window $n$, the model is a downward-opening parabola:

$$\text{ACF}(\ell) \approx h_n - A_n (\ell - c_n)^2, \qquad A_n \geq 0$$

where $\ell$ is lag, $h_n$ and $A_n$ are that window's free height and
curvature parameters, and the center $c_n$ is **not independently free**
— it is algebraically tied to the shared $P$ and $t_0$:

$$c_n = t_0 + nP + \delta_n, \qquad |\delta_n| \leq (\text{jitter\_frac}) \times P \ \ (n>0)$$

$P$ and $t_0$ are single parameters shared across every window in the
fit; this tying is what makes "evenly spaced" a hard structural
constraint on the model rather than a property checked after an
unconstrained fit. $A_n \geq 0$ is a hard bound enforcing "opens
downward" (a genuine ACF peak, not a trough) independently for every
window. The small per-window jitter term $\delta_n$ (default bound
5% of $P$, $n>0$ only — the $n{=}0$ window's center is $t_0$ exactly)
tolerates gentle real period drift (differential rotation, spot
evolution) without abandoning the tied structure.

### 5.2 Optimization

Fit via `lmfit`'s `least_squares` backend with a robust loss function
(`soft_l1` by default), rather than ordinary least squares, so that one
badly-behaved window does not dominate the fit of the *shared* $P$/$t_0$
that every other window's fit also depends on. $P$ is bounded to
$P_0 (1 \pm 0.3)$ around the candidate's initial value — i.e. the joint
fit can refine a candidate by up to 30% from where it started, which is
also the operational tolerance used when evaluating whether a raw
candidate-generation proposal is "close enough" to plausibly be recovered
downstream (see Section 4.5).

### 5.3 Iterative peak rejection

After each fit, every window's residual RMS is computed. If the worst
window's RMS exceeds the median RMS across all currently-included windows
by more than a robust-scaled threshold (default $4\times$ the
MAD-based $\hat\sigma = 1.4826 \times \text{MAD}$), and dropping it would
not reduce the window count below a required minimum, that window is
removed and the fit is redone. This repeats up to a fixed number of
iterations (default 3). This is a simple RANSAC-style cleanup: a single
cycle disrupted by a flare, a data gap, or genuinely anomalous starspot
behavior can otherwise degrade the shared $P$/$t_0$ fit for every other,
otherwise-good window.

---

## 6. Candidate arbitration and reliability assessment 🟢

### 6.1 Why fit *every* candidate rather than pick one upfront

Early in development, each candidate-generation method attempted to
self-validate by scoring its own candidates against the ACF (a cheap
weighted-average "comb score"). This was found to be gameable: a coarse
comb touching only a handful of the tallest ACF points could score
deceptively well, favoring harmonics of the true period over the
fundamental. The current design instead treats every candidate from every
method identically: none are pre-filtered by a cheap heuristic, all are
fit with the full joint model above, and comparison happens only on that
equal footing.

### 6.2 Adaptive vs. fixed thresholds

More generally, several thresholds in this pipeline were found to need
adaptive (data-scale-relative) rather than fixed definitions to
generalize across the very different noise/amplitude regimes real
targets span — the ACF peak-finding prominence (Section 4.1) is the
clearest example, but the general principle (do not hard-code a threshold
in the ACF's or periodogram's native units; scale it to that specific
light curve's own noise level) is a recurring theme worth stating once,
generally, in a methods section rather than case-by-case.

### 6.3 Reliability gates

A successfully-fit candidate must pass **all three** of the following to
be eligible for selection:

1. $n_\text{peaks,used} \geq$ `min_peaks_required` (default 4): enough
   expected peaks survived fitting (and rejection, Section 5.3) to
   indicate a sustained periodicity rather than a couple of coincidental
   local matches.
2. $\text{frac\_positive\_heights} \geq$ `min_frac_positive_heights`
   (default 0.8): the fitted peak heights $h_n$ must mostly be genuine
   positive bumps. This gate exists because a comb can otherwise achieve
   a deceptively small reduced $\chi^2$ by fitting parabolae to the ACF's
   smooth decay *through* zero near lag 0 rather than to real periodic
   structure — a technically excellent local fit to a slope, not evidence
   of periodicity. Discovered empirically as a real failure mode, not
   anticipated in advance.
3. $\overline{\text{height\_snr}} \geq$ `min_mean_height_snr` (default
   1.0), where $\text{height\_snr}_n = h_n / \sigma_\text{ACF}$: peaks
   must be tall enough relative to the ACF's overall noise level, on
   average, to be distinguishable from noise fluctuations.

Among candidates passing all three, the one with the lowest reduced
$\chi^2$ is selected. **If no candidate passes all three gates, no period
is returned** (`success = False`) — an explicit, logged non-detection
rather than the best available (but non-plausible) fit. This is a
deliberate design stance for an automated, large-scale pipeline: a
silently-wrong period is more costly downstream than a flagged
non-detection, and there is no reason to expect the "least bad" candidate
among a set that all failed basic plausibility checks to be meaningfully
more likely correct than any other.

### 6.4 Model-independent cross-checks (optional diagnostics)

Two further diagnostics are available (not part of the gating logic
above; computed on request, e.g. as ML features or for manual inspection)
specifically because they evaluate a candidate through a different lens
than the joint fit's own $\chi^2$ and can catch failure modes redchi
cannot.

**Phase Dispersion Minimization** (Stellingwerf 1978). For a candidate
period $P$, the light curve is folded (phase $= (t \bmod P)/P$) and
binned into $M$ phase bins. The statistic

$$\theta = \frac{\sum_j (n_j - 1) s_j^2}{(N - M)\, \sigma_\text{total}^2}$$

(where $s_j^2$ is the variance within bin $j$, $n_j$ its point count, $N$
the total point count, $\sigma_\text{total}^2$ the full light curve's
variance) approaches 0 for a correctly-folded, genuinely periodic signal
and approaches 1 for an uninformative period. Critically, this statistic
is evaluated **directly on the light curve**, with no ACF or parabola
model involved at all — a genuinely independent check.

A practical subtlety, discovered empirically: $\theta$ is far less
forgiving of period imprecision than the joint fit's own $P$ is, because
folding accumulates a period error over every cycle in the baseline
(order $1/N_\text{cycles}$ precision is needed just to avoid smearing the
fold into incoherence for a baseline of $N_\text{cycles}$ rotations),
whereas the comb fit only ever evaluates a handful of narrow local
windows and barely notices a comparable error. $P$ must therefore be
locally refined (a fine grid search around the joint fit's own $P$,
minimizing $\theta$) before this diagnostic is meaningful.

**Case study demonstrating why this diagnostic matters** (synthetic test,
not yet observed in real data, but constructed from a real, documented
astrophysical failure mode — the "double-dipping" problem, e.g. Basri &
Nguyen 2018): two unequal starspot groups 180° apart in longitude produce
a light curve that is exactly periodic at both the true period $P$ and at
$P/2$. When the two spot groups have sufficiently different depths (test
case: one 55% the depth of the other), *every* candidate-generation
method's top pick, and the joint comb fit's own reduced $\chi^2$, favor
the wrong ($P/2$) period — by roughly 50$\times$ in $\chi^2$ in the tested
case. This has a structural explanation, not just a numerical
coincidence: at half the true period, the comb-fit windows are also half
as wide, so a comb finds it structurally easier to fit tightly regardless
of whether $P/2$ is the physically correct period. $\theta$, computed on
the full folded light curve, is *not* fooled the same way: it is lower
(better) for the true $P$ than for the $P/2$ alias, because folding on
$P/2$ forces the two unequal-depth dips to overlap into the same phase
bin, which a variance-based statistic penalizes directly. This is a clear
example of a case where the joint fit's own goodness-of-fit is
structurally biased in a way an independent, model-different diagnostic
is not — worth a figure in the eventual paper.

**Composite-spectrum peak prominence** (G$_\text{ACF}$/H$_\text{ACF}$,
following the Santos/ROOSTER convention, e.g. Ceillier et al. 2017): read
directly off the raw ACF (not the fitted parabola) at the true local
maximum nearest a candidate's fitted peak location. $G$ is that maximum's
height; $H$ is its prominence relative to the mean of its two flanking
local minima. This is deliberately independent of the joint fit's own
fitted height (which comes from a parabola constrained by the tied
comb structure and can, in principle, be pulled slightly away from the
ACF's actual local peak) — comparing the two is itself a useful
diagnostic for whether the tied structure is distorting an otherwise
genuine peak.

### 6.5 Documented weak points

For completeness/honesty in an eventual limitations section: sub-10-day
periods remain the pipeline's weakest regime (Section 4.5); very
weak-signal targets (rotation amplitude small relative to noise) are
correctly *not* recovered by the gating logic in Section 6.3 rather than
being force-fit, but that means the pipeline's honest non-detection rate
on such targets has not been separately characterized from its
true failure rate — worth distinguishing carefully in any eventual
completeness/recovery-rate analysis, since "gates correctly triggered on
an undetectable signal" and "pipeline failed on a detectable signal" are
very different claims about the method's performance.

---

## 7. Not yet covered in this document

- The gradient-boosted ranking model itself (architecture, training
  procedure, feature set) — extensive design discussion has happened but
  no training code exists yet as of this writing. Will need its own
  section once implemented; `FEATURE_DOCUMENTATION.txt` is the current
  reference for the feature set specifically.
- Any large-scale (full-corpus) recovery-rate statistics — everything
  above is characterized on small, hand-picked, deliberately-hard test
  samples only (🔴 tags above), not a representative sample of the full
  intended target population.
- Formal completeness/reliability characterization (e.g. recovery rate as
  a function of true period, amplitude, noise level, gap fraction) —
  would be a natural, and probably necessary, figure/table for the
  eventual paper, and hasn't been done systematically yet.

"""
ml_features.py

Turn the output of comb_fit.fit_rotation_period() into a flat, per-candidate
feature table suitable for training a candidate-ranking model (e.g. an
XGBoost `rank:pairwise`/`rank:ndcg` model, grouped by light curve).

See FEATURE_DOCUMENTATION.txt for the full description of every feature
this module produces -- name, definition, rationale, literature references,
and which pipeline stage computes it. This module's docstrings describe
*how* each feature is computed; that file describes *why* each one exists.

Design notes
------------
- One row per successfully-fit candidate (EnsembleResult.candidates already
  excludes candidates that couldn't even be fit -- see fit_rotation_period).
- Three feature groups, computed differently:
    1. Per-candidate, intrinsic (from CandidateResult/CombFitResult/the
       diagnostics dict) -- cheap, already computed by fit_rotation_period.
    2. Per-candidate, cross-candidate/cross-method (agreement, harmonic
       relationships, "is this period recognized by other methods' own
       spectra") -- requires seeing every candidate and every method's
       raw spectrum/histogram for the same star at once.
    3. Star-level (light-curve variability diagnostics) -- computed once
       per star and copied into every row, since every candidate for a
       given star shares the same light curve.
- If you know the true injected period for this light curve (e.g. training
  on SMARTS), pass it in and a `label` column is added directly -- this is
  the only place "truth" enters; nothing about the feature values
  themselves depends on it.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew as _skew, kurtosis as _kurtosis

from comb_fit import EnsembleResult, InitialGuess, _peak_coverage_fraction


# ==========================================================================
# Per-candidate intrinsic features (existing set)
# ==========================================================================

def _height_snr_slope(height_snr: np.ndarray) -> float:
    """Linear-fit slope of height_snr vs. harmonic index n (0, 1, 2, ...).
    A genuine rotation signal typically decays gently with n (spot
    evolution -- see comb_score's default_comb_weight docstring for the
    same reasoning), so this slope is usually mildly negative for real
    candidates and can be erratic (near zero, positive, or steeply
    negative) for spurious ones. Returns NaN if fewer than 2 peaks.
    """
    n = len(height_snr)
    if n < 2:
        return float("nan")
    x = np.arange(n)
    slope, _ = np.polyfit(x, height_snr, 1)
    return float(slope)


def _agreement_and_harmonic_features(
    candidates: list,
    agreement_rel_tol: float,
    harmonic_factors: tuple,
    harmonic_rel_tol: float,
) -> pd.DataFrame:
    """Compute, for every candidate, features that depend on the *other*
    candidates tested for the same light curve.

    Two distinct kinds of "agreement" are captured here, and it matters
    which one you're looking at:

    1. PRE-FIT agreement (n_duplicate_guesses / n_contributing_methods,
       read directly off CandidateResult): fit_rotation_period deduplicates
       near-identical candidate periods *before* fitting (dedup_rel_tol),
       so multiple methods proposing basically the same period collapse
       into one fitted candidate. If we only looked at post-fit periods,
       that corroborating evidence would already be gone by the time we
       get here -- these fields recover it from what fit_rotation_period
       now records about what got merged into each surviving candidate.

    2. POST-FIT agreement (n_agreeing_candidates / n_agreeing_methods,
       computed here by comparing every candidate's FITTED period, fit.P,
       not its pre-fit period): two candidates that started more than
       dedup_rel_tol apart (so neither absorbed the other pre-fit) can
       still converge to nearly the same fitted period once the joint comb
       fit refines both of them -- that's independent, and arguably
       stronger, corroboration than pre-fit agreement, since it survived
       the actual curve fit rather than just a coarse initial estimate.

    Also computed:
    - best_competing_redchi / redchi_ratio_to_best_competing: the lowest
      reduced chi-squared among candidates whose FITTED period is NOT in
      (post-fit) agreement with this one, i.e. how good the best
      genuinely-different rival is.
    - redchi_ratio_to_global_best: this candidate's redchi divided by the
      best redchi among ALL tested candidates for this star (1.0 for the
      overall best, >1 for everything else) -- lets the model use relative
      standing directly instead of only raw redchi, which varies wildly in
      scale star-to-star.
    - is_harmonic_of_global_best / harmonic_factor_of_global_best: whether
      this candidate's fitted period is close to a simple integer ratio
      (0.5x, 2x, 3x, 1/3x, ...) of the fitted period belonging to the
      globally-best-redchi candidate. Directly targets the harmonic-
      confusion failure mode this pipeline has repeatedly run into.
    """
    fitted_periods = np.array([c.fit.P for c in candidates])
    redchis = np.array([
        c.fit.redchi if np.isfinite(c.fit.redchi) else np.inf for c in candidates
    ])
    methods = np.array([c.source_guess.method for c in candidates])

    best_idx = int(np.argmin(redchis))
    global_best_period = fitted_periods[best_idx]
    global_best_redchi = redchis[best_idx]

    rows = []
    for i in range(len(candidates)):
        c = candidates[i]

        # --- post-fit agreement, using FITTED periods ---
        rel_diff = np.abs(fitted_periods - fitted_periods[i]) / fitted_periods[i]
        agree_mask = rel_diff <= agreement_rel_tol
        agree_mask[i] = True  # include self when counting distinct methods

        n_agreeing_candidates = int(agree_mask.sum()) - 1  # exclude self
        n_agreeing_methods = len(set(methods[agree_mask]))

        outside = redchis[~agree_mask]
        best_competing_redchi = float(np.min(outside)) if len(outside) else np.nan
        redchi_ratio_to_best_competing = (
            redchis[i] / best_competing_redchi
            if best_competing_redchi and np.isfinite(best_competing_redchi)
            else np.nan
        )

        redchi_ratio_to_global_best = (
            redchis[i] / global_best_redchi
            if np.isfinite(global_best_redchi) and global_best_redchi > 0
            else np.nan
        )

        # harmonic relationship to the globally-best candidate's FITTED
        # period (skip comparing the best candidate to itself)
        is_harmonic = False
        harmonic_factor = np.nan
        if i != best_idx and global_best_period > 0:
            ratio = fitted_periods[i] / global_best_period
            if abs(ratio - 1.0) > agreement_rel_tol:  # not just "the same period"
                for f in harmonic_factors:
                    if abs(ratio - f) / f <= harmonic_rel_tol:
                        is_harmonic = True
                        harmonic_factor = f
                        break

        rows.append(dict(
            # pre-fit: recovered from what fit_rotation_period's dedup
            # step merged into this candidate before it was ever fit
            n_duplicate_guesses=c.n_duplicate_guesses,
            n_contributing_methods=len(c.contributing_methods),
            # post-fit: independently recomputed here from fitted periods
            n_agreeing_candidates=n_agreeing_candidates,
            n_agreeing_methods=n_agreeing_methods,
            best_competing_redchi=best_competing_redchi,
            redchi_ratio_to_best_competing=redchi_ratio_to_best_competing,
            redchi_ratio_to_global_best=redchi_ratio_to_global_best,
            is_harmonic_of_global_best=is_harmonic,
            harmonic_factor_of_global_best=harmonic_factor,
            is_global_best_redchi=(i == best_idx),
        ))
    return pd.DataFrame(rows)


# ==========================================================================
# Method-spectrum features: evaluate ALL THREE methods' raw evidence at
# each candidate's fitted period, not just whichever method proposed it
# ==========================================================================

def _nearest_peak_stats(x: np.ndarray, y: np.ndarray, target: float) -> Optional[dict]:
    """Given an ascending-x spectrum/histogram (x=period, y=power or
    count), find the local maximum nearest `target` -- among ALL local
    maxima in the array, not just whichever ones became guesses -- and
    return its rank (1 = globally strongest peak in this spectrum),
    fractional height relative to the spectrum's own tallest peak,
    relative distance from `target`, and half-max width. Returns None if
    the array has no local maxima at all (e.g. perfectly flat).
    """
    idx, _ = find_peaks(y)
    if len(idx) == 0:
        return None
    peak_x = x[idx]
    peak_y = y[idx]
    order = np.argsort(peak_y)[::-1]
    ranks = np.empty(len(idx), dtype=int)
    ranks[order] = np.arange(1, len(idx) + 1)

    nearest_local = int(np.argmin(np.abs(peak_x - target)))
    nearest_idx = idx[nearest_local]
    raw_value = float(peak_y[nearest_local])
    y_max = float(np.max(y))

    half = raw_value / 2
    i = nearest_idx
    while i > 0 and y[i] > half:
        i -= 1
    j = nearest_idx
    while j < len(y) - 1 and y[j] > half:
        j += 1

    return dict(
        rank=int(ranks[nearest_local]),
        raw_value=raw_value,
        frac_power=raw_value / y_max if y_max > 0 else np.nan,
        rel_distance=float(abs(peak_x[nearest_local] - target) / target),
        width=float(abs(x[j] - x[i])),
        is_local_max_within_tol=None,  # filled by caller, which knows the tolerance
    )


def _group_guesses_by_method(guesses: list) -> dict:
    """Return one representative InitialGuess per method from a (typically
    pre-dedup) guess pool. All guesses from a single guess_*() call share
    the same underlying spectrum/histogram arrays in `info`, so any one of
    them (the first found) carries everything needed here.
    """
    reps = {}
    for g in guesses:
        if g.method not in reps:
            reps[g.method] = g
    return reps


def _method_spectrum_features(
    fitted_P: float,
    fitted_t0: float,
    method_reps: dict,
    agreement_rel_tol: float,
) -> dict:
    """Evaluate every available method's raw spectrum/histogram at this
    candidate's fitted (P, t0), regardless of which method originally
    proposed the candidate. See FEATURE_DOCUMENTATION.txt for the full
    rationale for each field.
    """
    out = {}
    recognized = 0
    strengths = []

    # --- Lomb-Scargle ---
    ls_guess = method_reps.get("lombscargle")
    if ls_guess is not None:
        periods, power = ls_guess.info["periodogram"]
        stats = _nearest_peak_stats(periods, power, fitted_P)
        if stats is not None:
            out["ls_nearest_peak_rank"] = stats["rank"]
            out["ls_nearest_peak_frac_power"] = stats["frac_power"]
            out["ls_nearest_peak_rel_distance"] = stats["rel_distance"]
            out["ls_nearest_peak_width"] = stats["width"]
            ls_obj = ls_guess.info.get("ls_object")
            if ls_obj is not None and np.isfinite(stats["raw_value"]):
                try:
                    out["ls_nearest_peak_fap"] = float(
                        ls_obj.false_alarm_probability(stats["raw_value"], method="baluev")
                    )
                except Exception:  # noqa: BLE001 -- FAP is a nice-to-have, never fatal
                    out["ls_nearest_peak_fap"] = np.nan
            else:
                out["ls_nearest_peak_fap"] = np.nan
            if stats["rel_distance"] <= agreement_rel_tol:
                recognized += 1
            strengths.append(stats["frac_power"])
        else:
            for k in ("rank", "frac_power", "rel_distance", "width", "fap"):
                out[f"ls_nearest_peak_{k}"] = np.nan
    else:
        for k in ("rank", "frac_power", "rel_distance", "width", "fap"):
            out[f"ls_nearest_peak_{k}"] = np.nan

    # --- ACF FFT ---
    fft_guess = method_reps.get("acf_fft")
    if fft_guess is not None:
        freq, power = fft_guess.info["fft_freq"], fft_guess.info["fft_power"]
        periods = 1.0 / freq
        order = np.argsort(periods)
        periods_sorted, power_sorted = periods[order], power[order]
        stats = _nearest_peak_stats(periods_sorted, power_sorted, fitted_P)
        if stats is not None:
            out["fft_nearest_peak_rank"] = stats["rank"]
            out["fft_nearest_peak_frac_power"] = stats["frac_power"]
            out["fft_nearest_peak_rel_distance"] = stats["rel_distance"]
            out["fft_nearest_peak_width"] = stats["width"]
            noise_floor = float(np.median(power_sorted))
            out["fft_nearest_peak_snr"] = (
                stats["raw_value"] / noise_floor if noise_floor > 0 else np.nan
            )
            if stats["rel_distance"] <= agreement_rel_tol:
                recognized += 1
            strengths.append(stats["frac_power"])
        else:
            for k in ("rank", "frac_power", "rel_distance", "width", "snr"):
                out[f"fft_nearest_peak_{k}"] = np.nan
    else:
        for k in ("rank", "frac_power", "rel_distance", "width", "snr"):
            out[f"fft_nearest_peak_{k}"] = np.nan

    # --- pairwise-spacing histogram ---
    pw_guess = method_reps.get("pairwise_histogram")
    if pw_guess is not None:
        bin_centers, hist = pw_guess.info["histogram"]
        stats = _nearest_peak_stats(bin_centers, hist.astype(float), fitted_P)
        if stats is not None:
            out["hist_nearest_bin_rank"] = stats["rank"]
            out["hist_nearest_bin_frac_count"] = stats["frac_power"]
            out["hist_nearest_bin_rel_distance"] = stats["rel_distance"]
            if stats["rel_distance"] <= agreement_rel_tol:
                recognized += 1
            strengths.append(stats["frac_power"])
        else:
            for k in ("rank", "frac_count", "rel_distance"):
                out[f"hist_nearest_bin_{k}"] = np.nan

        peak_lags = pw_guess.info.get("peak_lags")
        if peak_lags is not None and len(peak_lags) > 0:
            out["coverage_at_fit"] = _peak_coverage_fraction(peak_lags, fitted_P, fitted_t0)
            out["n_acf_peaks_found"] = len(peak_lags)
        else:
            out["coverage_at_fit"] = np.nan
            out["n_acf_peaks_found"] = np.nan
    else:
        for k in ("rank", "frac_count", "rel_distance"):
            out[f"hist_nearest_bin_{k}"] = np.nan
        out["coverage_at_fit"] = np.nan
        out["n_acf_peaks_found"] = np.nan

    out["n_methods_recognize_as_peak"] = recognized
    out["combined_method_strength"] = float(np.mean(strengths)) if strengths else np.nan
    return out


# ==========================================================================
# ACF peak-to-trough depth (per candidate, aggregated across fitted peaks)
# ==========================================================================

def _peak_trough_depth_stats(acf_lags: np.ndarray, acf: np.ndarray, fit, P: float) -> dict:
    """For each fitted comb tooth, find the local minimum of the raw ACF on
    either side (searching out to +/- P/2, i.e. halfway to the neighboring
    tooth) and compute the fitted peak's height above the LOWER of the two
    flanking troughs -- the more conservative (harder to game) choice.
    Aggregated (mean/min/std) across all fitted peaks, in both raw ACF
    units and normalized by the ACF's overall standard deviation (parallel
    to how heights/height_snr are already reported).
    """
    lag_min, lag_max = acf_lags[0], acf_lags[-1]
    depths = []
    for pk in fit.per_peak.values():
        c, h = pk["center"], pk["height"]
        left_mask = (acf_lags >= max(c - 0.5 * P, lag_min)) & (acf_lags <= c)
        right_mask = (acf_lags >= c) & (acf_lags <= min(c + 0.5 * P, lag_max))
        troughs = []
        if left_mask.any():
            troughs.append(float(np.min(acf[left_mask])))
        if right_mask.any():
            troughs.append(float(np.min(acf[right_mask])))
        if troughs:
            depths.append(h - min(troughs))

    depths = np.array(depths)
    if len(depths) == 0:
        return dict(
            mean_peak_trough_depth=np.nan, min_peak_trough_depth=np.nan,
            std_peak_trough_depth=np.nan, mean_peak_trough_depth_snr=np.nan,
            min_peak_trough_depth_snr=np.nan, std_peak_trough_depth_snr=np.nan,
        )

    acf_std = float(np.std(acf))
    depths_snr = depths / acf_std if acf_std > 0 else depths * np.nan
    return dict(
        mean_peak_trough_depth=float(np.mean(depths)),
        min_peak_trough_depth=float(np.min(depths)),
        std_peak_trough_depth=float(np.std(depths)),
        mean_peak_trough_depth_snr=float(np.mean(depths_snr)),
        min_peak_trough_depth_snr=float(np.min(depths_snr)),
        std_peak_trough_depth_snr=float(np.std(depths_snr)),
    )


# ==========================================================================
# S_ph and n_cycles_spanned (per candidate: depend on the candidate's P)
# ==========================================================================

def _s_ph(time: np.ndarray, flux: np.ndarray, P: float, min_bins: int = 5) -> float:
    """Photometric activity index (Mathur et al. 2014): bin the light curve
    into consecutive windows of length P (the CANDIDATE's period, which is
    why this is a per-candidate feature, unlike the star-level variability
    metrics), compute the flux standard deviation within each bin, and
    average across bins. Returns NaN if P is too short or too long to give
    at least `min_bins` usable bins.
    """
    dt = np.median(np.diff(time))
    bin_pts = max(1, int(round(P / dt)))
    n_bins = len(flux) // bin_pts
    if bin_pts < 2 or n_bins < min_bins:
        return float("nan")

    stds = []
    for i in range(n_bins):
        seg = flux[i * bin_pts:(i + 1) * bin_pts]
        seg_valid = seg[np.isfinite(seg)]
        if len(seg_valid) >= max(3, bin_pts // 3):
            stds.append(np.std(seg_valid))
    return float(np.mean(stds)) if stds else float("nan")


# ==========================================================================
# Star-level (light-curve) features: one value per star, copied into every
# candidate row for that star
# ==========================================================================

def _smooth_boxcar(time: np.ndarray, flux: np.ndarray, window_hours: float) -> np.ndarray:
    """NaN-aware centered rolling mean, window expressed in hours and
    converted to points using the light curve's own median cadence.
    """
    dt_days = np.median(np.diff(time))
    window_pts = max(1, int(round((window_hours / 24.0) / dt_days)))
    if window_pts <= 1:
        return flux.copy()
    s = pd.Series(flux).rolling(
        window_pts, center=True, min_periods=max(1, window_pts // 3)
    ).mean()
    return s.to_numpy()


def _r_var(
    time: np.ndarray,
    flux: np.ndarray,
    segments: Optional[list] = None,
    smooth_window_hours: float = 4.0,
) -> float:
    """R_var (Reinhold & Gizon 2015): difference between the 95th and 5th
    percentile of smoothed, normalized flux -- a basic peak-to-peak
    variability amplitude, used in the literature as a pre-filter before
    even attempting a period search.

    `segments`: optional list of (t_start, t_end) tuples. If given, R_var
    is computed independently within each segment (e.g. one per TESS
    sector) and the results averaged -- matching Reinhold & Gizon's
    original per-quarter treatment for Kepler. Currently light curves are
    passed into this pipeline as a single already-combined timeseries, so
    the default (None) treats the whole array as one segment; per-sector
    segmentation will use this same parameter once multi-sector input
    support is added.
    """
    if segments is None:
        segments = [(time[0], time[-1] + 1e-9)]
    vals = []
    for t0, t1 in segments:
        m = (time >= t0) & (time < t1)
        if m.sum() < 10:
            continue
        smoothed = _smooth_boxcar(time[m], flux[m], smooth_window_hours)
        valid = np.isfinite(smoothed)
        if valid.sum() < 10:
            continue
        p5, p95 = np.nanpercentile(smoothed[valid], [5, 95])
        vals.append(p95 - p5)
    return float(np.mean(vals)) if vals else float("nan")


def _flicker(time: np.ndarray, flux: np.ndarray, smooth_window_hours: float = 8.0) -> float:
    """Standard deviation of the flux after smoothing on an ~8 hour
    timescale -- a simplified proxy inspired by the concept of "flicker"
    (Bastien et al. 2013), not a reproduction of their exact 8-hour-binned,
    trend-subtracted definition. See FEATURE_DOCUMENTATION.txt.
    """
    smoothed = _smooth_boxcar(time, flux, smooth_window_hours)
    valid = np.isfinite(smoothed)
    if valid.sum() < 10:
        return float("nan")
    return float(np.nanstd(smoothed[valid]))


def _duty_cycle(time: np.ndarray, flux: np.ndarray) -> float:
    """Fraction of the nominal (evenly-spaced, gap-free) cadence grid that
    actually has finite flux."""
    dt = np.median(np.diff(time))
    n_nominal = int(round((time[-1] - time[0]) / dt)) + 1
    return float(np.isfinite(flux).sum() / n_nominal)


def _von_neumann_eta(time: np.ndarray, flux: np.ndarray) -> float:
    """von Neumann's eta statistic: mean squared successive difference
    divided by the variance. Low (near 0) for smooth/periodic signals
    (successive points correlate), ~2 for pure white noise. Only
    time-adjacent valid pairs are used (pairs spanning a data gap are
    excluded), so gaps don't masquerade as large point-to-point jumps.
    """
    dt = np.median(np.diff(time))
    valid = np.isfinite(flux)
    good_pair = valid[:-1] & valid[1:] & (np.diff(time) <= 1.5 * dt)
    if good_pair.sum() < 10:
        return float("nan")
    diffs = np.diff(flux)[good_pair]
    var = float(np.nanvar(flux[valid]))
    if var == 0:
        return float("nan")
    return float(np.mean(diffs ** 2) / var)


def _fliper_lite(
    time: np.ndarray,
    flux: np.ndarray,
    cutoff_fractions: tuple = (0.5, 0.125, 0.03125),
) -> dict:
    """Simplified, "FliPer-inspired" broadband variability metric (Bugnet
    et al. 2018 define FliPer as noise-subtracted integrated power spectral
    density above several fixed frequency cutoffs, calibrated to Kepler's
    photon-counting noise model). This version substitutes a cheaper proxy
    -- see FEATURE_DOCUMENTATION.txt for the full rationale:

    - Reuses the same masked-FFT-of-the-flux trick as elsewhere in this
      pipeline (gaps zeroed after mean-subtraction) instead of a general
      Lomb-Scargle PSD, since the light curve is evenly sampled -- this is
      an O(N log N) FFT, not a slow arbitrary-frequency-grid computation.
    - Frequency cutoffs are defined as fractions of THIS light curve's own
      Nyquist frequency (so they scale automatically with cadence, rather
      than using Bugnet's fixed physical cutoffs which assume Kepler-like
      sampling), by default at 1/2, 1/8, and 1/32 of Nyquist.
    - Each band is reported as a FRACTION of total power (not an absolute,
      noise-subtracted PSD level), sidestepping the need for an
      instrument-specific analytic noise model. This makes the bands
      dimensionless and roughly comparable star-to-star, at the cost of
      being a genuine simplification of the published metric -- treat this
      as "how is this light curve's variability power distributed across
      timescales", not a reproduction of Bugnet et al.'s FliPer.
    """
    mask = np.isfinite(flux)
    if mask.sum() < 20:
        return {f"fliper_band_{i + 1}": float("nan") for i in range(len(cutoff_fractions))}

    x = np.where(mask, flux - np.mean(flux[mask]), 0.0)
    n = len(x)
    xw = x * np.hanning(n)

    dt = np.median(np.diff(time))
    nfft = 2 * n
    power = np.abs(np.fft.rfft(xw, n=nfft)) ** 2
    freq = np.fft.rfftfreq(nfft, d=dt)
    nyquist = 0.5 / dt

    total_power = float(np.sum(power[1:]))  # exclude DC bin
    out = {}
    for i, frac in enumerate(cutoff_fractions):
        band_power = float(np.sum(power[freq >= frac * nyquist]))
        out[f"fliper_band_{i + 1}"] = band_power / total_power if total_power > 0 else float("nan")
    return out


def _lightcurve_level_features(time: np.ndarray, flux: np.ndarray, sectors: Optional[list] = None) -> dict:
    """All star-level (constant-per-light-curve) features in one call.
    `sectors`: optional list of (t_start, t_end) tuples, forwarded to
    _r_var only for now (see its docstring) -- the other metrics here will
    gain the same per-sector-then-combine treatment once multi-sector
    input support is added to the pipeline.
    """
    valid = np.isfinite(flux)
    flux_valid = flux[valid]
    out = dict(
        r_var=_r_var(time, flux, segments=sectors),
        flicker_std=_flicker(time, flux),
        flux_std=float(np.nanstd(flux)),
        flux_skewness=float(_skew(flux_valid)) if len(flux_valid) > 2 else float("nan"),
        flux_kurtosis=float(_kurtosis(flux_valid)) if len(flux_valid) > 2 else float("nan"),
        duty_cycle=_duty_cycle(time, flux),
        von_neumann_eta=_von_neumann_eta(time, flux),
    )
    out.update(_fliper_lite(time, flux))
    return out


# ==========================================================================
# Top-level extraction function
# ==========================================================================

def extract_candidate_features(
    result: EnsembleResult,
    guesses: list,
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    *,
    star_id: Optional[Union[str, int]] = None,
    n_peaks_requested: Optional[int] = None,
    agreement_rel_tol: float = 0.03,
    harmonic_factors: tuple = (0.5, 2.0, 1.0 / 3.0, 3.0),
    harmonic_rel_tol: float = 0.03,
    sectors: Optional[list] = None,
    true_period: Optional[float] = None,
    label_rel_tol: float = 0.03,
) -> pd.DataFrame:
    """Flatten an EnsembleResult into a one-row-per-candidate feature table.

    Parameters
    ----------
    result : the EnsembleResult from fit_rotation_period(acf_lags, acf, guesses).
    guesses : the FULL candidate pool passed into fit_rotation_period (i.e.
        the pre-dedup list, e.g. straight from gather_initial_guesses()) --
        needed here (not just result.candidates) so every method's raw
        spectrum/histogram is available for evaluating EVERY candidate
        against EVERY method's evidence, not just the method that
        originally proposed it.
    time, flux : the original light curve arrays (used for the light-curve-
        level variability features and per-candidate S_ph).
    acf_lags, acf : same arrays passed to fit_rotation_period.
    star_id : optional identifier attached to every row, for grouping rows
        by light curve (required for grouped train/test splitting and for
        `group` arrays in ranking losses like XGBoost's rank:pairwise).
    n_peaks_requested : the `n_peaks` value passed to fit_rotation_period,
        if you want `n_peaks_used_frac` computed against it rather than
        falling back to n_peaks_used + n_peaks_dropped.
    agreement_rel_tol : relative period tolerance used for (a) treating two
        candidates as "agreeing" and (b) deciding whether a candidate's
        period counts as landing on another method's own local maximum.
    harmonic_factors, harmonic_rel_tol : which integer-ish ratios count as
        "this looks like a harmonic of the best candidate", and how much
        slack to allow.
    sectors : optional list of (t_start, t_end) tuples for per-sector R_var
        (see _r_var). Leave as None for the current single-combined-
        timeseries workflow.
    true_period : if you know the true (e.g. injected) period for this
        light curve, pass it here to add a `label` column: 1 for any
        candidate whose FITTED period (fit.P, the refined joint-fit
        estimate -- not the pre-fit candidate period in the `period`
        column) is within `label_rel_tol` relative difference of the
        truth, 0 otherwise. Leave as None for inference (no labels
        available).
    label_rel_tol : relative tolerance used to build `label` from
        `true_period`.

    Returns
    -------
    pd.DataFrame, one row per candidate in `result.candidates` (i.e. every
    candidate that could be fit at all -- candidates that failed to fit are
    not included, since they have no fit-derived features to report).
    """
    candidates = result.candidates
    if len(candidates) == 0:
        return pd.DataFrame()

    acf_std = float(np.std(acf))
    baseline = float(acf_lags[-1] - acf_lags[0])
    method_reps = _group_guesses_by_method(guesses)
    lc_features = _lightcurve_level_features(time, flux, sectors=sectors)

    # --- per-candidate (intrinsic) features ---
    rows = []
    for c in candidates:
        fit = c.fit
        diag = c.diagnostics
        guess = c.source_guess

        heights = diag["heights"]
        height_snr = diag["height_snr"]
        curvatures = diag["curvatures"]

        n_used = fit.n_peaks_used
        n_dropped = fit.n_peaks_dropped
        denom = n_peaks_requested if n_peaks_requested is not None else (n_used + n_dropped)

        row = dict(
            star_id=star_id,
            period=c.period,
            log_period=np.log10(c.period) if c.period > 0 else np.nan,
            period_frac_of_baseline=c.period / baseline if baseline > 0 else np.nan,
            n_cycles_spanned=baseline / fit.P if fit.P > 0 else np.nan,
            t0=c.t0,

            # candidate-generation provenance
            method=guess.method,
            rank=guess.rank,
            strength=guess.strength,
            coverage_at_generation=guess.info.get("coverage", np.nan),  # pairwise_histogram only, legacy

            # fit point estimates + uncertainties
            fitted_P=fit.P,
            P_err_frac=(fit.P_err / fit.P) if (fit.P_err is not None and fit.P > 0) else np.nan,
            t0_err_frac=(fit.t0_err / fit.P) if (fit.t0_err is not None and fit.P > 0) else np.nan,

            # fit quality
            redchi=fit.redchi,
            log_redchi=np.log10(fit.redchi) if np.isfinite(fit.redchi) and fit.redchi > 0 else np.nan,
            n_peaks_used=n_used,
            n_peaks_dropped=n_dropped,
            n_peaks_used_frac=(n_used / denom) if denom else np.nan,

            # per-peak shape summaries
            frac_positive_heights=diag["frac_positive_heights"],
            frac_non_increasing_height=diag["frac_non_increasing_height"],
            mean_height_snr=float(np.nanmean(height_snr)) if len(height_snr) else np.nan,
            min_height_snr=float(np.nanmin(height_snr)) if len(height_snr) else np.nan,
            max_height_snr=float(np.nanmax(height_snr)) if len(height_snr) else np.nan,
            std_height_snr=float(np.nanstd(height_snr)) if len(height_snr) else np.nan,
            height_snr_slope=_height_snr_slope(height_snr),
            mean_curvature=float(np.nanmean(curvatures)) if len(curvatures) else np.nan,
            std_curvature=float(np.nanstd(curvatures)) if len(curvatures) else np.nan,

            # existing hand-tuned gates, kept as features (cheap, and lets
            # the model learn to override or agree with them)
            passes_min_peaks=diag["passes_min_peaks"],
            passes_redchi=diag["passes_redchi"],
            passes_height_snr=diag["passes_height_snr"],
            passed_gates=c.passed_gates,

            # light-curve-level context (same for every row of this star,
            # but included per-row since each row is an independent
            # training example)
            acf_std=acf_std,
            baseline_days=baseline,
            n_acf_points=len(acf_lags),

            # per-candidate S_ph (bin width = THIS candidate's period)
            s_ph=_s_ph(time, flux, fit.P),
        )
        row.update(_peak_trough_depth_stats(acf_lags, acf, fit, fit.P))
        row.update(_method_spectrum_features(fit.P, fit.t0, method_reps, agreement_rel_tol))
        row.update(lc_features)
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- cross-candidate (relational) features ---
    cross_df = _agreement_and_harmonic_features(
        candidates, agreement_rel_tol, harmonic_factors, harmonic_rel_tol
    )
    df = pd.concat([df, cross_df], axis=1)

    # --- optional label, if truth is known (e.g. SMARTS injected period) ---
    if true_period is not None:
        df["label"] = (
            (np.abs(df["fitted_P"] - true_period) / true_period) <= label_rel_tol
        ).astype(int)

    return df
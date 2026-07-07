"""
ml_features.py

Turn the output of comb_fit.fit_rotation_period() into a flat, per-candidate
feature table suitable for training a candidate-ranking model (e.g. an
XGBoost `rank:pairwise`/`rank:ndcg` model, grouped by light curve).

Design notes
------------
- One row per successfully-fit candidate (EnsembleResult.candidates already
  excludes candidates that couldn't even be fit -- see fit_rotation_period).
- Every feature here is either already sitting in CandidateResult/
  CombFitResult/the diagnostics dict, or a cheap derived scalar (log, ratio,
  slope) of those -- nothing here re-fits anything or touches the ACF beyond
  a couple of O(1) global summary stats (acf_std, baseline).
- Cross-candidate features (the main addition over what fit_rotation_period
  already exposes) require seeing every candidate for the same light curve
  at once, which is why this is a separate pass over `result.candidates`
  rather than something attached per-candidate during fitting itself.
- If you know the true injected period for this light curve (e.g. training
  on SMARTS), pass it in and a `label` column is added directly -- this is
  the only place "truth" enters; nothing about the feature values themselves
  depends on it.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from comb_fit import EnsembleResult, CandidateResult


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


def extract_candidate_features(
    result: EnsembleResult,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    star_id: Optional[Union[str, int]] = None,
    n_peaks_requested: Optional[int] = None,
    agreement_rel_tol: float = 0.03,
    harmonic_factors: tuple = (0.5, 2.0, 1.0 / 3.0, 3.0),
    harmonic_rel_tol: float = 0.03,
    true_period: Optional[float] = None,
    label_rel_tol: float = 0.03,
) -> pd.DataFrame:
    """Flatten an EnsembleResult into a one-row-per-candidate feature table.

    Parameters
    ----------
    result : the EnsembleResult from fit_rotation_period(acf_lags, acf, guesses).
    acf_lags, acf : same arrays passed to fit_rotation_period (used only for
        a couple of O(1) global context features -- ACF noise level and
        baseline length -- not re-fit).
    star_id : optional identifier attached to every row. Set this to
        something that lets you group rows by light curve later (required
        for grouped train/test splitting and for `group` arrays in ranking
        losses like XGBoost's rank:pairwise) -- e.g. a SMARTS catalog ID.
    n_peaks_requested : the `n_peaks` value you passed to fit_rotation_period
        for this call, if you want `n_peaks_used_frac` (fraction of
        requested windows that survived fitting) as a feature. If omitted,
        the denominator falls back to n_peaks_used + n_peaks_dropped, which
        is usually the same but can differ if some windows were skipped
        entirely during window-building (e.g. ran off the end of the ACF's
        lag range) rather than dropped during RANSAC-style rejection.
    agreement_rel_tol : relative period tolerance for two candidates to
        count as "agreeing" (see _agreement_and_harmonic_features).
    harmonic_factors, harmonic_rel_tol : which integer-ish ratios count as
        "this looks like a harmonic of the best candidate", and how much
        slack to allow.
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
            t0=c.t0,

            # candidate-generation provenance
            method=guess.method,
            rank=guess.rank,
            strength=guess.strength,
            coverage=guess.info.get("coverage", np.nan),  # pairwise_histogram only

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
        )
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

"""
plotting.py

Diagnostic plots for the comb_fit.py pipeline:
    - the ACF itself, with detected peaks and/or comb teeth marked
    - the pairwise-spacing histogram (guess_pairwise_histogram)
    - the Lomb-Scargle periodogram (guess_lombscargle)
    - the ACF's FFT power spectrum (guess_acf_fft)
    - the final joint comb fit overplotted on the ACF

Every function takes an existing matplotlib Axes via `ax=...` (creating one
if not given) and returns (fig, ax), so plots can be composed into your own
multi-panel figures or saved/customized afterward.

These functions read directly from the InitialGuess.info dict and
CombFitResult objects produced by comb_fit.py -- no re-computation needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from comb_fit import InitialGuess, CombFitResult


# --------------------------------------------------------------------------
# ACF with peaks / comb teeth marked
# --------------------------------------------------------------------------

def plot_acf(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    peak_lags: Optional[np.ndarray] = None,
    comb_P: Optional[float] = None,
    comb_t0: Optional[float] = None,
    n_teeth: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "ACF",
):
    """Plot the ACF, optionally marking:
      - `peak_lags`: individual detected peaks (e.g. from find_peaks), as
        scatter points, and/or
      - a comb defined by (comb_P, comb_t0): vertical dashed lines at
        comb_t0 + n*comb_P, for a quick visual check of a candidate period
        against the raw ACF before/instead of running the full fit.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.figure

    ax.plot(acf_lags, acf, lw=1, color="0.35", zorder=1)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)

    if peak_lags is not None and len(peak_lags) > 0:
        peak_vals = np.interp(peak_lags, acf_lags, acf)
        ax.scatter(
            peak_lags, peak_vals, color="firebrick", s=28, zorder=3,
            label="detected peaks",
        )

    if comb_P is not None and comb_t0 is not None:
        if n_teeth is None:
            n_teeth = int((acf_lags[-1] - comb_t0) / comb_P) + 1
        teeth = comb_t0 + np.arange(0, max(n_teeth, 0)) * comb_P
        teeth = teeth[teeth <= acf_lags[-1]]
        for i, t in enumerate(teeth):
            ax.axvline(
                t, color="steelblue", lw=1, ls="--", alpha=0.7,
                zorder=2, label="comb teeth" if i == 0 else None,
            )

    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.set_title(title)
    if peak_lags is not None or comb_P is not None:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# Pairwise spacing histogram (guess_pairwise_histogram)
# --------------------------------------------------------------------------

def plot_pairwise_spacing_histogram(
    guess: InitialGuess,
    ax: Optional[plt.Axes] = None,
    title: str = "Pairwise peak-spacing histogram",
):
    """Plot the histogram of pairwise ACF-peak spacings used by
    guess_pairwise_histogram, with candidate periods (local maxima) marked
    and the chosen fundamental period highlighted.
    """
    if guess.method != "pairwise_histogram":
        raise ValueError(
            f"plot_pairwise_spacing_histogram expects a guess from "
            f"guess_pairwise_histogram, got method='{guess.method}'."
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.figure

    bin_centers, hist = guess.info["histogram"]
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
    ax.bar(bin_centers, hist, width=bin_width * 0.9, color="0.75", edgecolor="0.5")

    # ranked_candidates: [(coverage, score, P, t0), ...] sorted best-first
    for coverage, score, P_cand, t0_cand in guess.info["ranked_candidates"]:
        is_best = np.isclose(P_cand, guess.P0)
        ax.axvline(
            P_cand,
            color="firebrick" if is_best else "steelblue",
            lw=2 if is_best else 1,
            ls="-" if is_best else "--",
            alpha=1.0 if is_best else 0.6,
            label=(f"chosen P0={guess.P0:.3f} (coverage={coverage:.2f})" if is_best
                   else None),
        )

    ax.set_xlabel("pairwise peak spacing")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# Lomb-Scargle periodogram (guess_lombscargle)
# --------------------------------------------------------------------------

def plot_lombscargle_periodogram(
    guess: InitialGuess,
    ax: Optional[plt.Axes] = None,
    title: str = "Lomb-Scargle periodogram",
    log_x: bool = True,
):
    """Plot the Lomb-Scargle power spectrum (as a function of period), with
    the top candidate peaks marked and the chosen period highlighted.
    """
    if guess.method != "lombscargle":
        raise ValueError(
            f"plot_lombscargle_periodogram expects a guess from "
            f"guess_lombscargle, got method='{guess.method}'."
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.figure

    periods, power = guess.info["periodogram"]
    ax.plot(periods, power, lw=1, color="0.35")

    cand_periods = guess.info["candidate_periods"]
    cand_powers = guess.info["candidate_powers"]
    ax.scatter(cand_periods, cand_powers, color="steelblue", s=28, zorder=3,
               label="candidate peaks")
    ax.axvline(guess.P0, color="firebrick", lw=2,
               label=f"chosen P0={guess.P0:.3f}")

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("period")
    ax.set_ylabel("LS power")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# ACF FFT power spectrum (guess_acf_fft)
# --------------------------------------------------------------------------

def plot_acf_fft_spectrum(
    guess: InitialGuess,
    ax: Optional[plt.Axes] = None,
    title: str = "FFT of ACF",
    log_x: bool = True,
):
    """Plot the FFT power spectrum of the ACF (as a function of period =
    1/frequency), with candidate peaks marked and the chosen period
    highlighted.
    """
    if guess.method != "acf_fft":
        raise ValueError(
            f"plot_acf_fft_spectrum expects a guess from guess_acf_fft, "
            f"got method='{guess.method}'."
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.figure

    freq = guess.info["fft_freq"]
    power = guess.info["fft_power"]
    periods = 1.0 / freq
    order = np.argsort(periods)
    ax.plot(periods[order], power[order], lw=1, color="0.35")

    cand_periods = guess.info["candidate_periods"]
    cand_powers = guess.info["candidate_powers"]
    ax.scatter(cand_periods, cand_powers, color="steelblue", s=28, zorder=3,
               label="candidate peaks")
    ax.axvline(guess.P0, color="firebrick", lw=2,
               label=f"chosen P0={guess.P0:.3f}")

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("period")
    ax.set_ylabel("FFT power")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


def plot_initial_guess(guess: InitialGuess, ax: Optional[plt.Axes] = None):
    """Dispatch to the right guess-specific plot based on guess.method, so
    you can call one function regardless of which guess_* was used."""
    dispatch = {
        "pairwise_histogram": plot_pairwise_spacing_histogram,
        "lombscargle": plot_lombscargle_periodogram,
        "acf_fft": plot_acf_fft_spectrum,
    }
    if guess.method not in dispatch:
        raise ValueError(f"No plot function registered for method '{guess.method}'.")
    return dispatch[guess.method](guess, ax=ax)


# --------------------------------------------------------------------------
# Final comb fit overplotted on the ACF
# --------------------------------------------------------------------------

def plot_comb_fit(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    fit: CombFitResult,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    shade_windows: bool = True,
    n_model_points: int = 200,
):
    """Plot the ACF with the final joint comb-fit model overplotted: each
    fitted parabola drawn across its fitting window, fitted centers marked,
    and (optionally) the fitting windows themselves shaded so it's clear
    which points actually entered the fit.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(acf_lags, acf, lw=1, color="0.35", zorder=1, label="ACF")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)

    params = fit.lmfit_result.params
    for i, w in enumerate(fit.windows):
        n = w.n
        c = params[f"center_{n}"].value
        A = params[f"A_{n}"].value
        h = params[f"h_{n}"].value

        if shade_windows:
            ax.axvspan(w.lag_lo, w.lag_hi, color="steelblue", alpha=0.08, zorder=0)

        lag_model = np.linspace(w.lag_lo, w.lag_hi, n_model_points)
        model = h - A * (lag_model - c) ** 2
        ax.plot(
            lag_model, model, color="firebrick", lw=2, zorder=3,
            label="fitted parabolae" if i == 0 else None,
        )
        ax.scatter([c], [h], color="firebrick", s=35, zorder=4,
                   marker="x", label="fitted centers" if i == 0 else None)

    n_dropped = fit.n_peaks_dropped
    # P_err from a robust-loss (soft_l1/huber/...) fit is only a rough
    # covariance-based approximation and can occasionally come out
    # nonsensically large or NaN (see the note in comb_fit.fit_rotation_period).
    # Only display it when it looks like a plausible fraction of P itself.
    show_err = (
        fit.P_err is not None
        and np.isfinite(fit.P_err)
        and fit.P_err < 0.5 * fit.P
    )
    if title is None:
        title = (
            f"Joint comb fit: P = {fit.P:.4f}"
            + (f" \u00b1 {fit.P_err:.4f}" if show_err else "")
            + f"   (n_peaks used={fit.n_peaks_used}, dropped={n_dropped}, "
              f"redchi={fit.redchi:.3g})"
        )
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# One-call combined diagnostic figure
# --------------------------------------------------------------------------

def plot_full_diagnostic(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    guess: InitialGuess,
    fit: Optional[CombFitResult] = None,
    figsize=(10, 9),
):
    """Convenience wrapper: a stacked figure with (1) the raw ACF with
    detected peaks/comb teeth from the initial guess, (2) the method-specific
    periodogram/histogram panel, and (3) the final comb fit overplotted on
    the ACF (if `fit` is provided). Returns (fig, axes).
    """
    n_panels = 3 if fit is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize)

    peak_lags = guess.info.get("peak_lags")  # only present for pairwise_histogram
    plot_acf(
        acf_lags, acf, peak_lags=peak_lags, comb_P=guess.P0, comb_t0=guess.t0,
        ax=axes[0], title=f"ACF with initial guess ({guess.method})",
    )
    plot_initial_guess(guess, ax=axes[1])

    if fit is not None:
        plot_comb_fit(acf_lags, acf, fit, ax=axes[2])

    fig.tight_layout()
    return fig, axes

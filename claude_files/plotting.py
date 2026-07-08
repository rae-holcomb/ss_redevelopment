"""
plotting.py

Diagnostic plots for the comb_fit.py (v2) pipeline:
    - the ACF itself, with detected peaks and/or comb teeth marked
    - the specific ACF peaks used by guess_pairwise_histogram
    - the pairwise-spacing histogram, with all candidates from one
      guess_pairwise_histogram() call marked
    - the Lomb-Scargle periodogram, with all candidates from one
      guess_lombscargle() call marked
    - the ACF's FFT power spectrum, with all candidates from one
      guess_acf_fft() call marked
    - the candidate-ranking landscape from fit_rotation_period()
    - the final joint comb fit overplotted on the ACF

Every function takes an existing matplotlib Axes via `ax=...` (creating one
if not given) and returns (fig, ax), so plots can be composed into your own
multi-panel figures or saved/customized afterward.

Note on the v2 API: guess_pairwise_histogram/guess_lombscargle/guess_acf_fft
each return a LIST of InitialGuess (their top n_guesses candidates), not a
single one. The plotting functions below that visualize a single method's
output (plot_pairwise_histogram_peaks, plot_pairwise_spacing_histogram,
plot_lombscargle_periodogram, plot_acf_fft_spectrum) all accept either that
list or a single InitialGuess, and mark every candidate in the list (rank 1
highlighted) since they all share the same underlying histogram/periodogram
computed once per call.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from comb_fit import InitialGuess, CombFitResult, EnsembleResult, CandidateResult


def _as_list(guesses: Union[InitialGuess, list]) -> list:
    return [guesses] if isinstance(guesses, InitialGuess) else list(guesses)


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
        against the raw ACF.
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
# Peaks used by guess_pairwise_histogram
# --------------------------------------------------------------------------

def plot_pairwise_histogram_peaks(
    guesses: Union[InitialGuess, list],
    acf_lags: np.ndarray,
    acf: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """Highlight the specific ACF peaks that guess_pairwise_histogram used
    to build its pairwise-spacing histogram (i.e. the x_1, x_2, ..., x_m in
    that function's docstring). Every candidate period it returns is
    derived entirely from the pattern of spacings between these peaks, so
    if the final period guess looks wrong, this plot is the first place to
    look: are these actually the right peaks? Is one obviously spurious
    (noise) or is a real one missing (too low prominence)?

    Accepts either a single InitialGuess or the full list returned by one
    guess_pairwise_histogram() call (both carry the same peak_lags/
    peak_heights in `info`, computed once per call).
    """
    guesses = _as_list(guesses)
    g0 = guesses[0]
    if g0.method != "pairwise_histogram":
        raise ValueError(
            f"plot_pairwise_histogram_peaks expects guesses from "
            f"guess_pairwise_histogram, got method='{g0.method}'."
        )
    peak_lags = g0.info["peak_lags"]
    peak_heights = g0.info["peak_heights"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))
    else:
        fig = ax.figure

    ax.plot(acf_lags, acf, lw=1, color="0.35", zorder=1, label="ACF")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.scatter(
        peak_lags, peak_heights, color="firebrick", s=45, zorder=3,
        edgecolor="white", linewidth=0.6,
        label=f"peaks used ({len(peak_lags)} found)",
    )
    for i, (x, y) in enumerate(zip(peak_lags, peak_heights)):
        ax.annotate(str(i + 1), (x, y), textcoords="offset points",
                    xytext=(0, 8), fontsize=7, ha="center", color="firebrick")

    if title is None:
        title = (
            f"ACF peaks feeding guess_pairwise_histogram "
            f"({len(peak_lags)} peaks -> {len(peak_lags) * (len(peak_lags) - 1) // 2} pairwise spacings)"
        )
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# Pairwise spacing histogram (guess_pairwise_histogram)
# --------------------------------------------------------------------------

def plot_pairwise_spacing_histogram(
    guesses: Union[InitialGuess, list],
    ax: Optional[plt.Axes] = None,
    title: str = "Pairwise peak-spacing histogram",
):
    """Plot the histogram of pairwise ACF-peak spacings used by
    guess_pairwise_histogram, marking every candidate returned by that
    call (rank 1, the strongest by support count, highlighted in red).
    """
    guesses = _as_list(guesses)
    g0 = guesses[0]
    if g0.method != "pairwise_histogram":
        raise ValueError(
            f"plot_pairwise_spacing_histogram expects guesses from "
            f"guess_pairwise_histogram, got method='{g0.method}'."
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.figure

    bin_centers, hist = g0.info["histogram"]
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
    ax.bar(bin_centers, hist, width=bin_width * 0.9, color="0.75", edgecolor="0.5")

    for g in guesses:
        is_best = g.rank == 1
        ax.axvline(
            g.P0,
            color="firebrick" if is_best else "steelblue",
            lw=2 if is_best else 1,
            ls="-" if is_best else "--",
            alpha=1.0 if is_best else 0.6,
            label=(f"rank 1: P0={g.P0:.3f} (support={g.info['support_count']})" if is_best
                   else None),
        )

    ax.set_xlabel("pairwise peak spacing")
    ax.set_ylabel("count (pairs supporting that spacing)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# Lomb-Scargle periodogram (guess_lombscargle)
# --------------------------------------------------------------------------

def plot_lombscargle_periodogram(
    guesses: Union[InitialGuess, list],
    ax: Optional[plt.Axes] = None,
    title: str = "Lomb-Scargle periodogram",
    log_x: bool = True,
):
    """Plot the Lomb-Scargle power spectrum (as a function of period),
    marking every candidate returned by one guess_lombscargle() call
    (rank 1 highlighted).
    """
    guesses = _as_list(guesses)
    g0 = guesses[0]
    if g0.method != "lombscargle":
        raise ValueError(
            f"plot_lombscargle_periodogram expects guesses from "
            f"guess_lombscargle, got method='{g0.method}'."
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.figure

    periods, power = g0.info["periodogram"]
    ax.plot(periods, power, lw=1, color="0.35")

    for g in guesses:
        is_best = g.rank == 1
        ax.scatter([g.P0], [g.strength],
                   color="firebrick" if is_best else "steelblue",
                   s=60 if is_best else 28, zorder=3,
                   label=(f"rank 1: P0={g.P0:.3f} (power={g.strength:.3f})" if is_best
                          else ("other candidates" if g.rank == 2 else None)))

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
    guesses: Union[InitialGuess, list],
    ax: Optional[plt.Axes] = None,
    title: str = "FFT of ACF",
    log_x: bool = True,
):
    """Plot the FFT power spectrum of the ACF (as a function of period),
    marking every candidate returned by one guess_acf_fft() call (rank 1
    highlighted).
    """
    guesses = _as_list(guesses)
    g0 = guesses[0]
    if g0.method != "acf_fft":
        raise ValueError(
            f"plot_acf_fft_spectrum expects guesses from guess_acf_fft, "
            f"got method='{g0.method}'."
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.figure

    freq = g0.info["fft_freq"]
    power = g0.info["fft_power"]
    periods = 1.0 / freq
    order = np.argsort(periods)
    ax.plot(periods[order], power[order], lw=1, color="0.35")

    power_max = float(np.max(power)) if len(power) else 1.0
    for g in guesses:
        is_best = g.rank == 1
        ax.scatter([g.P0], [g.strength * power_max],
                   color="firebrick" if is_best else "steelblue",
                   s=60 if is_best else 28, zorder=3,
                   label=(f"rank 1: P0={g.P0:.3f}" if is_best
                          else ("other candidates" if g.rank == 2 else None)))

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("period")
    ax.set_ylabel("FFT power")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


def plot_wavelet_spectrum(
    guesses: Union[InitialGuess, list],
    ax: Optional[plt.Axes] = None,
    title: str = "Global Wavelet Power Spectrum",
    log_x: bool = True,
):
    """Plot the Global Wavelet Power Spectrum (GWPS) -- the time-averaged
    projection of the light curve's wavelet power surface -- marking every
    candidate returned by one guess_wavelet() call (rank 1 highlighted),
    each as a Gaussian fitted in log-period space (see guess_wavelet's
    docstring for the iterative peak-extraction procedure).

    A second, small panel below shows the full 2D wavelet power surface
    (period vs. time) for the rank-1 candidate's neighborhood, which is
    the one piece of information this method has access to that none of
    the other guess_* functions do: whether the periodicity is present
    throughout the baseline or only part of it.
    """
    guesses = _as_list(guesses)
    g0 = guesses[0]
    if g0.method != "wavelet":
        raise ValueError(
            f"plot_wavelet_spectrum expects guesses from guess_wavelet, "
            f"got method='{g0.method}'."
        )

    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[2, 1.2])
        ax, ax2 = axes
    else:
        fig = ax.figure
        ax2 = None

    periods = g0.info["periods"]
    gwps = g0.info["gwps"]
    ax.plot(periods, gwps, lw=1, color="0.35")

    for g in guesses:
        is_best = g.rank == 1
        peak = [p for p in g0.info["fitted_gaussians"]
                if np.isclose(np.exp(p["center_log_period"]), g.P0)]
        height = peak[0]["height"] if peak else g.strength * float(np.max(gwps))
        ax.scatter([g.P0], [height],
                   color="firebrick" if is_best else "steelblue",
                   s=60 if is_best else 28, zorder=3,
                   label=(f"rank 1: P0={g.P0:.3f}" if is_best
                          else ("other candidates" if g.rank == 2 else None)))

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("period")
    ax.set_ylabel("wavelet power (time-averaged)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    if ax2 is not None and "wavelet_power" in g0.info:
        wavelet_power = g0.info["wavelet_power"]
        time_idx = np.arange(wavelet_power.shape[1])
        # Robust color scaling: a handful of very-short-period rows can have
        # much larger power than the astrophysically interesting range,
        # which would otherwise wash out everything else under a linear
        # color scale spanning the full min/max.
        vmax = np.percentile(wavelet_power, 99.5)
        mesh = ax2.pcolormesh(
            time_idx, periods, wavelet_power, shading="auto",
            cmap="viridis", vmin=0, vmax=vmax,
        )
        if log_x:
            ax2.set_yscale("log")
        ax2.axhline(guesses[0].P0, color="white", lw=1, ls="--", alpha=0.8)
        ax2.set_xlabel("time index")
        ax2.set_ylabel("period")
        ax2.set_title("wavelet power surface (period vs. time)", fontsize=9)
        fig.colorbar(mesh, ax=ax2, label="power", pad=0.01)

    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# Phase-folded light curve (phase_dispersion_stat's own view of the data)
# --------------------------------------------------------------------------

def plot_phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    P: float,
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """Plot the light curve folded on a candidate period P, with the
    phase-bin means +/- standard deviations overplotted -- the direct,
    model-free visual counterpart of phase_dispersion_stat's theta number.
    A period with low theta should look like a coherent phased shape here;
    a period with high theta (e.g. a bad harmonic) will look like scatter.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]

    phase = np.mod(time, P) / P

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    ax.scatter(phase, flux, s=4, color="0.6", alpha=0.5, zorder=1, label="folded data")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.clip(np.digitize(phase, bin_edges) - 1, 0, n_bins - 1)
    means = np.array([np.mean(flux[bin_idx == j]) if np.any(bin_idx == j) else np.nan
                       for j in range(n_bins)])
    stds = np.array([np.std(flux[bin_idx == j]) if np.any(bin_idx == j) else np.nan
                      for j in range(n_bins)])
    ax.errorbar(bin_centers, means, yerr=stds, fmt="o-", color="firebrick",
                lw=1.5, capsize=3, zorder=3, label="phase-bin mean +/- std")

    from comb_fit import phase_dispersion_stat
    theta = phase_dispersion_stat(time, flux, P, n_bins=n_bins)

    if title is None:
        title = f"Phase-folded light curve, P={P:.4f}  (PDM theta={theta:.3f})"
    ax.set_xlabel("phase")
    ax.set_ylabel("flux")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


def plot_candidate_guesses(guesses: Union[InitialGuess, list], ax: Optional[plt.Axes] = None):
    """Dispatch to the right method-specific plot based on the guesses'
    method, so you can call one function regardless of which guess_* they
    came from. All guesses passed in one call must share the same method
    (e.g. the output of a single guess_lombscargle() call)."""
    guesses = _as_list(guesses)
    dispatch = {
        "pairwise_histogram": plot_pairwise_spacing_histogram,
        "lombscargle": plot_lombscargle_periodogram,
        "acf_fft": plot_acf_fft_spectrum,
        "wavelet": plot_wavelet_spectrum,
    }
    method = guesses[0].method
    if method not in dispatch:
        raise ValueError(f"No plot function registered for method '{method}'.")
    return dispatch[method](guesses, ax=ax)


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
# Candidate-ranking landscape (fit_rotation_period's EnsembleResult)
# --------------------------------------------------------------------------

def plot_candidate_ranking(
    result: EnsembleResult,
    ax: Optional[plt.Axes] = None,
    title: str = "Candidate period ranking (fit_rotation_period)",
    log_x: bool = True,
    top_n_labeled: int = 3,
):
    """Plot every candidate period fit_rotation_period actually fit,
    against its joint-fit reduced chi-squared, colored by whether it
    passed the three reliability gates (see fit_rotation_period's
    docstring). If result.success is False, the "winner" marker instead
    shows the closest (but not reliable) attempt, and the title says so.

    Mainly an ambiguity-detection tool: if a close runner-up has a similar
    redchi to the winner, that's worth a second look.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    periods = np.array([c.period for c in result.candidates])
    redchis = np.array([
        c.fit.redchi if np.isfinite(c.fit.redchi) else np.nan
        for c in result.candidates
    ])
    passed = np.array([c.passed_gates for c in result.candidates])

    ax.scatter(periods[~passed], redchis[~passed], color="0.7", s=20,
               label="rejected (gate failed)", zorder=2)
    ax.scatter(periods[passed], redchis[passed], color="steelblue", s=28,
               label="passed gate", zorder=3)

    if result.best_fit is not None:
        marker = "*" if result.success else "X"
        label = "winner" if result.success else "closest attempt (unreliable)"
        ax.scatter([result.best_fit.P],
                   [result.best_fit.redchi if np.isfinite(result.best_fit.redchi) else np.nan],
                   color="firebrick", s=110, marker=marker, zorder=5, label=label)

    for c in sorted(result.candidates, key=lambda c: c.fit.redchi)[:top_n_labeled]:
        redchi = c.fit.redchi if np.isfinite(c.fit.redchi) else None
        if redchi is not None:
            ax.annotate(
                f"{c.fit.P:.3f}", (c.period, redchi),
                textcoords="offset points", xytext=(4, 4), fontsize=8,
            )

    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("candidate period")
    ax.set_ylabel("joint-fit reduced chi-squared (log)")
    full_title = title if result.success else title + "  [NO RELIABLE PERIOD FOUND]"
    ax.set_title(full_title, fontsize=10, color=("black" if result.success else "firebrick"))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# One-call combined diagnostic figure
# --------------------------------------------------------------------------

def plot_full_diagnostic(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    guesses: list,
    result: Optional[EnsembleResult] = None,
    figsize=(10, 15),
):
    """Convenience wrapper: a stacked figure with (1) the raw ACF with the
    best candidate's comb overlaid (if a result is given) or just the ACF
    otherwise, (2) the pairwise-spacing histogram, (3) the Lomb-Scargle
    periodogram, (4) the ACF FFT spectrum -- each showing every candidate
    from its respective method -- (5) the candidate-ranking landscape, and
    (6) the final joint comb fit (if `result.success`). Returns (fig, axes).

    Parameters
    ----------
    guesses : the full candidate pool, e.g. the output of
        gather_initial_guesses() -- a flat list mixing all methods present.
    result : the EnsembleResult from fit_rotation_period(acf_lags, acf,
        guesses), if you've already run it. If omitted, only the
        candidate-generation panels (1-4) are shown.
    """
    by_method = {}
    for g in guesses:
        by_method.setdefault(g.method, []).append(g)

    panel_fns = []
    if "pairwise_histogram" in by_method:
        panel_fns.append(("pairwise_histogram", plot_pairwise_spacing_histogram))
    if "lombscargle" in by_method:
        panel_fns.append(("lombscargle", plot_lombscargle_periodogram))
    if "acf_fft" in by_method:
        panel_fns.append(("acf_fft", plot_acf_fft_spectrum))
    if "wavelet" in by_method:
        panel_fns.append(("wavelet", plot_wavelet_spectrum))

    n_panels = 1 + len(panel_fns) + (2 if result is not None else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # Panel 1: ACF overview
    best = result.best_fit if result is not None else None
    plot_acf(
        acf_lags, acf,
        comb_P=best.P if best is not None else None,
        comb_t0=best.t0 if best is not None else None,
        ax=axes[0],
        title="ACF" + (f" with best candidate (P={best.P:.3f})" if best is not None else ""),
    )

    # Panels 2..: one per method present, always shown
    for i, (method, fn) in enumerate(panel_fns, start=1):
        fn(by_method[method], ax=axes[i])

    idx = 1 + len(panel_fns)
    if result is not None:
        plot_candidate_ranking(result, ax=axes[idx])
        idx += 1
        if result.success:
            plot_comb_fit(acf_lags, acf, result.best_fit, ax=axes[idx])
        else:
            axes[idx].text(
                0.5, 0.5,
                "No reliable rotation period found.\n\n" + result.message,
                ha="center", va="center", transform=axes[idx].transAxes,
                fontsize=9, color="firebrick", wrap=True,
            )
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

    fig.tight_layout()
    return fig, axes

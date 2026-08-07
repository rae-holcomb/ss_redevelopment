"""
comb_fit.py

Stage 2 of the rotation-period pipeline: joint comb fitting and
multi-candidate arbitration.

Candidate PERIODS are proposed elsewhere (guesses.py -- see its module
docstring for the full Stage 1 discussion: guess_pairwise_histogram,
guess_lombscargle, guess_acf_fft, their short-period-focused variants, and
guess_wavelet). This module is where the real evaluation happens: every
candidate from every method is fed through the same joint least-squares
comb-of-parabolae fit against the actual ACF, and the results are compared
on equal footing. This is the only stage that looks at how well a
candidate's predicted peaks actually match the ACF's shape, height, and
spacing simultaneously -- which is a much stronger test than any single
candidate-generation heuristic, and it's why candidate generation doesn't
need to be clever or "correct" on its own: it just needs to not leave the
right answer off the list.

fit_rotation_period (the main entry point here) is also deliberately
willing to say "I couldn't find a reliable period" (EnsembleResult.success
= False) rather than always returning its best guess. A best guess that
didn't clear basic plausibility checks (enough peaks, peaks that are
genuinely positive bumps rather than noise, peaks tall enough relative to
the ACF's noise floor) is often worse than no answer at all.

This module also owns phase-seeding (_grid_search_t0 and its dependencies,
comb_score/default_comb_weight) and the sanity-filtering helper
(_teeth_count), even though they're relatively small utility functions --
they conceptually belong to fitting/arbitration (picking a starting phase
for, and sanity-checking, a candidate that's actually about to be fit), not
to candidate generation, which never computes a phase at all. See
guesses.py's InitialGuess docstring for the other half of that boundary.

This module was split out of what used to be a single, much longer
comb_fit.py, specifically to keep each file to a manageable size as the
pipeline grew (wavelet and short-period-focused candidate generation in
guesses.py were later additions that pushed the combined file past a
reasonable length). See guesses.py's module docstring for the Stage 1 half
of this same design discussion.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Callable, Optional, Union

import numpy as np
from scipy.optimize import least_squares as scipy_least_squares

from guesses import InitialGuess


# --------------------------------------------------------------------------
# Phase-seeding and sanity-filtering utilities
# --------------------------------------------------------------------------
#
# Used by fit_rotation_period (below) to pick a starting phase for a
# candidate that doesn't already have one, and to discard candidates too
# short or too long to plausibly fit. See this module's docstring for why
# these live here rather than in guesses.py, despite superficially looking
# like generic "ACF utilities".

def _teeth_count(P: float, t0: float, lag_max: float) -> int:
    """Number of comb teeth (n=0,1,2,...) that land within [t0, lag_max]
    for period P. Used only as a cheap sanity filter (e.g. discard a
    candidate period so long that fewer than ~2 teeth would even fit in the
    ACF's lag range) -- not for ranking candidates against each other.
    """
    if P <= 0:
        return 0
    return int(np.floor((lag_max - t0) / P)) + 1


def default_comb_weight(n: int) -> float:
    """Default per-tooth weight for comb_score: 1/(n+1). Teeth at low n
    (short lags) are weighted more heavily than teeth at high n (long
    lags). This reflects a real physical expectation: starspots evolve
    (grow, decay, migrate in longitude) on timescales that are often not
    much longer than the rotation period itself, so the periodic signal
    typically becomes less coherent -- and the corresponding ACF peaks
    genuinely weaker and less trustworthy -- at longer lags. Weighting the
    early teeth more heavily makes phase/period estimates rely more on the
    part of the ACF where the periodic signal is most likely to still look
    like the star's actual current rotation, rather than being pulled
    around by whatever noise happens to be doing at lag 10P.
    """
    return 1.0 / (n + 1)


def comb_score(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P: float,
    t0: float,
    n_max: Optional[int] = None,
    lag_max: Optional[float] = None,
    weight_func: Optional[Callable[[int], float]] = None,
) -> float:
    """Score how well a comb of teeth at t0 + n*P (n=0,1,2,...) lines up
    with tall points of the ACF. Larger is better.

    This is a *weighted* average of the (linearly-interpolated) ACF value
    at each comb tooth, using `weight_func(n)` as the weight for tooth n
    (default: default_comb_weight, i.e. 1/(n+1) -- see its docstring for
    why). It is intentionally cheap (no fitting) and is used only for two
    things in this module: (a) picking the best phase t0 for a given
    candidate period during _grid_search_t0, and (b) optional diagnostics.
    It is NOT used to rank candidate periods against each other -- that's
    what the actual joint comb fit in fit_rotation_period is for.
    """
    if weight_func is None:
        weight_func = default_comb_weight
    if lag_max is None:
        lag_max = acf_lags[-1]
    if n_max is None:
        n_max = int(np.floor((lag_max - t0) / P)) if P > 0 else 0
    n_max = max(n_max, 0)

    n = np.arange(0, n_max + 1)
    comb_lags = t0 + n * P
    in_range = comb_lags <= lag_max
    comb_lags = comb_lags[in_range]
    n = n[in_range]
    if len(comb_lags) == 0:
        return -np.inf

    vals = np.interp(comb_lags, acf_lags, acf)
    weights = np.array([weight_func(int(nn)) for nn in n])
    return float(np.sum(weights * vals) / np.sum(weights))


def _short_period_group_key(method: str) -> Optional[str]:
    """Map a candidate's method string to a short-period-family group key,
    or None if it doesn't belong to one of the three known short-period-
    focused families. Used by cap_short_period_candidates to decide which
    candidates the cap applies to.

    All guess_acf_fft_highpass window variants ("acf_fft_hp2d",
    "acf_fft_hp5d", "acf_fft_hp10d", ...) map to the SAME group
    ("acf_fft_highpass"), since every smoothing window in that function
    searches the same [min_period, max_period] band -- they're different
    filtering choices applied to the same underlying period range, not
    different bands, so the cap should apply to their combined output
    rather than separately per window (which would defeat the point: 5
    windows x top_n each would still let the total balloon).
    """
    if method in ("lombscargle_short", "acf_fft_short"):
        return method
    if method.startswith("acf_fft_hp"):
        return "acf_fft_highpass"
    return None


def cap_short_period_candidates(
    guesses: list,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    min_lag: float,
    top_n: int = 3,
    scoring_n_phase: int = 40,
) -> list:
    """Prune the short-period-focused candidate families
    (lombscargle_short, acf_fft_short, acf_fft_highpass) down to their
    top `top_n` candidates each, ranked by comb_score, before they reach
    the expensive joint comb fit. Candidates from every other method pass
    through unchanged.

    Why this exists: these three families exist specifically to search a
    narrow period band more finely than the general-purpose methods do,
    which means they can propose many candidates clustered in a small
    part of parameter space (e.g. acf_fft_highpass alone can return up to
    len(smooth_windows) * n_guesses candidates -- 25 with the defaults).
    Every one of those gets fully joint-fit by fit_rotation_period unless
    pruned first, which (a) is expensive (a single joint fit costs ~2-3
    orders of magnitude more than a comb_score evaluation -- see this
    project's timing notes) and (b) increases the chance that one
    candidate exploits the near-lag-0 comb pitfall purely by having more
    rolls of the dice (see this project's near-lag-0 regression case
    study). We don't need every short-period candidate fit -- just the
    few most promising ones -- so this cheaply ranks them first with the
    same comb_score used to seed t0 elsewhere, and only lets the survivors
    through to the expensive step.

    Parameters
    ----------
    guesses : list[InitialGuess]
        The full candidate pool (typically from gather_initial_guesses
        and/or calling several guess_* functions directly).
    acf_lags, acf : the ACF being fit against.
    min_lag : lower bound for the t0 grid search (see _grid_search_t0),
        for any candidate that doesn't already have a t0.
    top_n : int
        Number of candidates to keep per short-period family (default 3).
    scoring_n_phase : int
        Phase-grid resolution used for the t0 search here (default 40,
        vs. _grid_search_t0's own default of 200). A coarser grid is fine:
        _grid_search_t0 is explicitly documented as "a coarse, cheap
        phase estimate meant to seed the real joint fit -- not a fit in
        itself" regardless of which caller derives it, so there's no
        precision this cap's ranking pass needs that the survivors'
        eventual joint fit doesn't already tolerate. This t0 IS carried
        forward onto survivors (fit_rotation_period reuses it rather than
        re-deriving one), since re-searching at full resolution for the
        few candidates that already have a perfectly adequate seed would
        just re-spend the time this function exists to save.

    Returns
    -------
    list[InitialGuess]
        All non-short-period-family candidates, unchanged, plus up to
        `top_n` per short-period family, each with its t0 filled in (if
        it didn't already have one) and its comb_score recorded in `info`
        (key "comb_score"), sorted highest-score-first within each
        family.
    """
    groups: dict = {}
    passthrough = []
    for g in guesses:
        key = _short_period_group_key(g.method)
        if key is None:
            passthrough.append(g)
        else:
            groups.setdefault(key, []).append(g)

    capped = []
    for group_guesses in groups.values():
        scored = []
        for g in group_guesses:
            t0 = g.t0 if g.t0 is not None else _grid_search_t0(
                acf_lags, acf, g.P0, min_lag=min_lag, n_phase=scoring_n_phase
            )
            score = comb_score(acf_lags, acf, g.P0, t0)
            scored.append((score, replace(g, t0=t0, info={**g.info, "comb_score": score})))
        scored.sort(key=lambda x: x[0], reverse=True)
        capped.extend(g for _, g in scored[:top_n])

    return passthrough + capped


def _grid_search_t0(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P0: float,
    min_lag: float,
    n_phase: int = 200,
    n_teeth_for_score: int = 5,
    weight_func: Optional[Callable[[int], float]] = None,
) -> float:
    """Given a candidate period P0, grid-search the phase t0 in
    [min_lag, min_lag + P0) that maximizes comb_score. The returned t0 is
    itself the lag of the (n=0) tooth. This is a coarse, cheap phase
    estimate meant to seed the real joint fit -- not a fit in itself.
    """
    trial_t0 = np.linspace(min_lag, min_lag + P0, n_phase, endpoint=False)
    scores = [
        comb_score(acf_lags, acf, P0, t0, n_max=n_teeth_for_score, weight_func=weight_func)
        for t0 in trial_t0
    ]
    return float(trial_t0[int(np.argmax(scores))])


# --------------------------------------------------------------------------
# Joint comb fit (single candidate)
# --------------------------------------------------------------------------

@dataclass
class PeakWindow:
    n: int              # harmonic index (0, 1, 2, ...)
    lag_lo: float
    lag_hi: float
    mask: np.ndarray     # boolean mask into acf_lags/acf


def _build_windows(
    acf_lags: np.ndarray,
    P0: float,
    t0: float,
    n_peaks: int,
    window_frac: float = 0.25,
) -> list:
    """Build one fitting window per harmonic n=0..n_peaks-1, each spanning
    +/- window_frac * P0 around the expected center t0 + n*P0, clipped to
    the available lag range. Windows are frozen once built: point selection
    happens once, before optimization, rather than being re-derived every
    iteration as P and t0 are refined.
    """
    lag_min, lag_max = acf_lags[0], acf_lags[-1]
    half_width = window_frac * P0
    windows = []
    for n in range(n_peaks):
        center = t0 + n * P0
        lo = max(center - half_width, lag_min)
        hi = min(center + half_width, lag_max)
        if lo >= hi or center > lag_max:
            break
        mask = (acf_lags >= lo) & (acf_lags <= hi)
        if mask.sum() < 4:
            # not enough points to constrain a 3-parameter parabola well
            continue
        windows.append(PeakWindow(n=n, lag_lo=lo, lag_hi=hi, mask=mask))
    return windows


def _fit_comb_core(
    windows: list,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P0: float,
    t0: float,
    P_bounds_frac: float = 0.3,
    allow_jitter: bool = True,
    jitter_frac: float = 0.05,
    loss: str = "soft_l1",
):
    """Fit the joint comb-of-parabolae model directly via
    scipy.optimize.least_squares with a hand-derived ANALYTIC Jacobian,
    rather than through lmfit.Parameters with a finite-difference Jacobian.

    Why: profiling found two compounding costs in the old lmfit-based
    approach (see this project's fitting-performance notes): (1) tying
    each window's center to the shared P/t0 via an lmfit `expr=` string
    made every residual evaluation re-interpret that expression through
    lmfit's asteval engine -- fixed separately by computing centers with
    plain arithmetic instead; (2) without an analytic Jacobian, scipy
    estimates one by finite differences, which perturbs every one of the
    ~20-30 free parameters separately and re-evaluates the FULL residual
    for each -- roughly 20-30x more residual evaluations than fitting
    needs, per Jacobian estimate, per optimizer iteration. This function
    fixes the second, larger cost: the model is just h - A*(lag-c)**2
    with c = t0 + n*P (+ delta_n), so its Jacobian is a few lines of
    closed-form algebra (see the `jacobian` closure below) -- giving
    scipy the exact Jacobian directly, in one cheap vectorized call,
    instead of estimating an approximate one the expensive way.

    This also drops the lmfit.Parameters layer entirely (not just the
    `expr=` ties within it): the parameter vector here is a plain numpy
    array with plain scipy bounds, with no per-call attribute-lookup
    overhead. CombFitResult.fit_result stores the raw scipy OptimizeResult
    for anyone who needs to inspect the fit directly, in place of the
    lmfit MinimizerResult this field used to hold.

    Parameter vector layout (built fresh per call, since `windows` can
    change between RANSAC-rejection iterations): [P, t0, delta_n for each
    window with n>0 (only if allow_jitter), A_n for each window in order,
    h_n for each window in order].

    Returns
    -------
    dict with keys: P, P_err, t0, t0_err, redchi, success, centers (dict
    n -> fitted center), A (dict n -> fitted curvature), h (dict n ->
    fitted height), raw (the scipy OptimizeResult).
    """
    m = len(windows)
    ns = [w.n for w in windows]
    lag_subs = [acf_lags[w.mask] for w in windows]
    acf_subs = [acf[w.mask] for w in windows]
    n_points = [len(l) for l in lag_subs]
    total_pts = sum(n_points)
    starts = np.cumsum([0] + n_points)  # residual-block start offsets

    delta_positions = [i for i, n in enumerate(ns) if n > 0] if allow_jitter else []
    n_delta = len(delta_positions)
    delta_pos_of = {i: k for k, i in enumerate(delta_positions)}  # window idx -> delta slot

    idx_P, idx_t0 = 0, 1
    idx_delta0 = 2
    idx_A0 = idx_delta0 + n_delta
    idx_h0 = idx_A0 + m
    n_params = idx_h0 + m

    x0 = np.zeros(n_params)
    lb = np.full(n_params, -np.inf)
    ub = np.full(n_params, np.inf)

    x0[idx_P] = P0
    lb[idx_P], ub[idx_P] = P0 * (1 - P_bounds_frac), P0 * (1 + P_bounds_frac)
    x0[idx_t0] = t0
    lb[idx_t0], ub[idx_t0] = t0 - 0.5 * P0, t0 + 0.5 * P0
    for i in delta_positions:
        pos = idx_delta0 + delta_pos_of[i]
        lb[pos], ub[pos] = -jitter_frac * P0, jitter_frac * P0

    for i, w in enumerate(windows):
        acf_sub, lag_sub = acf_subs[i], lag_subs[i]
        h0_guess = float(np.max(acf_sub))
        half_w = 0.5 * (w.lag_hi - w.lag_lo)
        edge_drop = h0_guess - float(np.min(acf_sub))
        A0_guess = max(edge_drop, 1e-6) / max(half_w**2, 1e-6)
        x0[idx_A0 + i] = A0_guess
        lb[idx_A0 + i] = 0.0  # forces downward-opening
        x0[idx_h0 + i] = h0_guess

    ns_arr = np.array(ns, dtype=float)

    def _centers(x):
        P, t0v = x[idx_P], x[idx_t0]
        c = t0v + ns_arr * P
        for i in delta_positions:
            c[i] += x[idx_delta0 + delta_pos_of[i]]
        return c

    def residual(x):
        centers = _centers(x)
        out = np.empty(total_pts)
        for i in range(m):
            A, h, c = x[idx_A0 + i], x[idx_h0 + i], centers[i]
            out[starts[i]:starts[i + 1]] = (h - A * (lag_subs[i] - c) ** 2) - acf_subs[i]
        return out

    def jacobian(x):
        centers = _centers(x)
        J = np.zeros((total_pts, n_params))
        for i in range(m):
            n, A, c = ns[i], x[idx_A0 + i], centers[i]
            s0, s1 = starts[i], starts[i + 1]
            diff = lag_subs[i] - c
            dresid_dc = 2.0 * A * diff  # d/dc[h - A*(lag-c)^2] = 2*A*(lag-c)
            J[s0:s1, idx_A0 + i] = -diff**2
            J[s0:s1, idx_h0 + i] = 1.0
            J[s0:s1, idx_P] = dresid_dc * n
            J[s0:s1, idx_t0] = dresid_dc
            if i in delta_pos_of:
                J[s0:s1, idx_delta0 + delta_pos_of[i]] = dresid_dc
        return J

    fit_kws = {"loss": loss} if loss != "linear" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = scipy_least_squares(
            residual, x0, jac=jacobian, bounds=(lb, ub),
            # Looser than scipy's 1e-8 defaults: this project only needs
            # ~1% relative period precision out of the comb fit itself
            # (PDM refinement, where used, and the reliability gates both
            # operate at that level, not 1e-8) -- tightening convergence
            # far beyond what anything downstream uses just burns extra
            # trust-region iterations (each one a fresh SVD solve, the
            # single largest remaining cost per fit; see this project's
            # fitting-performance notes) for precision nothing consumes.
            xtol=1e-6, ftol=1e-6, gtol=1e-6,
            **fit_kws,
        )

    centers_fit = _centers(res.x)
    dof = total_pts - n_params
    chisqr = float(np.sum(res.fun ** 2))
    redchi = chisqr / dof if dof > 0 else float("nan")

    P_err, t0_err = None, None
    with warnings.catch_warnings():
        # Same expected case the old lmfit-based code already anticipated:
        # under a robust loss / near-degenerate J^T J, some diagonal
        # covariance entries can come out negative (approximation
        # artifact, not a real fit problem) -- sqrt of those is NaN by
        # design here, not a bug to surface as a warning every call.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            # Same convention lmfit/scipy.curve_fit use: covariance from
            # the (pseudo-inverted) J^T J, scaled by the residual
            # variance. Only approximate under a robust loss, same
            # caveat as before.
            JTJ = res.jac.T @ res.jac
            cov = np.linalg.pinv(JTJ) * (redchi if np.isfinite(redchi) else 1.0)
            stderrs = np.sqrt(np.diag(cov))
            if np.isfinite(stderrs[idx_P]):
                P_err = float(stderrs[idx_P])
            if np.isfinite(stderrs[idx_t0]):
                t0_err = float(stderrs[idx_t0])
        except (np.linalg.LinAlgError, ValueError):
            pass

    return dict(
        P=float(res.x[idx_P]), P_err=P_err,
        t0=float(res.x[idx_t0]), t0_err=t0_err,
        redchi=redchi, success=bool(res.success),
        centers={n: float(centers_fit[i]) for i, n in enumerate(ns)},
        A={n: float(res.x[idx_A0 + i]) for i, n in enumerate(ns)},
        h={n: float(res.x[idx_h0 + i]) for i, n in enumerate(ns)},
        raw=res,
    )


@dataclass
class CombFitResult:
    P: float
    P_err: Optional[float]
    t0: float
    t0_err: Optional[float]
    windows: list
    fit_result: object  # raw scipy.optimize.OptimizeResult (was an lmfit MinimizerResult before this project's fitting-performance rework -- see _fit_comb_core)
    per_peak: dict
    n_peaks_used: int
    n_peaks_dropped: int
    redchi: float
    success: bool


def _fit_single_candidate(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P0: float,
    t0: float,
    n_peaks: int = 8,
    window_frac: float = 0.25,
    allow_jitter: bool = True,
    jitter_frac: float = 0.05,
    loss: str = "soft_l1",
    max_reject_iters: int = 3,
    reject_threshold_sigma: float = 4.0,
    min_peaks_required: int = 2,
) -> CombFitResult:
    """Run the actual joint least-squares comb-of-parabolae fit for ONE
    candidate (P0, t0). This is the expensive, authoritative step that
    fit_rotation_period calls once per candidate; see that function for the
    multi-candidate orchestration and the module docstring for why
    candidate generation and fitting are kept separate.

    What happens here, in order:

    1. Build one fitting window per expected peak (harmonic n = 0, 1, 2,
       ..., n_peaks-1), each a fixed lag range around t0 + n*P0. These
       windows are frozen for the whole fit -- see _build_windows.

    2. Set up the joint parameter set (_build_comb_params): a single shared
       P and t0, with every window's parabola center algebraically tied to
       them (center_n = t0 + n*P [+ small jitter]), each window's own
       curvature/height left free. This is what "evenly spaced" and
       "downward-opening" mean as HARD constraints on the model, rather
       than properties we'd have to check after an unconstrained fit.

    3. Fit with a robust loss (soft_l1 by default, via scipy's
       least_squares under lmfit) rather than plain least-squares, so that
       one badly-behaved window doesn't dominate the fit of the *shared*
       P and t0 that every other window also depends on.

    4. Iteratively drop the worst-fitting window and refit, up to
       `max_reject_iters` times, IF that window's residual RMS is a
       clear outlier (more than `reject_threshold_sigma` robust-sigma above
       the median residual RMS across all currently-fit windows) AND doing
       so wouldn't drop below `min_peaks_required` windows. This is a
       simple RANSAC-style cleanup: real data sometimes has one cycle
       disrupted by a flare, a data gap, or a genuinely anomalous spot
       configuration, and letting that one window silently degrade the fit
       of P and t0 for every other (good) window would be worse than
       excluding it.

    5. Once no more windows are dropped (or the iteration budget runs out),
       summarize each surviving window's fitted center/height/curvature and
       residual RMS into `per_peak`, and package everything into a
       CombFitResult.

    Returns
    -------
    CombFitResult for this single candidate.
    """
    windows = _build_windows(acf_lags, P0, t0, n_peaks, window_frac=window_frac)
    if len(windows) < min_peaks_required:
        raise RuntimeError(
            f"Only {len(windows)} usable peak window(s) built from "
            f"(P0={P0:.4g}, t0={t0:.4g}); need at least {min_peaks_required}."
        )

    n_dropped_total = 0
    fit = None

    for iteration in range(max_reject_iters + 1):
        # --- fit with the current set of windows ---
        fit = _fit_comb_core(
            windows, acf_lags, acf, P0, t0,
            allow_jitter=allow_jitter, jitter_frac=jitter_frac, loss=loss,
        )

        # --- compute each window's own residual RMS, to decide what (if
        # anything) is bad enough to drop before the next iteration ---
        per_peak_rms = {}
        for w in windows:
            n = w.n
            lag_sub = acf_lags[w.mask]
            acf_sub = acf[w.mask]
            c = fit["centers"][n]
            A = fit["A"][n]
            h = fit["h"][n]
            model = h - A * (lag_sub - c) ** 2
            per_peak_rms[n] = float(np.sqrt(np.mean((model - acf_sub) ** 2)))

        if iteration == max_reject_iters or len(windows) <= min_peaks_required:
            break  # out of iterations, or can't afford to drop any more

        # robust (MAD-based) outlier threshold across the current windows'
        # residual RMS values
        rms_vals = np.array(list(per_peak_rms.values()))
        med = np.median(rms_vals)
        mad = np.median(np.abs(rms_vals - med))
        sigma = 1.4826 * mad if mad > 0 else np.std(rms_vals)
        if sigma == 0:
            break  # everything fits identically well (or there's only 1-2 windows); nothing to reject

        worst_n, worst_rms = max(per_peak_rms.items(), key=lambda kv: kv[1])
        if worst_rms > med + reject_threshold_sigma * sigma and len(windows) > min_peaks_required:
            windows = [w for w in windows if w.n != worst_n]
            n_dropped_total += 1
        else:
            break  # nothing egregious left; stop iterating

    # --- final per-peak summary from the surviving windows' last fit ---
    per_peak = {}
    for w in windows:
        n = w.n
        lag_sub = acf_lags[w.mask]
        acf_sub = acf[w.mask]
        c = fit["centers"][n]
        A = fit["A"][n]
        h = fit["h"][n]
        model = h - A * (lag_sub - c) ** 2
        # NOTE: w.lag_lo/w.lag_hi are frozen at candidate-generation time
        # (built from the CANDIDATE's initial P0/t0, before this fit ever
        # ran), but `c` above is the algebraically-tied center from the
        # FINAL fitted (shared) P/t0 -- which allow_jitter, the +/-30% P
        # bound, and the +/-50%*P0 t0 bound can all move well away from
        # where the window was originally built. If the fitted vertex has
        # drifted outside its own window, this window's "fit" isn't
        # actually constrained by a local peak at all -- the parabola is
        # being extrapolated onto data it was never centered on, which can
        # still produce a deceptively small residual on a smooth/shallow
        # stretch of the ACF (e.g. the near-lag-0 decay envelope) without
        # reflecting any real periodic structure there.
        vertex_in_window = bool(w.lag_lo <= c <= w.lag_hi)
        per_peak[n] = dict(
            center=c,
            height=h,
            curvature=A,
            residual_rms=float(np.sqrt(np.mean((model - acf_sub) ** 2))),
            n_points=int(w.mask.sum()),
            vertex_in_window=vertex_in_window,
        )

    return CombFitResult(
        P=fit["P"],
        P_err=fit["P_err"],
        t0=fit["t0"],
        t0_err=fit["t0_err"],
        windows=windows,
        fit_result=fit["raw"],
        per_peak=per_peak,
        n_peaks_used=len(windows),
        n_peaks_dropped=n_dropped_total,
        redchi=fit["redchi"],
        success=bool(fit["success"]) and len(windows) >= min_peaks_required,
    )


# --------------------------------------------------------------------------
# Goodness-of-fit / acceptance helper
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Phase-dispersion and composite-spectrum diagnostics (optional)
# --------------------------------------------------------------------------
#
# These are model-different cross-checks on a candidate that's already been
# through the joint comb fit -- they don't generate new candidates, and
# they're never computed unless assess_rotation_candidate is explicitly
# given the extra data they need (see that function below). Motivation: the
# joint comb fit and its redchi evaluate a candidate against the ACF's
# *shape* in narrow local windows; both of the diagnostics below instead
# check the candidate directly against the light curve's own phase
# coherence (phase_dispersion_stat) or against the raw ACF's local
# prominence rather than a fitted parabola (acf_peak_prominence_diagnostics)
# -- genuinely different failure modes than height/redchi-based checks can
# catch. See this project's half-period-alias case study (two unequal
# starspot groups) for a worked example where every existing method AND the
# joint fit's own redchi confidently prefer a wrong P/2 answer, and
# phase_dispersion_theta is the one diagnostic that isn't fooled.

def phase_dispersion_stat(
    time: np.ndarray,
    flux: np.ndarray,
    P: float,
    n_bins: int = 10,
) -> float:
    """Phase Dispersion Minimization statistic (Stellingwerf 1978) for a
    candidate period P: fold the light curve on P, bin it in phase, and
    compare the scatter *within* phase bins to the scatter of the whole
    (unfolded) light curve.

        theta = [sum_j (n_j - 1) * s_j^2] / [(N - M) * sigma_total^2]

    where s_j^2 is the variance of the points in phase bin j, n_j is the
    number of points in bin j, N is the total number of points, M is the
    number of non-empty bins, and sigma_total^2 is the variance of the
    full (unfolded) light curve.

    Interpretation: if P is (close to) the true period, folding the light
    curve on it lines up points from different cycles that really do
    belong at the same phase, so each phase bin should look tight compared
    to the light curve's overall scatter -- theta << 1. If P is wrong, the
    fold scrambles unrelated points together in every bin, so each bin's
    scatter approaches the light curve's overall scatter -- theta -> 1 (or
    even slightly above 1, since folding on a bad period doesn't reduce
    variance at all). theta is bounded below by 0 for a perfectly periodic,
    noise-free signal correctly folded.

    This is a genuinely different diagnostic from anything else in this
    module: comb_score and the joint fit's redchi both evaluate a
    candidate against the ACF's *shape*, whereas theta evaluates it
    directly against the light curve's phase coherence, with no ACF or
    parabola model involved at all. A candidate that fits the ACF
    reasonably well but is actually a harmonic of the true period (e.g. 2x
    or 0.5x too long/short) will often show a visibly worse (higher) theta
    than the true period, since folding on the wrong multiple misaligns
    the underlying light curve's repeating shape even when the ACF's comb
    of peaks looks superficially plausible -- making theta a useful,
    cheap, independent cross-check on a joint comb fit's winning candidate
    (or on close runner-ups) before trusting it.

    Parameters
    ----------
    time, flux : the light curve (NaNs are dropped automatically).
    P : candidate period to test, same units as `time`.
    n_bins : number of phase bins in [0, 1) to fold into. Default 10,
        following Stellingwerf (1978)'s original recommendation of order
        ~10 bins for typical sampling; too few bins washes out real
        structure, too many starves each bin of points.

    Returns
    -------
    theta : float. Lower is better; theta ~ 0 indicates strong phase
        coherence at this period, theta ~ 1 indicates no more phase
        coherence than a random period would show.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]

    if len(flux) < n_bins * 2 or P <= 0:
        return float("nan")

    phase = np.mod(time, P) / P
    bin_idx = np.clip((phase * n_bins).astype(int), 0, n_bins - 1)

    sigma_total_sq = np.var(flux, ddof=1)
    if sigma_total_sq == 0:
        return float("nan")

    numerator = 0.0
    n_nonempty_bins = 0
    for j in range(n_bins):
        in_bin = bin_idx == j
        n_j = int(np.sum(in_bin))
        if n_j < 2:
            continue
        numerator += (n_j - 1) * np.var(flux[in_bin], ddof=1)
        n_nonempty_bins += 1

    dof = len(flux) - n_nonempty_bins
    if dof <= 0:
        return float("nan")

    theta = numerator / (dof * sigma_total_sq)
    return float(theta)


def refine_period_by_pdm(
    time: np.ndarray,
    flux: np.ndarray,
    P0: float,
    n_bins: int = 10,
    search_frac: float = 0.02,
    n_trial: int = 201,
) -> dict:
    """Locally refine a candidate period by minimizing phase_dispersion_stat
    over a small grid of trial periods around P0, and return the best one.

    Why this matters (found empirically while adding this diagnostic):
    theta is a much less forgiving function of period precision than the
    joint comb fit's own P is. The comb fit only ever looks at a handful of
    narrow windows near each expected peak, so a fractional-percent error
    in P barely nudges its redchi. Folding the ENTIRE light curve on P,
    however, accumulates that same fractional error over every cycle in
    the baseline -- for a baseline spanning N_cycles = baseline/P rotations,
    a period error of order 1/N_cycles is already enough to smear a phase
    fold into near-total incoherence (theta -> 1) regardless of whether P0
    was close to correct. Concretely, a light curve with ~20 cycles in its
    baseline needs P known to within roughly 1/20 = 5% just to avoid this,
    and considerably better than that to get a theta clean enough to be
    useful as a discriminating statistic.

    Practically: don't hand fit.P from a CombFitResult straight to
    phase_dispersion_stat and expect a reliable answer if the light curve
    spans many cycles -- refine it first, exactly as this function does.
    assess_rotation_candidate below does this automatically.

    Parameters
    ----------
    time, flux : the light curve.
    P0 : starting-point period to refine (e.g. a CombFitResult.P or an
        InitialGuess.P0).
    n_bins : phase bins for phase_dispersion_stat.
    search_frac : half-width of the search grid, as a fraction of P0 (e.g.
        0.02 searches +/- 2% around P0).
    n_trial : number of trial periods in the search grid.

    Returns
    -------
    dict with 'P_refined' (the trial period achieving the lowest theta) and
    'theta_min' (its theta value).
    """
    trial_periods = np.linspace(P0 * (1 - search_frac), P0 * (1 + search_frac), n_trial)
    thetas = np.array([
        phase_dispersion_stat(time, flux, P, n_bins=n_bins) for P in trial_periods
    ])
    if not np.isfinite(thetas).any():
        return dict(P_refined=float(P0), theta_min=float("nan"))
    i_best = int(np.nanargmin(thetas))
    return dict(P_refined=float(trial_periods[i_best]), theta_min=float(thetas[i_best]))


def acf_peak_prominence_diagnostics(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    peak_lag: float,
    search_frac: float = 0.5,
    P_for_search_window: Optional[float] = None,
) -> dict:
    """Composite-spectrum-style peak height/prominence diagnostics for a
    single ACF peak, following the G_ACF/H_ACF statistics used in the
    Santos/ROOSTER rotation pipeline (see Ceillier et al. 2017): the
    height of the peak itself, and its prominence relative to the two
    local minima flanking it on either side.

    This deliberately mirrors, but is independent of, the joint comb fit:
    _fit_single_candidate's fitted `height` for a window comes from a
    parabola fit local to a narrow window, while G here is read directly
    off the raw ACF at its actual local maximum near `peak_lag` (which may
    drift slightly from the comb's algebraic prediction). Computing both
    gives you two largely independent height estimates for the same
    feature -- if the joint-fit height and the raw ACF's G disagree a lot,
    that's a sign the joint fit's tied structure is dragging the fitted
    parabola away from the ACF's actual local peak, worth a look via
    plot_comb_fit.

    Parameters
    ----------
    acf_lags, acf : the full ACF.
    peak_lag : the lag near which to look for the actual local ACF
        maximum -- typically fit.t0 (for the fundamental) or
        fit.t0 + n*fit.P for harmonic n, taken from a CombFitResult.
    search_frac : how far (as a fraction of P_for_search_window) to look
        on either side of `peak_lag` for the true local maximum and its
        flanking local minima. Default 0.5 (i.e. +/- half a period).
    P_for_search_window : the period defining the search window width. If
        None, defaults to 10% of the ACF's full lag range (a reasonable
        width when you don't have a specific P in hand).

    Returns
    -------
    dict with:
        peak_lag_actual : lag of the true local ACF maximum nearest
            `peak_lag` (may differ slightly from the input).
        G : height of that local maximum (the ACF value there).
        left_min, right_min : ACF values at the local minima immediately
            to the left and right of the peak, within the search window.
        H : mean of (G - left_min) and (G - right_min) -- the peak's
            prominence relative to its immediate surroundings. A tall but
            *unprominent* peak (e.g. one just riding down the ACF's broad
            envelope near lag 0) will have a high G but a low H; H is
            usually the more trustworthy "is this a real, distinct bump"
            indicator of the two.
    """
    if P_for_search_window is None:
        P_for_search_window = 0.1 * (acf_lags[-1] - acf_lags[0])
    half_width = search_frac * P_for_search_window

    lo = peak_lag - half_width
    hi = peak_lag + half_width
    mask = (acf_lags >= lo) & (acf_lags <= hi)
    if mask.sum() < 3:
        return dict(peak_lag_actual=np.nan, G=np.nan, left_min=np.nan,
                     right_min=np.nan, H=np.nan)

    sub_lags = acf_lags[mask]
    sub_acf = acf[mask]
    i_peak = int(np.argmax(sub_acf))
    peak_lag_actual = float(sub_lags[i_peak])
    G = float(sub_acf[i_peak])

    left_vals = sub_acf[:i_peak + 1]
    right_vals = sub_acf[i_peak:]
    left_min = float(np.min(left_vals)) if len(left_vals) > 0 else np.nan
    right_min = float(np.min(right_vals)) if len(right_vals) > 0 else np.nan

    prominences = [G - v for v in (left_min, right_min) if np.isfinite(v)]
    H = float(np.mean(prominences)) if prominences else np.nan

    return dict(peak_lag_actual=peak_lag_actual, G=G, left_min=left_min,
                right_min=right_min, H=H)


def composite_spectrum_diagnostics(
    fit: CombFitResult,
    acf_lags: np.ndarray,
    acf: np.ndarray,
) -> dict:
    """Convenience wrapper: apply acf_peak_prominence_diagnostics to the
    fundamental (n=0 tooth, i.e. fit.t0) of a CombFitResult, giving a
    single G_ACF/H_ACF-style height/prominence summary for the winning (or
    any candidate) fit -- ready to drop straight into a feature vector
    alongside fit.redchi, fit.n_peaks_used, and phase_dispersion_stat for
    an eventual ML selection step (see ROOSTER, Breton et al. 2021, for
    the precedent this is modeled on).

    Returns
    -------
    dict, the output of acf_peak_prominence_diagnostics evaluated at
    fit.t0, with P_for_search_window=fit.P.
    """
    return acf_peak_prominence_diagnostics(
        acf_lags, acf, peak_lag=fit.t0, P_for_search_window=fit.P,
    )


# --------------------------------------------------------------------------
# Goodness-of-fit / acceptance helper
# --------------------------------------------------------------------------

def assess_rotation_candidate(
    fit: CombFitResult,
    acf: np.ndarray,
    min_peaks: int = 3,
    max_redchi: float = 5.0,
    min_height_over_local_std: float = 3.0,
    time: Optional[np.ndarray] = None,
    flux: Optional[np.ndarray] = None,
    acf_lags: Optional[np.ndarray] = None,
    n_pdm_bins: int = 10,
) -> dict:
    """Bundle a handful of acceptance diagnostics for a CombFitResult into a
    single dict. Does not make a hard accept/reject call (thresholds are
    target- and noise-regime-dependent) -- returns the ingredients so you
    can set your own cuts, or use fit_rotation_period's built-in gating.

    Two additional, OPTIONAL diagnostics are computed if you supply the
    extra data they need, on top of the original height/curvature-based
    ones (which only ever needed `fit` and `acf`, and still only need
    those):

    - `phase_dispersion_theta` (needs `time` and `flux`): the Stellingwerf
      (1978) PDM statistic, LOCALLY REFINED around fit.P -- see
      refine_period_by_pdm's docstring for why a single evaluation at
      fit.P is not good enough (the joint comb fit's own period precision
      is generally not tight enough for a many-cycle phase fold to stay
      coherent, even when fit.P is essentially correct). This checks phase
      coherence directly in the light curve, independent of the
      ACF/parabola model entirely, so it's a genuinely different failure
      mode than anything the height/redchi-based checks below can catch
      (e.g. it will often flag a harmonic alias that nonetheless produces
      a deceptively clean-looking comb fit). `phase_dispersion_P_refined`
      (the locally-refined period PDM actually settled on) is also
      included, so you can see how far it moved from fit.P.

    - `G_ACF`, `H_ACF` (needs `acf_lags`; reuses `acf`): the composite-
      spectrum-style peak height and flanking-minima prominence of the
      fundamental (fit.t0), from acf_peak_prominence_diagnostics /
      composite_spectrum_diagnostics -- see those docstrings. H_ACF in
      particular is a cheap, model-free prominence check that complements
      `height_snr` below (which uses the *fitted* parabola height and the
      ACF's global standard deviation, not the local flanking minima).

    Both are omitted (left out of the returned dict) if their required
    inputs aren't supplied, so this function's default, minimal call
    signature (just `fit` and `acf`) is unchanged from before -- e.g.
    fit_rotation_period's internal gating calls remain exactly as fast and
    exactly as they were before these two diagnostics existed.

    A third, always-computed diagnostic, `frac_vertex_in_window`: the
    fraction of surviving windows whose fitted parabola vertex (per_peak
    center) actually falls within that window's own frozen lag bounds. A
    window's bounds are fixed at candidate-generation time, but its center
    is tied to the jointly-fit (shared) P/t0, which allow_jitter and the
    fit's P/t0 bounds can move well away from where the window was built.
    A vertex outside its window means that window's low residual isn't
    evidence of a real local peak -- the parabola is being extrapolated
    onto data it was never centered on. See this project's near-lag-0
    regression case study (short-period candidates achieving a deceptively
    low redchi by fitting the ACF's smooth decay envelope) for a worked
    example where every surviving window failed this check.
    """
    heights = np.array([p["height"] for p in fit.per_peak.values()])
    curvatures = np.array([p["curvature"] for p in fit.per_peak.values()])
    vertex_in_window = np.array([p.get("vertex_in_window", True) for p in fit.per_peak.values()])
    ns = np.array(list(fit.per_peak.keys()))
    order = np.argsort(ns)
    heights, curvatures, ns = heights[order], curvatures[order], ns[order]
    vertex_in_window = vertex_in_window[order]

    acf_std = float(np.std(acf))
    height_snr = heights / acf_std if acf_std > 0 else heights * np.nan

    if len(heights) > 1:
        non_increasing = np.diff(heights) <= 1e-3 * np.abs(heights[:-1])
        frac_non_increasing = float(np.mean(non_increasing))
    else:
        frac_non_increasing = np.nan

    frac_positive = float(np.mean(heights > 0)) if len(heights) > 0 else 0.0
    frac_vertex_in_window = float(np.mean(vertex_in_window)) if len(vertex_in_window) > 0 else 0.0

    diagnostics = dict(
        n_peaks_used=fit.n_peaks_used,
        n_peaks_dropped=fit.n_peaks_dropped,
        redchi=fit.redchi,
        heights=heights,
        height_snr=height_snr,
        curvatures=curvatures,
        frac_non_increasing_height=frac_non_increasing,
        frac_positive_heights=frac_positive,
        frac_vertex_in_window=frac_vertex_in_window,
        passes_min_peaks=fit.n_peaks_used >= min_peaks,
        passes_redchi=fit.redchi <= max_redchi if np.isfinite(fit.redchi) else False,
        passes_height_snr=bool(np.all(height_snr >= min_height_over_local_std)),
    )

    if time is not None and flux is not None:
        pdm = refine_period_by_pdm(time, flux, fit.P, n_bins=n_pdm_bins)
        diagnostics["phase_dispersion_theta"] = pdm["theta_min"]
        diagnostics["phase_dispersion_P_refined"] = pdm["P_refined"]

    if acf_lags is not None:
        composite = composite_spectrum_diagnostics(fit, acf_lags, acf)
        diagnostics["G_ACF"] = composite["G"]
        diagnostics["H_ACF"] = composite["H"]

    return diagnostics


# --------------------------------------------------------------------------
# Multi-candidate fitting and arbitration
# --------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """The outcome of testing one candidate period all the way through the
    joint comb fit."""
    period: float
    t0: float
    source_guess: InitialGuess
    fit: Optional[CombFitResult]
    diagnostics: Optional[dict]
    passed_gates: bool = False
    error: Optional[str] = None
    # How many raw candidate proposals (across all guess_* methods, before
    # the dedup step below merged near-identical periods into this one
    # representative) supported this same period, and which methods they
    # came from. n_duplicate_guesses=0 and contributing_methods=(source
    # method,) means this candidate was the only one proposing this period
    # -- no cross-method corroboration at the candidate-generation stage.
    # This is computed BEFORE fitting, so don't confuse it with post-fit
    # agreement between different (still-distinct) fitted candidates.
    n_duplicate_guesses: int = 0
    contributing_methods: tuple = ()


@dataclass
class EnsembleResult:
    success: bool
    message: str
    best_fit: Optional[CombFitResult]
    best_guess: Optional[InitialGuess]
    candidates: list        # list[CandidateResult], successfully-fit ones, sorted best-first
    n_candidates_tried: int
    failed_candidates: list  # list[CandidateResult] that errored during fitting


def fit_rotation_period(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    initial_guesses: Union[InitialGuess, list],
    n_peaks: int = 8,
    window_frac: float = 0.25,
    allow_jitter: bool = True,
    jitter_frac: float = 0.05,
    loss: str = "soft_l1",
    max_reject_iters: int = 3,
    reject_threshold_sigma: float = 4.0,
    min_peaks_required: int = 4,
    min_frac_positive_heights: float = 0.8,
    min_mean_height_snr: float = 1.0,
    min_frac_vertex_in_window: float = 0.8,
    dedup_rel_tol: float = 0.03,
    short_period_cap_n: Optional[int] = 3,
) -> EnsembleResult:
    """Fit the joint comb model to EVERY candidate in `initial_guesses`,
    and pick whichever one produces the most convincing result -- or
    report that none of them do.

    This is the arbitration stage described in the module docstring: each
    guess_* function only proposes candidates using its own cheap,
    method-specific evidence, and never checks them against the ACF's
    actual shape. This function is where that real check happens, on equal
    footing, for every candidate regardless of which method proposed it.

    Step by step
    ------------
    1. Normalize `initial_guesses` to a list (a single InitialGuess is
       also accepted, for convenience/backward compatibility).

    2. Sanity-filter (period not absurdly short, and long enough baseline
       to fit at least 2 cycles), then cap the short-period-focused
       candidate families (lombscargle_short, acf_fft_short,
       acf_fft_highpass) down to their top `short_period_cap_n` candidates
       each by comb_score (see cap_short_period_candidates) -- these
       families can otherwise flood the pool with many candidates
       clustered in a narrow period band. Then deduplicate: candidates
       within `dedup_rel_tol` relative difference of each other are
       treated as the same candidate (only the first encountered, after
       sorting by period, is kept) -- there's no point fitting nearly-
       identical periods twice just because two different methods
       happened to propose them independently.

    3. For each surviving candidate:
         a. If it doesn't already have a t0 (candidate generation doesn't
            compute one -- see module docstring), find a reasonable
            starting phase via a coarse grid search against the ACF
            (_grid_search_t0). This is a cheap heuristic to seed the fit,
            not a fit in itself.
         b. Run the real joint least-squares comb fit at that (P, t0) via
            _fit_single_candidate. This is the expensive, authoritative
            step -- see that function's docstring for what it does.
         c. Compute acceptance diagnostics (assess_rotation_candidate) for
            the result.
       A candidate that can't even be fit (e.g. too few usable windows) is
       recorded with an error message rather than silently dropped, so you
       can see what was tried.

    4. Apply four reliability gates to every successfully-fit candidate.
       A candidate must satisfy ALL four to be considered "passed":
         - n_peaks_used >= min_peaks_required: enough of the expected
           peaks survived the fit (and the RANSAC-style rejection inside
           _fit_single_candidate) to be confident this is a real,
           sustained periodicity rather than a couple of coincidental
           matches.
         - frac_positive_heights >= min_frac_positive_heights: the fitted
           peak heights must mostly be genuine positive bumps. A comb that
           just rides the ACF's smooth decay down through zero near lag 0
           can otherwise achieve a deceptively good (tiny) reduced
           chi-squared without corresponding to any real periodicity at
           all -- this gate catches that.
         - mean(height_snr) >= min_mean_height_snr: the peaks must be tall
           enough relative to the ACF's overall noise level (its standard
           deviation) to be distinguishable from noise fluctuations, on
           average across the surviving peaks.
         - frac_vertex_in_window >= min_frac_vertex_in_window: each
           window's fitted parabola vertex must actually fall within that
           window's own (frozen) lag bounds. Window bounds are fixed at
           candidate-generation time, but a window's center is tied to the
           jointly-fit (shared) P/t0, which allow_jitter and the fit's own
           P/t0 bounds can move well away from where the window was
           originally built. A vertex outside its window means that
           window's low residual isn't evidence of a real local peak --
           the parabola was extrapolated onto data it was never centered
           on. Default 0.8 (matching min_frac_positive_heights) rather
           than a stricter 1.0: the n=0 window in particular sits against
           the acf_lags domain edge (lag >= 0), so a legitimately-good fit
           can still see its vertex drift a small amount past that
           boundary during normal t0 refinement -- this is a boundary
           artifact, not evidence the window failed to capture a real
           peak, and shouldn't be penalized the same as a candidate where
           EVERY window's vertex has drifted away by a large amount (the
           actual near-lag-0 pathology this gate exists to catch).

    5. Among candidates that pass all three gates, pick the one with the
       lowest reduced chi-squared -- the tightest joint fit. Candidates
       are returned sorted this way (passed candidates first, each
       sub-sorted by reduced chi-squared) so you can inspect runner-ups.

    6. If NO candidate passes all three gates, this function does NOT fall
       back to just returning its single best (but not-good-enough)
       attempt as if it were reliable. Instead it returns
       success=False, with a plain-language `message` explaining why, and
       `best_fit`/`best_guess` still populated with the closest attempt
       (clearly documented as unreliable) purely so you can inspect what
       almost worked -- e.g. via the diagnostic plots -- rather than being
       left with nothing to look at.

    Parameters
    ----------
    initial_guesses : an InitialGuess, or (typically) a list of them, e.g.
        the concatenated output of gather_initial_guesses() or of calling
        several guess_* functions yourself.
    n_peaks, window_frac, allow_jitter, jitter_frac, loss,
    max_reject_iters, reject_threshold_sigma : forwarded to
        _fit_single_candidate for every candidate.
    min_peaks_required, min_frac_positive_heights, min_mean_height_snr,
    min_frac_vertex_in_window : the four reliability gates described above.
    dedup_rel_tol : relative-difference tolerance for treating two
        candidate periods as duplicates.
    short_period_cap_n : if not None, cap the lombscargle_short,
        acf_fft_short, and acf_fft_highpass candidate families down to
        their top `short_period_cap_n` candidates each (by comb_score)
        before the expensive fitting step -- see
        cap_short_period_candidates for why. Set to None to disable and
        fit every candidate from these families (the old behavior).

    Returns
    -------
    EnsembleResult
    """
    if isinstance(initial_guesses, InitialGuess):
        initial_guesses = [initial_guesses]
    if len(initial_guesses) == 0:
        return EnsembleResult(
            success=False,
            message="No candidate periods were provided to fit_rotation_period.",
            best_fit=None, best_guess=None, candidates=[], n_candidates_tried=0,
            failed_candidates=[],
        )

    dt_acf = np.median(np.diff(acf_lags))
    lag_min, lag_max = acf_lags[0], acf_lags[-1]
    min_lag = 3 * dt_acf

    # --- sanity filter + dedup (by period, ascending) ---
    sane = [
        g for g in initial_guesses
        if g.P0 > 2 * dt_acf and _teeth_count(g.P0, g.t0 or lag_min, lag_max) >= 2
    ]

    # --- cap short-period-focused families before the expensive fitting
    # step (see cap_short_period_candidates docstring) ---
    if short_period_cap_n is not None:
        sane = cap_short_period_candidates(
            sane, acf_lags, acf, min_lag=min_lag, top_n=short_period_cap_n
        )

    sane.sort(key=lambda g: g.P0)
    deduped = []
    merged_groups = []  # parallel to `deduped`: every raw guess merged into it
    for g in sane:
        if not deduped or (g.P0 - deduped[-1].P0) / deduped[-1].P0 > dedup_rel_tol:
            deduped.append(g)
            merged_groups.append([g])
        else:
            # within dedup_rel_tol of the last kept candidate: don't fit it
            # separately, but DO remember it supported this same period --
            # otherwise this information (e.g. "two different methods
            # independently proposed ~this period") is silently lost before
            # it ever reaches feature extraction / the ML ranker.
            merged_groups[-1].append(g)

    if len(deduped) == 0:
        return EnsembleResult(
            success=False,
            message=(
                "No candidate period survived basic sanity checks (period "
                "too short, or too long for at least 2 cycles to fit in "
                "the ACF's lag range)."
            ),
            best_fit=None, best_guess=None, candidates=[],
            n_candidates_tried=0, failed_candidates=[],
        )

    # --- fit every surviving candidate ---
    results = []
    for guess, group in zip(deduped, merged_groups):
        n_dup = len(group) - 1
        contributing_methods = tuple(sorted(set(gg.method for gg in group)))
        t0 = guess.t0 if guess.t0 is not None else _grid_search_t0(
            acf_lags, acf, guess.P0, min_lag=min_lag
        )
        try:
            fit = _fit_single_candidate(
                acf_lags, acf, guess.P0, t0,
                n_peaks=n_peaks, window_frac=window_frac,
                allow_jitter=allow_jitter, jitter_frac=jitter_frac, loss=loss,
                max_reject_iters=max_reject_iters,
                reject_threshold_sigma=reject_threshold_sigma,
                min_peaks_required=min(min_peaks_required, 2),
            )
            diag = assess_rotation_candidate(fit, acf, min_peaks=min_peaks_required)
            passed = (
                fit.n_peaks_used >= min_peaks_required
                and diag["frac_positive_heights"] >= min_frac_positive_heights
                and np.nanmean(diag["height_snr"]) >= min_mean_height_snr
                and diag["frac_vertex_in_window"] >= min_frac_vertex_in_window
            )
            results.append(CandidateResult(
                period=guess.P0, t0=t0, source_guess=guess, fit=fit,
                diagnostics=diag, passed_gates=bool(passed),
                n_duplicate_guesses=n_dup, contributing_methods=contributing_methods,
            ))
        except Exception as exc:  # noqa: BLE001 -- keep trying other candidates
            results.append(CandidateResult(
                period=guess.P0, t0=t0, source_guess=guess, fit=None,
                diagnostics=None, passed_gates=False,
                error=f"{type(exc).__name__}: {exc}",
                n_duplicate_guesses=n_dup, contributing_methods=contributing_methods,
            ))

    fit_ok = [r for r in results if r.fit is not None]
    failed = [r for r in results if r.fit is None]

    if len(fit_ok) == 0:
        return EnsembleResult(
            success=False,
            message=(
                f"None of the {len(deduped)} candidate period(s) could even "
                "be fit (too few usable windows in every case). Try "
                "widening window_frac or lowering min_peaks_required."
            ),
            best_fit=None, best_guess=None, candidates=[],
            n_candidates_tried=len(deduped), failed_candidates=failed,
        )

    def _sort_key(r):
        redchi = r.fit.redchi if np.isfinite(r.fit.redchi) else np.inf
        return (r.passed_gates, -redchi)

    fit_ok.sort(key=_sort_key, reverse=True)
    passing = [r for r in fit_ok if r.passed_gates]

    if len(passing) == 0:
        best = fit_ok[0]
        return EnsembleResult(
            success=False,
            message=(
                f"Tested {len(deduped)} candidate period(s); none met the "
                f"reliability thresholds (min_peaks_required="
                f"{min_peaks_required}, min_frac_positive_heights="
                f"{min_frac_positive_heights}, min_mean_height_snr="
                f"{min_mean_height_snr}, min_frac_vertex_in_window="
                f"{min_frac_vertex_in_window}). The closest attempt was P="
                f"{best.fit.P:.4g} (n_peaks_used={best.fit.n_peaks_used}, "
                f"frac_positive_heights={best.diagnostics['frac_positive_heights']:.2f}, "
                f"mean_height_snr={np.nanmean(best.diagnostics['height_snr']):.2f}, "
                f"frac_vertex_in_window={best.diagnostics['frac_vertex_in_window']:.2f}) "
                "-- attached for inspection, but should NOT be treated as a "
                "reliable rotation period measurement."
            ),
            best_fit=best.fit,
            best_guess=InitialGuess(P0=best.period, t0=best.t0, method="ensemble_best_unreliable"),
            candidates=fit_ok, n_candidates_tried=len(deduped),
            failed_candidates=failed,
        )

    best = passing[0]
    return EnsembleResult(
        success=True,
        message=(
            f"Selected P={best.fit.P:.4g} from {len(passing)} candidate(s) "
            f"that passed reliability gates (of {len(deduped)} tested)."
        ),
        best_fit=best.fit,
        best_guess=InitialGuess(P0=best.period, t0=best.t0, method="ensemble_best"),
        candidates=fit_ok, n_candidates_tried=len(deduped),
        failed_candidates=failed,
    )

"""Roman-cadence spotted-rotator generator (P34 prototype): quasi-periodic GP spots on WG6's GBTDS grid.

What it makes
-------------
One light curve = one star in one 71-d GBTDS season, F146, on WG6's own time grid
(``roman_times_shortcadence.npy``, 8,450 points at 12.1 min per season), with white noise drawn from WG6's
**improved** sigma(F146) relation (``docs/WG6_REFERENCE.md`` §8.3). The spot signal is a Gaussian process with the
``celerite2.terms.RotationTerm`` kernel (a sum of two stochastically driven damped harmonic oscillators, SHOs, at
P and P/2), so the spots *evolve*: the coherence time of the modulation is set by a spot lifetime.

Differential rotation: the signal is the sum of ``n_bands`` independent ``RotationTerm`` processes, one per spot
latitude, each at P(lat) = P_eq / (1 - shear sin^2 lat) (the SMARTS/butterpy convention, ``evaluation.latitude_period``).
Truth therefore carries P_eq, shear, MINLAT/MAXLAT and the active-band ``period_range`` exactly as SMARTS headers do,
so ``evaluation.classify_period`` scores CORRECT / DIFFROT / HALF / DOUBLE unchanged.

Sampling backend
----------------
The SHO process has an exact linear-Gaussian state-space form (state = position, velocity; transition = expm(A dt)),
so it can be sampled exactly in O(N) with numpy/scipy. ``backend="numpy"`` does that; ``backend="celerite2"`` calls
``celerite2.GaussianProcess(...).sample()`` (same kernel, different random stream); ``"auto"`` uses celerite2 when it
is importable, else numpy. Both give the same *distribution*; ``tests/test_make_roman_rotators.py`` checks the numpy
sampler's empirical autocovariance against the analytic RotationTerm kernel (and against celerite2's own kernel
values when celerite2 is installed). Light curves are regenerated from ``seed`` (the parameter table is the dataset;
no light-curve files need to be stored). Use one backend per dataset for bit-reproducibility.

Population priors
-----------------
``TYPE_PRIORS`` holds literature-informed per-type ranges (period, Kepler-band amplitude, spot lifetime, shear,
F146 range). They are *priors for a development set*, not a population synthesis; see
``reports/2026-11-13_p34-roman-rotators/README.md`` for the citations and the open questions.

CLI (prototype)
---------------
    python tools/make_roman_rotators.py --prototype --out reports/2026-11-13_p34-roman-rotators/data/prototype_params.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
from evaluation import active_band_period_range, latitude_period  # noqa: E402

# --- Roman grid and noise -------------------------------------------------------------------------------------------
WG6_TIMES_REL = "simple_noise/drive-download-20261111T184124Z-1-001/roman_times_shortcadence.npy"
CADENCE_DAYS = 12.1 / 1440.0
SEASON_DAYS = 71.0
SEASON_POINTS = 8450

# WG6 *improved* sigma per 12.1-min F146 point (mmag), median over OGLE-class stars, measured in
# reports/2026-11-12_wg6-dataset-scoping (Part 2 table). 21 mag: FL improved (12.5) x the OGLE/FL improved ratio at 20 mag
# (4.27/5.19 = 0.82); 22-24: the simple curve x the improved/simple ratio at 20 mag (0.73). Both extrapolations, flagged in
# the report.
SIGMA_MAG = np.array([14.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 24.0])
SIGMA_MMAG = np.array([1.00, 1.15, 1.27, 1.62, 2.72, 4.27, 12.5 * 0.82, 18.0 * 0.73, 68.7 * 0.73])
SIGMA_STAR_SCATTER_DEX = 0.05  # per-star scatter of the improved variant at fixed magnitude (WG6_REFERENCE §8.3)

# Kepler-band -> F146 spot-amplitude factor. Claytor & Tayar (CS23 poster, 2026): 4500 K photosphere / 3500 K spot
# contrast 53% in F146 vs 76% in F062 (~Kepler band) -> ~0.7. Assumption; refine per Teff later.
F146_AMP_FACTOR = 0.7
RVAR_PER_SIGMA = 3.29  # Rvar (5-95% range) of a Gaussian process = 2 * 1.645 * sigma


def roman_season_grid(season: int = 1, times_path: str | os.PathLike | None = None) -> np.ndarray:
    """Time stamps (days from season start) of one WG6 GBTDS season in F146.

    Reads WG6's ``roman_times_shortcadence.npy`` (``$WG6_RESAMPLED_ROOT`` or the Data_Vault default) when present and
    cuts the 1-indexed ``season`` at gaps > 1 d. Falls back to the identical uniform grid (8,450 points at 12.1 min,
    which is what the file holds within a season) when the drive is not mounted, so tests never need the drive.
    """
    if times_path is None:
        root = Path(os.environ.get("WG6_RESAMPLED_ROOT", "/Volumes/Data_Vault/wg6_roman_data_resampled"))
        times_path = root / WG6_TIMES_REL
    p = Path(times_path)
    if p.is_file():
        jd = np.asarray(np.load(p), float)
        cut = np.r_[0, np.where(np.diff(jd) > 1.0)[0] + 1, jd.size]
        if not 1 <= season < cut.size:
            raise ValueError(f"season {season} not in 1..{cut.size - 1}")
        seg = jd[cut[season - 1]:cut[season]]
        return seg - seg[0]
    return np.arange(SEASON_POINTS) * CADENCE_DAYS


def sigma_f146_mmag(mag) -> np.ndarray:
    """Median WG6-improved per-point sigma (mmag) at F146 magnitude ``mag`` (log-linear interpolation, clipped ends)."""
    return 10 ** np.interp(np.asarray(mag, float), SIGMA_MAG, np.log10(SIGMA_MMAG))


# --- RotationTerm kernel as SHOs, and an exact SHO sampler ----------------------------------------------------------
def rotation_term_sho(sigma: float, period: float, Q0: float, dQ: float, f: float) -> list:
    """celerite2 ``RotationTerm(sigma, period, Q0, dQ, f)`` as two SHO terms ``[(S0, w0, Q), ...]``.

    Same algebra as celerite2: amp = sigma^2/(1+f); term 1 at ``period`` with Q1 = 1/2 + Q0 + dQ, term 2 at
    ``period/2`` with Q2 = 1/2 + Q0 and relative variance f. Total variance = sigma^2.
    """
    amp = sigma ** 2 / (1.0 + f)
    Q1 = 0.5 + Q0 + dQ
    w1 = 4 * np.pi * Q1 / (period * np.sqrt(4 * Q1 ** 2 - 1))
    S1 = amp / (w1 * Q1)
    Q2 = 0.5 + Q0
    w2 = 8 * np.pi * Q2 / (period * np.sqrt(4 * Q2 ** 2 - 1))
    S2 = f * amp / (w2 * Q2)
    return [(S1, w1, Q1), (S2, w2, Q2)]


def sho_kernel(tau, S0: float, w0: float, Q: float) -> np.ndarray:
    """Analytic SHO autocovariance k(tau) (celerite convention), valid for Q > 1/2."""
    tau = np.abs(np.asarray(tau, float))
    eta = np.sqrt(1 - 1 / (4 * Q ** 2))
    return S0 * w0 * Q * np.exp(-w0 * tau / (2 * Q)) * (np.cos(eta * w0 * tau) + np.sin(eta * w0 * tau) / (2 * eta * Q))


def rotation_term_kernel(tau, sigma, period, Q0, dQ, f) -> np.ndarray:
    """Analytic RotationTerm autocovariance at lags ``tau`` (sum of its two SHOs)."""
    return sum(sho_kernel(tau, *p) for p in rotation_term_sho(sigma, period, Q0, dQ, f))


def sample_sho(t: np.ndarray, S0: float, w0: float, Q: float, rng: np.random.Generator) -> np.ndarray:
    """Exact draw of a zero-mean SHO Gaussian process at (sorted) times ``t``.

    State x = (position, velocity) obeys dx = A x dt + noise with A = [[0, 1], [-w0^2, -w0/Q]]; its stationary
    covariance is diag(v, w0^2 v), v = S0 w0 Q. Between samples x_{i+1} = F x_i + e, F = expm(A dt),
    cov(e) = P_inf - F P_inf F^T. Transitions are cached per distinct dt (one for the uniform Roman grid).
    """
    from scipy.linalg import expm
    t = np.asarray(t, float)
    v = S0 * w0 * Q
    Pinf = np.diag([v, w0 ** 2 * v])
    A = np.array([[0.0, 1.0], [-w0 ** 2, -w0 / Q]])
    out = np.empty(t.size)
    x = rng.multivariate_normal(np.zeros(2), Pinf)
    out[0] = x[0]
    dts = np.round(np.diff(t), 10)
    cache = {}
    for dt in np.unique(dts):
        F = expm(A * dt)
        C = Pinf - F @ Pinf @ F.T
        C = 0.5 * (C + C.T)
        w, V = np.linalg.eigh(C)
        cache[dt] = (F, V * np.sqrt(np.clip(w, 0, None)))
    z = rng.standard_normal((t.size, 2))
    for i in range(1, t.size):
        F, L = cache[dts[i - 1]]
        x = F @ x + L @ z[i]
        out[i] = x[0]
    return out


def sample_rotation_gp(t, sigma, period, Q0, dQ, f, rng: np.random.Generator, backend: str = "auto") -> np.ndarray:
    """One draw of a ``RotationTerm`` GP at times ``t``. ``backend``: "numpy", "celerite2" or "auto"."""
    if backend in ("auto", "celerite2"):
        try:
            import celerite2
            from celerite2 import terms
        except ImportError:
            if backend == "celerite2":
                raise
            backend = "numpy"
        else:
            gp = celerite2.GaussianProcess(terms.RotationTerm(sigma=sigma, period=period, Q0=Q0, dQ=dQ, f=f))
            gp.compute(np.asarray(t, float), diag=1e-12 * sigma ** 2, quiet=True)
            seed = int(rng.integers(2 ** 31))
            np.random.seed(seed)  # celerite2.sample uses the legacy global RNG; seed it from ours
            return np.asarray(gp.sample())
    return sum(sample_sho(t, *p, rng) for p in rotation_term_sho(sigma, period, Q0, dQ, f))


# --- population priors ----------------------------------------------------------------------------------------------
# Literature-informed ranges for a *development* set (citations in the P34 report). P: (lo, hi) days of the type's
# allowed range; dev draws are log-uniform inside the requested period bin ∩ this range. rvar_kep: Kepler-band Rvar
# (mmag) log-normal (median, dex). tau_P: spot lifetime in units of P (log-uniform). shear: (lo, hi) for the
# solar-like draw; frac_anti: fraction drawn anti-solar in (-hi/2, -lo). f146: (lo, hi) magnitudes for that type.
TYPE_PRIORS = {
    # bulge main sequence (old, ~8-10 Gyr): Claytor & Tayar 2026 forecast; van Saders+2016 stalled braking.
    "F":      dict(teff=(6000, 7000), P=(1.0, 25.0),  rvar_kep=(0.6, 0.35), tau_P=(1.0, 4.0),  shear=(0.05, 0.4),  frac_anti=0.0, f146=(17.5, 21.0)),
    "G":      dict(teff=(5200, 6000), P=(5.0, 50.0),  rvar_kep=(1.2, 0.35), tau_P=(2.0, 8.0),  shear=(0.05, 0.3),  frac_anti=0.0, f146=(18.5, 22.0)),
    "K":      dict(teff=(3900, 5200), P=(8.0, 70.0),  rvar_kep=(2.5, 0.35), tau_P=(3.0, 12.0), shear=(0.01, 0.2),  frac_anti=0.0, f146=(19.5, 23.0)),
    # foreground disk M dwarfs: fast + slow branches (Newton+2016; Medina+2020).
    "M":      dict(teff=(3000, 3900), P=(0.1, 150.0), rvar_kep=(5.0, 0.4),  tau_P=(5.0, 20.0), shear=(0.0, 0.05),  frac_anti=0.0, f146=(17.0, 22.0)),
    # young / active (disk foreground, BY Dra, active binaries), any FGK.
    "active": dict(teff=(3900, 6500), P=(0.2, 12.0),  rvar_kep=(12.0, 0.35), tau_P=(2.0, 20.0), shear=(0.02, 0.4), frac_anti=0.2, f146=(16.0, 21.0)),
    # subgiants / giants (OGLE ROT-like; Iwanek+2019,2024). Mostly at/brighter than Roman's F146 ~16 saturation.
    "giant":  dict(teff=(4000, 5200), P=(2.0, 100.0), rvar_kep=(120.0, 0.3), tau_P=(1.0, 5.0), shear=(0.0, 0.2),  frac_anti=0.25, f146=(14.0, 17.5)),
}
PERIOD_BINS = [(0.1, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 23.0), (23.0, 35.5), (35.5, 71.0)]  # 35.5 d = R/2


@dataclass
class RomanRotator:
    """One generated light curve; mirrors ``synthetic.SyntheticLightCurve`` so ``scorecard._load`` accepts it."""
    time: np.ndarray
    flux: np.ndarray
    true_period: float
    params: dict = field(default_factory=dict)
    star_id: str = ""


def draw_params(stype: str, rng: np.random.Generator, p_bin: tuple | None = None, f146: float | None = None,
                n_bands: int = 2) -> dict:
    """Draw one star's truth parameters from ``TYPE_PRIORS[stype]`` (period log-uniform within ``p_bin`` ∩ type range)."""
    pr = TYPE_PRIORS[stype]
    lo, hi = pr["P"]
    if p_bin is not None:
        lo, hi = max(lo, p_bin[0]), min(hi, p_bin[1])
        if lo >= hi:
            raise ValueError(f"period bin {p_bin} outside the {stype} range {pr['P']}")
    P = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    rvar_kep = float(pr["rvar_kep"][0] * 10 ** rng.normal(0, pr["rvar_kep"][1]))
    tau_P = float(np.exp(rng.uniform(*np.log(pr["tau_P"]))))
    s_lo, s_hi = pr["shear"]
    shear = float(rng.uniform(s_lo, s_hi)) if s_hi > 0 else 0.0
    if rng.uniform() < pr["frac_anti"]:
        shear = -float(rng.uniform(s_lo, max(s_hi / 2, s_lo + 1e-3)))
    minlat = float(rng.uniform(0, 30))
    maxlat = float(rng.uniform(minlat + 5, 60))
    lats = rng.uniform(minlat, maxlat, n_bands)
    weights = rng.dirichlet(np.ones(n_bands))
    Q1 = np.pi * tau_P  # SHO amplitude e-fold = 2Q/w0 ~ Q P / pi  ->  Q = pi * tau / P
    Q0 = float(rng.uniform(0.3, 0.8) * (Q1 - 0.5))
    mag = float(f146) if f146 is not None else float(rng.uniform(*pr["f146"]))
    return dict(stype=stype, P_eq=P, rvar_kep_mmag=rvar_kep, rvar_f146_mmag=rvar_kep * F146_AMP_FACTOR,
                tau_spot_P=tau_P, tau_spot_days=tau_P * P, Q1=float(Q1), Q0=Q0, dQ=float(Q1 - 0.5 - Q0),
                f_harm=float(rng.uniform(0.1, 0.6)), shear=shear, minlat=minlat, maxlat=maxlat,
                band_lats=[float(x) for x in lats], band_weights=[float(x) for x in weights],
                band_periods=[float(latitude_period(P, shear, x)) for x in lats],
                f146=mag, sigma_mmag=float(sigma_f146_mmag(mag) * 10 ** rng.normal(0, SIGMA_STAR_SCATTER_DEX)))


def make_rotator(params: dict, seed: int, t: np.ndarray | None = None, backend: str = "auto",
                 star_id: str | None = None) -> RomanRotator:
    """Light curve for ``params`` (from ``draw_params``) on grid ``t`` (default: WG6 season 1), deterministic in ``seed``.

    flux is relative flux - 1 (median 0), like ``load_wg6_roman``. ``params['period_range']`` is the active-band
    (P_lo, P_hi) for the DIFFROT outcome; ``params['rvar_realized_mmag']`` the realized 5-95% range of the noiseless
    signal (the GP draw scatters around the requested Rvar).
    """
    rng = np.random.default_rng(seed)
    t = roman_season_grid(1) if t is None else np.asarray(t, float)
    sigma_tot = params["rvar_f146_mmag"] / 1000.0 / RVAR_PER_SIGMA
    sig = np.zeros(t.size)
    for P_k, w_k in zip(params["band_periods"], params["band_weights"]):
        if not np.isfinite(P_k) or P_k <= 0:
            continue
        sig += sample_rotation_gp(t, sigma_tot * np.sqrt(w_k), P_k, params["Q0"], params["dQ"], params["f_harm"],
                                  rng, backend=backend)
    rel = np.clip(1.0 + sig, 1e-3, None)  # relative flux of the star (spot dimming / brightening)
    mag = params["f146"] - 2.5 * np.log10(rel) + rng.normal(0.0, params["sigma_mmag"] / 1000.0, t.size)
    flux = 10 ** (-0.4 * (mag - np.median(mag))) - 1.0
    p = dict(params)
    p.update(period=params["P_eq"], period_range=active_band_period_range(params["P_eq"], params["shear"],
                                                                         params["minlat"], params["maxlat"]),
             rvar_realized_mmag=float(1000 * np.subtract(*np.percentile(sig, [95, 5]))), seed=int(seed),
             backend=backend, sectors=[1], n_sectors=1)
    return RomanRotator(time=t, flux=flux, true_period=float(params["P_eq"]), params=p,
                        star_id=star_id or f"romrot-{seed:06d}")


def scorecard_task(lc: RomanRotator) -> dict:
    """Wrap a generated light curve as a ``scorecard.score_one_star`` synthetic task."""
    return {"kind": "synthetic", "lc": lc, "star_id": lc.star_id}


# --- prototype ------------------------------------------------------------------------------------------------------
PROTOTYPE = [  # (type, P_eq or period bin, F146) chosen to span types, periods and magnitudes
    ("F", 6.0, 18.0), ("G", 18.0, 20.0), ("G", 30.0, 18.0), ("K", 28.0, 20.5), ("K", 12.0, 16.0),
    ("M", 0.7, 20.0), ("M", 3.5, 16.0), ("active", 2.2, 20.0), ("active", 0.45, 18.0), ("giant", 22.0, 16.0),
]


def prototype_params(seed0: int = 3400) -> list:
    """The ~10 prototype stars: params with P_eq pinned to the listed value; other fields drawn from the priors."""
    out = []
    for i, (st, P, mag) in enumerate(PROTOTYPE):
        rng = np.random.default_rng(seed0 + i)
        p = draw_params(st, rng, p_bin=(P * 0.999, P * 1.001), f146=mag)
        p["P_eq"] = float(P)
        p["band_periods"] = [float(latitude_period(P, p["shear"], x)) for x in p["band_lats"]]
        p["seed"] = seed0 + i
        out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prototype", action="store_true", help="write the 10 prototype stars' parameters")
    ap.add_argument("--out", required=True, help="CSV of truth parameters (one row per star)")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "celerite2", "auto"])
    a = ap.parse_args(argv)
    import pandas as pd
    rows = prototype_params() if a.prototype else []
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(a.out) + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(a.out)
    print(f"wrote {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()

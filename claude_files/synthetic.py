"""
synthetic.py

Fast, dependency-light synthetic spotted-star light curves with a KNOWN
rotation period, sampled on a TESS-like gappy grid.

Purpose: give the test suite and benchmark scorecard ground truth that
doesn't require any real data on disk. This is NOT a replacement for the
SMARTS corpus (real TESS systematics + injected signal) -- it's a
controlled sandbox where each difficulty knob (noise, spot evolution,
inclination/double-dip morphology, gaps, systematics) can be turned
independently, so failures can be attributed to a specific cause.

Output format deliberately matches preprocessing.load_smarts_fits:
evenly-sampled `time` in days (no NaNs) and 0-centered relative `flux`
with NaN at missing cadences, ready for acf_utils.compute_acf.

Spot model
----------
Each spot k has a latitude b_k, initial longitude phi_k, a fractional
depth A_k, a Gaussian lifetime envelope (peak time tc_k, width tau_k),
and a rotation period set by solar-like differential rotation:

    P_k = P_eq / (1 - shear * sin^2(b_k))

The flux deficit of spot k is its depth times its *projected* area on
the visible hemisphere for stellar inclination i:

    mu_k(t) = max(0, cos(i) sin(b_k) + sin(i) cos(b_k) cos(2 pi t / P_k + phi_k))

which naturally produces single-dip (one dominant spot) and double-dip
(spots on opposite hemispheres -> strong P/2 power) morphologies. The
"true period" returned is P_eq, matching how SMARTS defines its injected
period.

Gap model
---------
Each TESS-like sector is two orbits of length sector_days/2, separated by
a ~1 d data downlink gap; consecutive sectors are separated by another
~1 d gap. Each orbit can optionally carry a short exponential "ramp"
systematic at its start, mimicking TESS scattered-light/thermal ramps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CADENCE_DAYS = 1800.0 / 86400.0  # 30-min, matches load_smarts_fits default timestep


@dataclass
class SyntheticLightCurve:
    """A synthetic light curve plus the parameters that generated it."""
    time: np.ndarray
    flux: np.ndarray
    true_period: float
    params: dict = field(default_factory=dict)
    star_id: str = ""


def tess_like_time_grid(
    n_sectors: int = 2,
    sector_days: float = 27.4,
    orbit_gap_days: float = 1.0,
    cadence_days: float = CADENCE_DAYS,
) -> tuple:
    """Build an evenly-spaced time grid plus a validity mask with TESS-like gaps.

    Parameters
    ----------
    n_sectors : number of back-to-back sectors.
    sector_days : length of each sector (two orbits of sector_days/2).
        Note TESS Cycle 8 uses ~55 d pointings; pass that here to mimic it.
    orbit_gap_days : length of the downlink gap ending each orbit.
    cadence_days : sampling interval.

    Returns
    -------
    time : np.ndarray, evenly spaced, starting at 0.
    valid : np.ndarray of bool, False inside gaps.
    orbit_starts : np.ndarray, start time of each orbit (days).
    """
    total = n_sectors * sector_days
    time = np.arange(0.0, total, cadence_days)
    orbit_len = sector_days / 2.0
    orbit_starts = np.arange(0.0, total, orbit_len)
    phase_in_orbit = time - orbit_starts[np.searchsorted(orbit_starts, time, side="right") - 1]
    valid = phase_in_orbit < (orbit_len - orbit_gap_days)
    return time, valid, orbit_starts


def spot_signal(
    time: np.ndarray,
    period: float,
    rng: np.random.Generator,
    n_spots: int = 2,
    amplitude: float = 0.01,
    inclination_deg: float = 70.0,
    shear: float = 0.0,
    spot_lifetime_days: float = np.inf,
) -> tuple:
    """Flux deficit from evolving spots on a differentially rotating star.

    Parameters
    ----------
    time : times (days).
    period : equatorial rotation period (days) -- the "true" period.
    rng : numpy Generator (all randomness flows through it).
    n_spots : number of spots.
    amplitude : approximate peak-to-peak modulation depth (relative flux).
    inclination_deg : stellar inclination (90 = equator-on).
    shear : differential-rotation shear (0 = solid body; Sun ~0.2).
    spot_lifetime_days : Gaussian lifetime (sigma) of each spot's envelope.
        np.inf gives static spots (strictly periodic signal).

    Returns
    -------
    signal : np.ndarray, 0-centered flux variation.
    spots : list of dict, per-spot parameters (for provenance).
    """
    inc = np.deg2rad(inclination_deg)
    t_span = float(time[-1] - time[0]) if len(time) > 1 else 1.0
    deficit = np.zeros_like(time, dtype=float)
    spots = []
    for _ in range(n_spots):
        lat = np.arcsin(rng.uniform(-0.8, 0.8))
        phi = rng.uniform(0, 2 * np.pi)
        depth = rng.uniform(0.5, 1.0)
        p_k = period / (1.0 - shear * np.sin(lat) ** 2)
        mu = np.cos(inc) * np.sin(lat) + np.sin(inc) * np.cos(lat) * np.cos(2 * np.pi * time / p_k + phi)
        mu = np.clip(mu, 0.0, None)
        if np.isfinite(spot_lifetime_days):
            tc = rng.uniform(time[0] - 0.2 * t_span, time[-1] + 0.2 * t_span)
            env = np.exp(-0.5 * ((time - tc) / spot_lifetime_days) ** 2)
        else:
            tc, env = np.nan, 1.0
        deficit += depth * mu * env
        spots.append(dict(lat_deg=float(np.rad2deg(lat)), phi=float(phi), depth=float(depth),
                          period=float(p_k), t_center=float(tc)))
    ptp = np.ptp(deficit)
    if ptp > 0:
        deficit *= amplitude / ptp
    signal = -(deficit - np.mean(deficit))
    return signal, spots


def red_noise(n: int, sigma: float, tau_days: float, rng: np.random.Generator,
              cadence_days: float = CADENCE_DAYS) -> np.ndarray:
    """AR(1) correlated noise with stationary std `sigma` and timescale `tau_days`.

    Parameters
    ----------
    n : number of samples.
    sigma : stationary standard deviation.
    tau_days : e-folding correlation time.
    rng : numpy Generator.
    cadence_days : sample spacing.

    Returns
    -------
    np.ndarray of length n.
    """
    if sigma <= 0 or n == 0:
        return np.zeros(n)
    a = np.exp(-cadence_days / tau_days)
    innov = rng.normal(0.0, sigma * np.sqrt(1 - a ** 2), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        out[i] = a * out[i - 1] + innov[i]
    return out


def make_lightcurve(
    period: float,
    amplitude: float = 0.01,
    white_noise: float = 0.002,
    red_noise_sigma: float = 0.0,
    red_noise_tau_days: float = 2.0,
    n_spots: int = 2,
    inclination_deg: float = 70.0,
    shear: float = 0.0,
    spot_lifetime_days: float = np.inf,
    n_sectors: int = 2,
    sector_days: float = 27.4,
    ramp_amplitude: float = 0.0,
    seed: int = 0,
    star_id: str = "",
) -> SyntheticLightCurve:
    """Generate one synthetic light curve with known equatorial period.

    Parameters
    ----------
    period : true (equatorial) rotation period, days.
    amplitude : peak-to-peak spot modulation depth (relative flux).
    white_noise : per-cadence Gaussian noise std.
    red_noise_sigma, red_noise_tau_days : AR(1) correlated-noise std and
        timescale (stellar granulation / residual systematics stand-in).
    n_spots, inclination_deg, shear, spot_lifetime_days : see spot_signal.
    n_sectors, sector_days : see tess_like_time_grid.
    ramp_amplitude : amplitude of an exponential ramp (e-fold 0.5 d) at the
        start of every orbit; 0 disables.
    seed : RNG seed; same seed + same args -> identical output.
    star_id : label carried through to results tables.

    Returns
    -------
    SyntheticLightCurve with NaN flux in gaps.
    """
    rng = np.random.default_rng(seed)
    time, valid, orbit_starts = tess_like_time_grid(n_sectors=n_sectors, sector_days=sector_days)
    signal, spots = spot_signal(time, period, rng, n_spots=n_spots, amplitude=amplitude,
                                inclination_deg=inclination_deg, shear=shear,
                                spot_lifetime_days=spot_lifetime_days)
    flux = signal + rng.normal(0.0, white_noise, len(time))
    flux += red_noise(len(time), red_noise_sigma, red_noise_tau_days, rng)
    if ramp_amplitude > 0:
        idx = np.searchsorted(orbit_starts, time, side="right") - 1
        flux += ramp_amplitude * np.exp(-(time - orbit_starts[idx]) / 0.5)
    flux = flux - np.mean(flux[valid])
    flux[~valid] = np.nan
    params = dict(period=period, amplitude=amplitude, white_noise=white_noise,
                  red_noise_sigma=red_noise_sigma, red_noise_tau_days=red_noise_tau_days,
                  n_spots=n_spots, inclination_deg=inclination_deg, shear=shear,
                  spot_lifetime_days=spot_lifetime_days, n_sectors=n_sectors,
                  sector_days=sector_days, ramp_amplitude=ramp_amplitude, seed=seed,
                  snr=amplitude / white_noise if white_noise > 0 else np.inf, spots=spots)
    return SyntheticLightCurve(time=time, flux=flux, true_period=float(period),
                               params=params, star_id=star_id or f"synth-{seed:06d}")


def make_suite(
    n: int,
    seed: int = 0,
    period_range: tuple = (0.2, 40.0),
    snr_range: tuple = (1.0, 20.0),
    n_sectors: int = 13,
) -> list:
    """Generate a reproducible, varied population of synthetic light curves.

    Periods are log-uniform over `period_range`; SNR (amplitude/white
    noise) is log-uniform over `snr_range`; spot count, inclination,
    shear, spot lifetime, red noise and orbit ramps are all randomized so
    the suite covers single-dip, double-dip and evolving-spot morphologies.

    Parameters
    ----------
    n : number of light curves.
    seed : master seed; light curve i uses seed*100000 + i.
    period_range : (min, max) period, days.
    snr_range : (min, max) amplitude / white_noise.
    n_sectors : sectors per light curve. Default 13 (x 27.4 d = 356 d)
        matches the full SMARTS baseline. Note the comb fit needs >= 4 ACF
        peaks inside compute_acf's 1/3-baseline lag range, i.e. baseline
        >~ 12 P, so long periods honestly fail on short baselines.

    Returns
    -------
    list[SyntheticLightCurve]
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        period = float(np.exp(rng.uniform(np.log(period_range[0]), np.log(period_range[1]))))
        snr = float(np.exp(rng.uniform(np.log(snr_range[0]), np.log(snr_range[1]))))
        white = 0.002
        out.append(make_lightcurve(
            period=period,
            amplitude=snr * white,
            white_noise=white,
            red_noise_sigma=float(rng.uniform(0.0, 0.5)) * white,
            n_spots=int(rng.integers(1, 4)),
            inclination_deg=float(np.rad2deg(np.arccos(rng.uniform(0.05, 1.0)))),
            shear=float(rng.uniform(0.0, 0.2)),
            spot_lifetime_days=float(rng.choice([np.inf, rng.uniform(2, 10) * period])),
            n_sectors=n_sectors,
            ramp_amplitude=float(rng.choice([0.0, rng.uniform(0.5, 2.0) * white])),
            seed=seed * 100000 + i,
            star_id=f"synth-s{seed}-{i:05d}",
        ))
    return out

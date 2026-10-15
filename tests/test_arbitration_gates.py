"""T18: arbitration / reliability gates in fit_rotation_period.

Failure modes targeted (see reports/2026-10-14_t18-arbitration-gates/):
1. Short (~0.2 d) "noise combs" on red noise or low-SNR stars pass every
   comb-shape gate and win on redchi. Their ACF shows no consistent
   P-vs-P/2 structure: some h(kP) - h((k-1/2)P) terms are negative
   (half-lag consistency gate).
2. Few-peak (long-P, 2-3 teeth) false positives on red noise: their raw ACF
   barely rises at P (few-peak ACF-height guard).
3. Clean long-period stars lost to such a noise comb on redchi.

The light-curve parameters are frozen here (copied from
synthetic.make_suite(seed=0) stars 1/64 and pb_scan dev noise seeds), so
these tests don't depend on make_suite's sampling.
"""

import numpy as np
import pytest

from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf
from comb_fit import fit_rotation_period
from evaluation import CORRECT, DIFFROT, classify_period
from guesses import gather_initial_guesses
from synthetic import make_lightcurve


def _run(lc, **fit_kwargs):
    """Scorecard-equivalent pipeline: two-view Stage 1 -> pair-count-weighted comb fit."""
    lags, acf, counts = compute_acf(lc.time, lc.flux, return_counts=True)
    lags1, acf1 = compute_acf(lc.time, lc.flux, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    g, _ = gather_initial_guesses(lc.time, lc.flux, lags1, acf1, n_guesses=10, long_view=(lags, acf))
    res = fit_rotation_period(lags, acf, g, acf_counts=counts, **fit_kwargs)
    return res, (res.best_fit.P if res.success else None)


def _red_noise(n_sectors, seed):
    """pb_scan-style red-noise-only light curve (white 0.001 + AR(1) 0.001, tau 2 d)."""
    return make_lightcurve(1.0, amplitude=0.0, white_noise=0.001, n_sectors=n_sectors, seed=seed,
                           red_noise_sigma=0.001, red_noise_tau_days=2.0)


# synthetic.make_suite(200, seed=0) star parameters (frozen)
STAR_64 = dict(period=37.470569395218945, amplitude=0.006942694476116573, white_noise=0.002,
               red_noise_sigma=0.00018266865139080879, n_spots=2, inclination_deg=31.417259563647246,
               shear=0.0, spot_lifetime_days=370.8187499525049, minlat_deg=7.987090326120394,
               maxlat_deg=15.29279919886684, n_sectors=13, seed=64)
STAR_1 = dict(period=18.790699807817617, amplitude=0.002211698057765234, white_noise=0.002,
              red_noise_sigma=0.0007296554464299441, n_spots=1, inclination_deg=24.778593646068227,
              shear=-0.34790544050625977, spot_lifetime_days=np.inf, minlat_deg=11.988475621495391,
              maxlat_deg=43.62264176447401, n_sectors=13, seed=1)


# --------------------------------------------------------------------------
# helper: half-lag contrast terms (shared module)
# --------------------------------------------------------------------------

def test_half_lag_terms_mean_is_the_ml_feature():
    """alias_metrics.half_lag_contrast_terms averages to exactly acf_half_lag_contrast."""
    from alias_metrics import acf_harmonic_contrasts, half_lag_contrast_terms
    rng = np.random.default_rng(3)
    t = np.arange(0, 80, 1 / 48.)
    f = 0.01 * np.sin(2 * np.pi * t / 4.3) + rng.normal(0, 0.003, t.size)
    lags, acf = compute_acf(t, f)
    for P in (4.3, 2.15, 8.6, 0.37):
        terms = half_lag_contrast_terms(lags, acf, P)
        assert float(np.mean(terms)) == acf_harmonic_contrasts(lags, acf, P)["acf_half_lag_contrast"]
    assert np.all(half_lag_contrast_terms(lags, acf, 4.3) > 0)  # true P: every half-lag sits in a trough
    assert np.any(half_lag_contrast_terms(lags, acf, 8.6) < 0)  # 2P alias: half-lags ARE true peaks


def test_half_lag_terms_nan_safe():
    """Degenerate inputs give an empty term array, never an exception."""
    from alias_metrics import half_lag_contrast_terms
    lags = np.linspace(0, 10, 200)
    acf = np.cos(lags)
    assert half_lag_contrast_terms(lags, acf, np.nan).size == 0
    assert half_lag_contrast_terms(lags, acf, -1.0).size == 0
    assert half_lag_contrast_terms(lags, acf, 50.0).size == 0  # no window fits


# --------------------------------------------------------------------------
# end-to-end failure modes
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("n_sectors,seed", [(1, 901501), (1, 901513), (2, 902511)])
def test_red_noise_short_noise_comb_declined(n_sectors, seed):
    """Red noise used to get a confident ~0.2 d period (open issue 10); now declined."""
    res, Pm = _run(_red_noise(n_sectors, seed))
    assert not res.success, f"claimed P={Pm} on red noise: {res.message}"


@pytest.mark.slow
@pytest.mark.parametrize("n_sectors,seed", [(2, 902501), (2, 902503)])
def test_red_noise_few_peak_false_positive_declined(n_sectors, seed):
    """Red noise used to get a long (13-16 d) few-peak period (open issue 14b); now declined."""
    res, Pm = _run(_red_noise(n_sectors, seed))
    assert not res.success, f"claimed P={Pm} on red noise: {res.message}"


@pytest.mark.slow
@pytest.mark.parametrize("params", [STAR_64, STAR_1], ids=["star64_P37d", "star1_P19d_snr1.1"])
def test_clean_candidate_not_lost_to_noise_comb(params):
    """These stars used to answer ~0.22-0.24 d (a noise comb with lower redchi)
    although their true-period candidate passed every gate; now they succeed."""
    lc = make_lightcurve(**params)
    res, Pm = _run(lc)
    assert classify_period(Pm, lc.true_period, period_range=lc.params["period_range"]) in (CORRECT, DIFFROT), \
        f"P={Pm}: {res.message}"


@pytest.mark.slow
def test_new_gates_can_be_switched_off():
    """Backward compatibility: with the T18 gates disabled, the old answer comes back."""
    res, Pm = _run(_red_noise(1, 901501), half_lag_gate=False, few_peaks_min_acf_height=None)
    assert res.success and Pm < 0.5


@pytest.mark.slow
def test_gate_diagnostics_recorded():
    """Every fitted candidate carries the contrast diagnostics and its base-gate verdict."""
    res, _ = _run(_red_noise(1, 901501))
    assert res.candidates
    for c in res.candidates:
        for k in ("acf_half_lag_contrast", "half_lag_min_term", "acf_height_at_P"):
            assert k in c.diagnostics
        assert isinstance(c.passed_base_gates, bool)
        assert not c.passed_gates or c.passed_base_gates
    assert any(c.passed_base_gates and not c.passed_gates for c in res.candidates)


# synthetic.make_suite(200, seed=0) star 175 (frozen): a genuine SNR-1.6 star whose
# only negative half-lag term is k=1 (P/2 still on the lag-0 decay)
STAR_175 = dict(period=1.2217670118993469, amplitude=0.003289925851360166, white_noise=0.002,
                red_noise_sigma=0.0007223702289359587, n_spots=2, inclination_deg=70.92861860694224,
                shear=-0.1366301477930964, spot_lifetime_days=11.3828329466552, minlat_deg=36.54160290437184,
                maxlat_deg=57.26370569960449, n_sectors=13, seed=175)


@pytest.mark.slow
def test_fallback_tier_answers_low_contrast_star():
    """Strict default declines this weak genuine star (the documented cost);
    the optional fallback tier (half_lag_fallback_min=0.0, report option D) answers it."""
    lc = make_lightcurve(**STAR_175)
    res, _ = _run(lc)
    assert not res.success
    res_fb, Pm = _run(lc, half_lag_fallback_min=0.0)
    assert res_fb.success and "fallback" in res_fb.message
    assert classify_period(Pm, lc.true_period, period_range=lc.params["period_range"]) in (CORRECT, DIFFROT)

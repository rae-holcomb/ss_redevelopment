"""Tests for tools/headline_card.py on tiny synthetic per-star frames (no data needed)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import headline_card as hc  # noqa: E402


def _frame(true, answer, lo=None, hi=None, score=None, cov=None, best=None):
    """Build a standard frame directly (bypassing CSV loading)."""
    n = len(true)
    df = pd.DataFrame({
        "star_id": [f"s{i}" for i in range(n)],
        "true_period": true,
        "period_lo": lo if lo is not None else true,
        "period_hi": hi if hi is not None else true,
        "coverage_class": cov if cov is not None else ["single"] * n,
    })
    return hc._standard_frame(df, pd.Series(answer, dtype=float),
                              pd.Series(best if best is not None else answer, dtype=float), score)


def test_outcomes_success_vs_correct_and_claytor_windows():
    """Diffrot counts as success but not correct; within20 accepts a 15% miss that within10 rejects."""
    f = hc.add_outcomes(_frame(true=[10.0, 10.0, 10.0, 10.0],
                               answer=[10.2, 12.0, 11.5, np.nan],
                               lo=[10.0, 11.0, 10.0, 10.0], hi=[10.0, 13.0, 10.0, 10.0]))
    assert f.success.tolist() == [True, True, False, False]
    assert f.correct.tolist() == [True, False, False, False]
    assert f.within10.tolist() == [True, False, False, False]
    assert f.within20.tolist() == [True, True, True, False]
    assert f.answered.tolist() == [True, True, True, False]


def test_population_shares_sum_to_one_and_reweighting_matches_marginal():
    """Bin shares sum to 1, and population_weights reproduce them as the frame's bin marginal."""
    for pop in hc.POPULATIONS:
        s = hc.population_bin_shares(pop)
        assert sum(s.values()) == pytest.approx(1.0)
    f = _frame(true=[0.5, 2.0, 2.5, 5.0, 20.0, 50.0], answer=[np.nan] * 6)
    w = hc.population_weights(f, "loguniform")
    shares = hc.population_bin_shares("loguniform")
    got = pd.Series(w).groupby(f.period_bin.to_numpy()).sum()
    for b, v in got.items():
        assert v == pytest.approx(shares[b])
    assert hc.population_weights(f, "none").sum() == pytest.approx(1.0)


def test_precision_formula():
    """prec = fS / (fA + (1-f)FP); zero FP gives S/A independent of prevalence."""
    assert hc.precision_at(0.5, 0.2, 0.3, 0.1) == pytest.approx(0.1 / (0.15 + 0.05))
    assert hc.precision_at(0.1, 0.2, 0.25, 0.0) == pytest.approx(0.8)


def test_credible_half_uses_score_or_answers_first():
    """Scored systems take the top-scored half; unscored ones take answered stars, then misses."""
    f = hc.add_outcomes(_frame(true=[1.0, 2.0, 3.0, 4.0], answer=[1.0, np.nan, 3.0, np.nan],
                               score=[0.1, 0.9, 0.8, 0.2], best=[1.0, 2.0, 3.0, 9.0]))
    assert hc.credible_half(f).tolist() == [False, True, True, False]
    g = f.copy()
    g["score"] = np.nan
    assert hc.credible_half(g).tolist() == [True, False, True, False]


def test_compute_card_assigned_uses_coverage_weights_and_fp():
    """R1 is the coverage-weighted success at P <= 30 d; full-year leaves R1 empty."""
    true = [5.0, 5.0, 5.0, 50.0]
    f = hc.add_outcomes(_frame(true=true, answer=[5.0, np.nan, np.nan, 50.0],
                               cov=["single", "full", "full", "full"]))
    manifest = pd.DataFrame({"star_id": f.star_id, "coverage_weight": [3.0, 1.0, 1.0, 1.0],
                             "coverage_class": ["single", "full", "full", "full"]})
    card = hc.compute_card(f, "assigned", manifest, population="none", fp_rate=0.0, n_boot=0)
    v = card["values"]
    assert v["R1_success_tess_cov_P30"] == pytest.approx(3.0 / 5.0)
    assert v["R0_success_raw"] == pytest.approx(0.5)
    assert v["R4_precision"]["0.1"] == pytest.approx(1.0)
    fy = hc.compute_card(f, "fullyear", population="none", n_boot=5)
    assert fy["values"]["R1_success_tess_cov_P30"] is None and fy["values"]["R4_precision"] is None
    assert fy["se"]["R0_success_raw"] >= 0


def test_negatives_fp_rate_reweights_to_real_coverage_mix():
    """FP rate weights each coverage class by its real share implied by coverage_weight."""
    neg = pd.DataFrame({"coverage_class": ["single", "single", "full", "full"],
                        "hand_answered": [True, False, False, False]})
    manifest = pd.DataFrame({"coverage_class": ["single", "full"], "coverage_weight": [3.0, 1.0]})
    assert hc.negatives_fp_rate(neg, "hand_answered", "assigned", manifest) == pytest.approx(0.5 * 0.75)
    assert hc.negatives_fp_rate(neg, "hand_answered", "fullyear", None) == pytest.approx(0.25)


def test_dev_medium_population_is_default_and_matches_manifest():
    """The default population is dev_medium's own period-bin mix (Rae, 2026-11-12), read from the manifest."""
    assert hc.DEFAULT_POPULATION == "dev_medium"
    s = hc.population_bin_shares("dev_medium")
    assert sum(s.values()) == pytest.approx(1.0)
    tp = pd.read_csv(hc.DEV_MEDIUM_MANIFEST, usecols=["true_period"])["true_period"]
    expect = tp.map(hc.period_bin).value_counts(normalize=True)
    for b, v in s.items():
        assert v == pytest.approx(expect.get(b, 0.0))
    assert s["<1d"] > 0.1  # fast rotators keep real weight, unlike mcq14_like

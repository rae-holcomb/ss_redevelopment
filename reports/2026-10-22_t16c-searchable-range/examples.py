"""T16c example stars (dm_assigned before run): light curve and Stage 2 ACF
with R, R/2, the true period band, and every candidate the fit was given
(in reach: blue ticks; beyond reach, skipped after T16c: red ticks).

Writes figures/examples.png and data/examples.csv. Re-runs Stage 1 + 2 per
star (~3 s each).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import profile_fit as pf  # noqa: E402  (sets sys.path, pins BLAS)
import analysis as an  # noqa: E402
import coverage as cov  # noqa: E402
from comb_fit import fit_rotation_period  # noqa: E402


def pick_examples() -> pd.DataFrame:
    """One star per failure mode, chosen by rule (first match in star_id order)."""
    ps = an.add_reach_cols(an.load("before", "dm_assigned")[0]).sort_values("star_id")
    pm = pd.read_csv(an.RUNS / "before" / "dm_assigned" / "per_method.csv")
    ph0 = set(pm[(pm.method == "pairwise_histogram") & (pm.n_candidates == 0)].star_id)
    short = ps.coverage_class.isin(["single", "consecutive-2", "gapped-2"])
    rules = [
        ("success near the limit (P/R 0.4-0.5)", (ps.truth_over_R.between(0.4, 0.5)) & ps.success),
        ("declined inside reach (P/R 0.3-0.5)", (ps.truth_over_R.between(0.3, 0.5)) & ~ps.answered),
        ("wrong on an unsearchable star (P/R > 0.5)", (ps.truth_over_R > 0.5) & ps.wrong & short),
        ("unsearchable, correctly declined", (ps.truth_over_R > 1.0) & ~ps.answered & short),
        ("pairwise_histogram: no candidates", ps.star_id.isin(ph0) & (ps.truth_over_R < 0.3) & short),
        ("gapped-2: reach = one sector", (ps.coverage_class == "gapped-2") & (ps.truth_over_R.between(0.5, 1.0))),
    ]
    rows = []
    for name, sel in rules:
        hit = ps[sel]
        if len(hit):
            rows.append(hit.iloc[0].to_dict() | {"case": name})
    return pd.DataFrame(rows)


def main():
    """Plot each example: light curve (top) and ACF with candidates (bottom)."""
    ex = pick_examples()
    man = pd.read_csv(an.ROOT / "datasets/dev_medium.csv").set_index("star_id")
    ncol = 3
    nrow = 2 * int(np.ceil(len(ex) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.9 * nrow), squeeze=False)
    for j, (_, e) in enumerate(ex.iterrows()):
        r = man.loc[e.star_id]
        t, f, lags, acf, counts, pooled = pf.pooled_inputs(an.ROOT / "data/smarts_dev" / r.path,
                                                           cov.parse_pattern(r.sectors))
        res = fit_rotation_period(lags, acf, pooled, acf_counts=counts)
        sr = cov.searchable_range(t, f)
        a0, a1 = axes[2 * (j // ncol), j % ncol], axes[2 * (j // ncol) + 1, j % ncol]
        a0.plot(t, f, ",", color="0.3")
        a0.set_title(f"{e.case}\n{e.star_id[-6:]} {e.coverage_class}, P={e.true_period:.2f} d",
                     fontsize=9)
        a0.set_xlabel("time (d)", fontsize=8)
        a0.set_ylabel("rel. flux", fontsize=8)
        a1.plot(lags, acf, lw=0.7, color="0.2")
        a1.axvline(sr["reach"], color="k", lw=1, label=f"R = {sr['reach']:.1f} d")
        a1.axvline(sr["p_max"], color="C0", lw=1, ls="--", label="R/2")
        a1.axvspan(e.P_lo, e.period_hi if np.isfinite(e.period_hi) else e.true_period, color="C2", alpha=0.3,
                   label="true P band")
        for c in res.candidates + res.failed_candidates:
            a1.axvline(c.period, ymin=0.92, ymax=1.0, color="C0", lw=1)
        for c in res.beyond_reach_candidates:
            a1.axvline(c.period, ymin=0.92, ymax=1.0, color="C3", lw=1.5)
        ans = f"{res.best_fit.P:.2f} d" if res.success else "declined"
        a1.set_title(f"answer: {ans} ({e.final_outcome}); {len(res.beyond_reach_candidates)} beyond reach",
                     fontsize=9)
        a1.set_xlim(0, max(lags[-1], e.true_period * 1.2))
        a1.set_xlabel("lag (d); ticks at top: candidates (blue fit, red beyond reach)", fontsize=8)
        a1.set_ylabel("ACF", fontsize=8)
        if j == 0:
            a1.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "examples.png", dpi=120)
    ex[["case", "star_id", "coverage_class", "true_period", "R", "truth_over_R", "final_period",
        "final_outcome"]].to_csv(HERE / "data" / "examples.csv", index=False)


if __name__ == "__main__":
    main()

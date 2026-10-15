"""figs.py -- figure functions for the T18 report notebook. Each takes the small
tables in data/ (or re-simulates a handful of frozen example stars) and writes
a PNG to figures/."""
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))
FIG = Path("figures")
GOOD, BAD, NOISE, ALIAS = "#2b6cb0", "#dd6b20", "#4a5568", "#9f7aea"


def _sl(ax, axis="both", lin=0.01):
    """Symmetric-log axes (contrast values straddle zero)."""
    if axis in ("x", "both"):
        ax.set_xscale("symlog", linthresh=lin)
    if axis in ("y", "both"):
        ax.set_yscale("symlog", linthresh=lin)


def fig_lost_categories(lost: pd.DataFrame):
    """Stacked bars: why each non-success seed-0 star was lost, split by its final outcome."""
    order = ["never_proposed", "succ_failed_gate", "lost_on_selection"]
    labels = ["no fitted success\ncandidate (Stage 1)", "success candidate\nfailed a gate", "success candidate passed,\nanother one selected"]
    ct = pd.crosstab(lost.category, lost.outcome).reindex(order).fillna(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    left = np.zeros(len(ct))
    cols = {"none": "#a0aec0", "wrong": BAD, "half": ALIAS, "double": "#d53f8c"}
    for o in [c for c in ("none", "wrong", "half", "double") if c in ct.columns]:
        ax[0].barh(labels, ct[o], left=left, color=cols[o], label=o)
        left += ct[o].to_numpy()
    for i, v in enumerate(left):
        ax[0].text(v + 0.3, i, f"{int(v)}", va="center")
    ax[0].invert_yaxis()
    ax[0].set_xlabel("stars (synthetic:n=200,seed=0, pre-T18)")
    ax[0].legend(title="final outcome", fontsize=8)
    g = lost[lost.category == "succ_failed_gate"].failed_gates.value_counts()
    ax[1].barh(g.index, g.values, color="#718096")
    ax[1].invert_yaxis()
    ax[1].set_xlabel("stars")
    ax[1].set_title("gate(s) failed by the best success candidate", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_lost_categories.png", dpi=130)
    return fig


def fig_contrast_gate(sel: pd.DataFrame):
    """Pre-T18 selected candidates: min half-lag term vs mean contrast. Gate passes x > 0 or y >= 0.1."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for ax, split, sets in ((axs[0], "dev (tuning)", ("synth_s0", "pb_dev", "pb_dev_noise2")),
                            (axs[1], "validation (held out)", ("synth_s1", "pb_eval"))):
        x = sel[sel.dataset.isin(sets)]
        sig = x[x.kind != "noise"]
        good = sig.final_outcome.isin(["correct", "diffrot"])
        alias = sig.final_outcome.isin(["half", "double"])
        fp = x[(x.kind == "noise")]
        kw = dict(s=14, alpha=0.75, lw=0)
        ax.scatter(sig[good].hlc_min_term, sig[good].acf_half_lag_contrast, c=GOOD, label=f"success ({good.sum()})", **kw)
        ax.scatter(sig[alias].hlc_min_term, sig[alias].acf_half_lag_contrast, c=ALIAS, marker="s", s=22,
                   label=f"half/double alias ({alias.sum()})")
        wr = sig.final_outcome == "wrong"
        ax.scatter(sig[wr].hlc_min_term, sig[wr].acf_half_lag_contrast, c=BAD, marker="D", s=22, label=f"wrong ({wr.sum()})")
        ax.scatter(fp.hlc_min_term, fp.acf_half_lag_contrast, c=NOISE, marker="x", s=26,
                   label=f"noise false positive ({len(fp)})")
        _sl(ax)
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0.1, color="k", lw=0.8, ls="--")
        ax.fill_betweenx([-2, 0.1], -2, 0, color="#fed7d7", alpha=0.35, zorder=0)
        ax.set_xlim(-0.6, 2)
        ax.set_ylim(-0.6, 2)
        ax.text(-0.5, -0.4, "rejected by\nhalf-lag gate", fontsize=8, color="#9b2c2c")
        ax.set_title(f"pre-T18 selected answers, {split}", fontsize=10)
        ax.set_xlabel("min over k of  h(kP) - h((k-1/2)P)")
        ax.legend(fontsize=7, loc="upper left")
    axs[0].set_ylabel("acf_half_lag_contrast (mean of the terms)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_contrast_gate.png", dpi=130)
    return fig


def fig_few_peak_guard(few: pd.DataFrame):
    """Raw ACF height at P for few-peak candidates that pass the comb gates (pb_scan + synth)."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    groups = [("noise (false positive)", few.kind == "noise", NOISE),
              ("signal, correct", few.label == "correct", GOOD),
              ("signal, half/double alias", few.label.isin(["half", "double"]), ALIAS),
              ("signal, wrong", few.label == "wrong", BAD)]
    rng = np.random.default_rng(0)
    for i, (lab, m, col) in enumerate(groups):
        v = few[m].acf_height_at_P
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=12, c=col, alpha=0.7, lw=0)
        ax.text(1.06, i, f"n={len(v)}", va="center", fontsize=8)
    ax.axvline(0.4, color="k", ls="--", lw=1)
    ax.text(0.41, 3.35, "guard: >= 0.4", fontsize=8)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlabel("raw ACF max within +/- 0.1 P of lag P  (acf_height_at_P)")
    ax.set_xlim(-0.4, 1.15)
    ax.set_title("few-peak candidates passing the comb gates (all caches, pre-T18)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_few_peak_guard.png", dpi=130)
    return fig


def fig_ablation(abl: pd.DataFrame, rules: list, short: dict):
    """Bar chart: success / bad answers / declines on synth s0 & s1, and red-noise FPs on pb dev/eval."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 0.45 * len(rules) + 1.4), sharey=True)
    y = np.arange(len(rules))
    for ax, ds, ttl in ((axs[0], "synth_s0", "synth seed 0 (dev)"), (axs[1], "synth_s1", "synth seed 1 (validation)")):
        x = abl[abl.dataset == ds].set_index("rule").loc[rules]
        bad = x.wrong + x.alias
        ax.barh(y - 0.27, x.success, 0.27, color=GOOD, label="success")
        ax.barh(y, bad, 0.27, color=BAD, label="wrong + alias")
        ax.barh(y + 0.27, x.none, 0.27, color="#a0aec0", label="declined")
        for yy, a, b, c in zip(y, x.success, bad, x.none):
            ax.text(a + 0.01, yy - 0.27, f"{a:.3f}", va="center", fontsize=7)
            ax.text(b + 0.01, yy, f"{b:.3f}", va="center", fontsize=7)
            ax.text(c + 0.01, yy + 0.27, f"{c:.3f}", va="center", fontsize=7)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlim(0, 0.95)
    axs[0].legend(fontsize=7, loc="lower right")
    for ds, off, col in (("pb_dev", -0.2, "#718096"), ("pb_eval", 0.2, "#2d3748")):
        x = abl[abl.dataset == ds].set_index("rule").loc[rules]
        n = x.n_noise_each.iloc[0]
        axs[2].barh(y + off, x.fp_red / n, 0.38, color=col, label=f"{ds} ({int(n)} red LCs)")
        for yy, v in zip(y + off, x.fp_red):
            axs[2].text(v / n + 0.01, yy, f"{int(v)}", va="center", fontsize=7)
    axs[2].set_title("pb_scan red-noise false-positive rate", fontsize=10)
    axs[2].legend(fontsize=7, loc="lower right")
    axs[0].set_yticks(y, [short.get(r, r) for r in rules], fontsize=8)
    axs[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.png", dpi=130)
    return fig


def _pipeline(params: dict, **fit_kwargs):
    """Simulate a frozen example star and run the scorecard pipeline. Returns (lc, lags, acf, result)."""
    warnings.simplefilter("ignore")
    from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf
    from comb_fit import fit_rotation_period
    from guesses import gather_initial_guesses
    from synthetic import make_lightcurve
    p = {k: (np.inf if v is None else v) for k, v in params.items()}
    lc = make_lightcurve(**p)
    lags, acf, counts = compute_acf(lc.time, lc.flux, return_counts=True)
    lags1, acf1 = compute_acf(lc.time, lc.flux, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    g, _ = gather_initial_guesses(lc.time, lc.flux, lags1, acf1, n_guesses=10, long_view=(lags, acf))
    return lc, lags, acf, fit_rotation_period(lags, acf, g, acf_counts=counts, **fit_kwargs)


def _comb(ax, fit, color, label):
    """Overplot a candidate's fitted parabolae (as in plotting.plot_comb_fit)."""
    for i, w in enumerate(fit.windows):
        pk = fit.per_peak[w.n]
        xx = np.linspace(w.lag_lo, w.lag_hi, 60)
        ax.plot(xx, pk["height"] - pk["curvature"] * (xx - pk["center"]) ** 2, color=color, lw=2,
                label=label if i == 0 else None)


def _terms_text(c):
    """Half-lag diagnostics string for a CandidateResult."""
    d = c.diagnostics
    return (f"hlc={d['acf_half_lag_contrast']:+.3f}, min term={d['half_lag_min_term']:+.3f}, "
            f"ACF(P)={d['acf_height_at_P']:.2f}")


def fig_examples(params_path="data/example_params.json"):
    """One row per example: the ACF with the pre-T18 pick's comb (orange) and the
    best success / closest candidate's comb (blue); right panel zooms near lag 0."""
    from evaluation import classify_period
    ex = json.loads(Path(params_path).read_text())
    titles = {
        "synth-s0-00064": "1) synth-s0-00064, P=37.5 d, SNR 3.5: lost to a 0.24 d noise comb -> now CORRECT",
        "pb-noise-red-1s-901501": "2) red noise, 1 sector: 0.22 d false positive -> now DECLINED",
        "pb-noise-red-2s-902501": "3) red noise, 2 sectors: 16 d few-peak false positive -> now DECLINED",
        "synth-s0-00175": "4) synth-s0-00175, P=1.22 d, SNR 1.6: genuine low-contrast answer -> now DECLINED (cost)",
        "synth-s0-00144": "5) synth-s0-00144, P=0.53 d, SNR 14: declined before and after (vertex gate)",
    }
    fig, axs = plt.subplots(len(titles), 2, figsize=(14, 3.0 * len(titles)), gridspec_kw=dict(width_ratios=[2.2, 1]))
    rows = []
    for r, sid in enumerate(titles):
        lc, lags, acf, old = _pipeline({k: v for k, v in ex[sid].items() if k != "star_id"},
                                       half_lag_gate=False, few_peaks_min_acf_height=None)
        _, _, _, new = _pipeline({k: v for k, v in ex[sid].items() if k != "star_id"})
        Ptrue = lc.true_period if lc.params["amplitude"] > 0 else None
        pr = lc.params.get("period_range")
        lab = (lambda P: classify_period(P, Ptrue, period_range=pr)) if Ptrue else (lambda P: "noise")
        sel_old = old.candidates[0] if old.success else None
        succ = [c for c in new.candidates if Ptrue and lab(c.fit.P) in ("correct", "diffrot")]
        other = succ[0] if succ else next((c for c in new.candidates if c.fit.P > 1.5 * (sel_old.fit.P if sel_old else 0)), None)
        for ax, zoom in ((axs[r, 0], False), (axs[r, 1], True)):
            ax.plot(lags, acf, color="0.4", lw=0.8)
            ax.axhline(0, color="0.85", lw=0.6)
            if sel_old is not None:
                _comb(ax, sel_old.fit, "#dd6b20", f"pre-T18 pick P={sel_old.fit.P:.3g} ({lab(sel_old.fit.P)})")
            if other is not None and other is not sel_old:
                _comb(ax, other.fit, GOOD, f"{'success cand.' if succ else 'other cand.'} P={other.fit.P:.3g}")
            if zoom:
                P = sel_old.fit.P if sel_old is not None else (other.fit.P if other else 1)
                ax.set_xlim(0, 7 * P)
                lo = acf[(lags > 0.3 * P) & (lags < 7 * P)]
                ax.set_ylim(lo.min() - 0.05, lo.max() + 0.05)
                for k in range(1, 7):
                    ax.axvline(k * P, color="#dd6b20", lw=0.5, alpha=0.6)
                    ax.axvline((k - 0.5) * P, color="#dd6b20", lw=0.5, ls=":", alpha=0.6)
                ax.set_title("zoom on the pre-T18 pick: kP (solid), (k-1/2)P (dotted)", fontsize=8)
            else:
                ax.set_title(titles[sid], fontsize=9, loc="left")
                ax.legend(fontsize=7, loc="upper right")
                txt = []
                if sel_old is not None:
                    c = next(c for c in new.candidates if abs(c.fit.P - sel_old.fit.P) < 1e-9 * sel_old.fit.P) \
                        if any(abs(c.fit.P - sel_old.fit.P) < 1e-9 * sel_old.fit.P for c in new.candidates) else None
                    if c is not None:
                        txt.append("pick: " + _terms_text(c))
                if other is not None and other is not sel_old:
                    txt.append("blue: " + _terms_text(other))
                txt.append("T18 answer: " + (f"P={new.best_fit.P:.4g} ({lab(new.best_fit.P)})" if new.success else "declined"))
                ax.text(0.01, 0.04, "\n".join(txt), transform=ax.transAxes, fontsize=7, va="bottom",
                        bbox=dict(fc="white", ec="0.8", alpha=0.9))
        axs[r, 0].set_xlabel("lag (d)")
        rows.append(dict(star=sid, pre_T18=(f"{sel_old.fit.P:.4g}" if sel_old else "declined"),
                         T18=(f"{new.best_fit.P:.4g}" if new.success else "declined")))
    fig.tight_layout()
    fig.savefig(FIG / "fig_examples.png", dpi=110)
    return fig, pd.DataFrame(rows)


def fig_bins(before: dict, after: dict):
    """Success by period bin, before vs after, for each scorecard run pair {label: metrics dict}."""
    fig, axs = plt.subplots(1, len(before), figsize=(6 * len(before), 3.4), sharey=True)
    axs = np.atleast_1d(axs)
    bins = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
    for ax, k in zip(axs, before):
        b = [before[k]["final"]["by_bin"][x]["success"] for x in bins]
        a = [after[k]["final"]["by_bin"][x]["success"] for x in bins]
        n = [before[k]["final"]["by_bin"][x]["n"] for x in bins]
        xx = np.arange(len(bins))
        ax.bar(xx - 0.2, b, 0.4, color="#a0aec0", label="before (main 589cb8d)")
        ax.bar(xx + 0.2, a, 0.4, color=GOOD, label="after (T18)")
        for i in range(len(bins)):
            ax.text(xx[i] - 0.2, b[i] + 0.01, f"{b[i]:.2f}", ha="center", fontsize=7)
            ax.text(xx[i] + 0.2, a[i] + 0.01, f"{a[i]:.2f}", ha="center", fontsize=7)
        ax.set_xticks(xx, [f"{x}\n(n={m})" for x, m in zip(bins, n)])
        ax.set_title(k, fontsize=10)
        ax.set_ylim(0, 1.1)
    axs[0].set_ylabel("success (correct + diffrot)")
    axs[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_success_by_bin.png", dpi=130)
    return fig

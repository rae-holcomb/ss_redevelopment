"""
figs.py -- figure functions for the T16b report (called from the notebook).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gate_lib as gl

HERE = Path(__file__).resolve().parent
COV_ORDER = ["full", "consecutive-2", "gapped-2", "single"]
CLS_COLORS = {"success": "#2a7ab9", "wrong": "#d1495b", "alias": "#edae49", "noise": "#6c6c6c"}


def load_examples(path: Path = HERE / "data" / "examples"):
    """(records, arrays) saved by examples.py."""
    recs = json.loads(path.with_suffix(".json").read_text())
    arr = np.load(path.with_suffix(".npz"))
    return recs, arr


def _gate_text(c: dict) -> str:
    """Compact multi-line per-gate summary for one candidate (x = failed)."""
    lines = []
    for g in gl.GATES:
        d = c["gates"][g]
        if not d["enabled"]:
            continue
        mark = "ok" if (d["passed"] or d.get("fallback")) else "FAIL"
        extra = f" (mean {d['contrast']:.3f})" if g == "half_lag_consistency" else ""
        lines.append(f"{gl.GATE_SHORT[g]:>11s} {d['value']:7.3f} {d['op']} {d['threshold']:.2f}{extra}  {mark}")
    return "\n".join(lines)


def plot_example(rec: dict, arr, i: int, ax_lc, ax_acf):
    """Light curve + ACF with the best success candidate's comb (blue) and the
    best-redchi candidate's comb (grey), annotated with per-gate values."""
    t, f = arr[f"{i}_time"], arr[f"{i}_flux"]
    lags, acf = arr[f"{i}_lags"], arr[f"{i}_acf"]
    ax_lc.plot(t, f, ",", color="0.3")
    ax_lc.set_title(f"{rec['dataset']}  {rec['star_id']}  P_eq={rec['true_period']:.3f} d  sectors={rec['sectors']}",
                    fontsize=9)
    ax_lc.set_xlabel("time [d]", fontsize=8)
    ax_acf.plot(lags, acf, color="0.35", lw=0.8)
    ax_acf.axhline(0, color="0.85", lw=0.6)
    cands = rec["candidates"]
    succ = [c for c in cands if c["outcome"] in gl.SUCCESS]
    best = cands[0]
    bs = min(succ, key=lambda c: c["redchi"] if np.isfinite(c["redchi"]) else np.inf) if succ else None
    xmax = lags[-1]
    for c, col, lab in ((best, "0.55", "lowest redchi"), (bs, CLS_COLORS["success"], "best success cand.")):
        if c is None:
            continue
        for w in c["windows"]:
            x = np.linspace(w["lo"], w["hi"], 50)
            ax_acf.plot(x, w["height"] - w["curvature"] * (x - w["center"]) ** 2, color=col, lw=1.6)
        for k in np.arange(0.5, 6.5, 0.5):
            if k * c["P_fit"] < xmax:
                ax_acf.axvline(k * c["P_fit"], color=col, lw=0.6, ls="-" if k % 1 == 0 else ":", alpha=0.7)
        ax_acf.plot([], [], color=col, label=f"{lab}: P={c['P_fit']:.3f} ({c['outcome']})")
    ax_acf.set_xlim(0, min(xmax, 8 * (bs or best)["P_fit"]))
    lo = acf[lags <= ax_acf.get_xlim()[1]]
    ax_acf.set_ylim(min(lo.min(), -0.2), max(lo[5:].max() * 1.3, 0.3))
    ax_acf.legend(fontsize=7, loc="upper right")
    ax_acf.set_xlabel("lag [d]", fontsize=8)
    if bs is not None:
        ax_acf.text(0.01, 0.02, "best success cand.:\n" + _gate_text(bs), transform=ax_acf.transAxes,
                    fontsize=6.5, family="monospace", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))


def fig_examples(recs, arr, out: Path | None = None):
    """Grid of example stars (one row each: light curve | ACF + combs)."""
    n = len(recs)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.9 * n), gridspec_kw=dict(width_ratios=[1, 2.2]))
    for i, rec in enumerate(recs):
        plot_example(rec, arr, i, axes[i, 0], axes[i, 1])
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=110)
    return fig


# ------------------------------------------------------------------ tables

SUCCESS_DEFS = {
    "loose": "fitted P scored success (correct or diffrot, evaluation.classify_period)",
    "strict": "fitted P within REL_TOL of P_eq (outcome 'correct')",
    "matched": "fitted P within 3% of the same star's full-year final period (full-year success only)",
}


def rescue_candidates(stars: pd.DataFrame, cands: pd.DataFrame, definition: str = "loose") -> pd.DataFrame:
    """For every DECLINED star, its lowest-redchi candidate that meets the
    success `definition` (see SUCCESS_DEFS), with the star's columns merged in.
    'matched' is only defined for the synthetic coverage runs (needs the
    full-year run of the same stars)."""
    declined = stars[stars.final_outcome.eq("none")][["dataset", "star_id", "period_bin", "coverage_class",
                                                       "n_sectors", "true_period"]]
    c = cands.merge(declined[["dataset", "star_id"]], on=["dataset", "star_id"])
    if definition == "loose":
        m = c.outcome.isin(gl.SUCCESS)
    elif definition == "strict":
        m = c.outcome.eq("correct")
    elif definition == "matched":
        full = stars[(stars.dataset == "full") & stars.final_outcome.isin(gl.SUCCESS)].set_index("star_id").final_period
        pf = c.star_id.map(full).where(c.dataset.isin(gl.SYNTH))
        m = c.outcome.isin(gl.SUCCESS) & ((c.P_fit / pf - 1).abs() < 0.03)
    else:
        raise ValueError(definition)
    b = gl.best_per_star(c, m)
    return b.drop(columns=["period_bin"], errors="ignore").merge(declined, on=["dataset", "star_id"])


def gate_flags(b: pd.DataFrame) -> pd.DataFrame:
    """Boolean column per gate: did this candidate fail it (enabled gates only)."""
    fg = b.failed_gates.fillna("").str.split(";")
    out = pd.DataFrame({g: fg.apply(lambda x, g=g: g in x) for g in gl.GATES}, index=b.index)
    out["n_failed"] = out[list(gl.GATES)].sum(axis=1)
    return out


def gate_table(b: pd.DataFrame, by=("dataset", "period_bin")) -> pd.DataFrame:
    """Counts of rescue candidates failing each gate ('any') and failing only
    that gate ('sole'), grouped by `by`."""
    f = gate_flags(b)
    f[list(by)] = b[list(by)].to_numpy()
    any_ = f.groupby(list(by))[list(gl.GATES)].sum()
    sole = f[f.n_failed == 1].groupby(list(by))[list(gl.GATES)].sum().reindex(any_.index, fill_value=0)
    n = f.groupby(list(by)).size().rename("n_stars")
    any_.columns = pd.MultiIndex.from_product([["any"], [gl.GATE_SHORT[g] for g in gl.GATES]])
    sole.columns = pd.MultiIndex.from_product([["sole"], [gl.GATE_SHORT[g] for g in gl.GATES]])
    n.index = any_.index
    return pd.concat([pd.concat({"": n.to_frame()}, axis=1), any_, sole], axis=1)


# ------------------------------------------------------------------ figures

STATS = [  # (column, label, gate whose threshold to mark, threshold column)
    ("mean_height_snr_value", "mean height SNR", "mean_height_snr_threshold"),
    ("half_lag_consistency_value", "half-lag min term", "half_lag_consistency_threshold"),
    ("half_lag_contrast", "half-lag mean contrast", None),
    ("frac_vertex_in_window_value", "frac vertex in window", "frac_vertex_in_window_threshold"),
    ("frac_positive_heights_value", "frac positive heights", "frac_positive_heights_threshold"),
    ("few_peaks_height_value", "ACF height at P (few-peak guard)", None),
]
NOISE_FOR = {"full": "noise-13s", "consecutive-2": "noise-2s", "gapped-2": "noise-2s", "single": "noise-1s"}


def fig_decline_attribution(stars, cands, definition="loose", out=None):
    """Stacked bars per coverage x period bin: declined stars with a rescue
    candidate, split by the set of gates that candidate failed."""
    b = rescue_candidates(stars, cands, definition)
    b = b[b.dataset.isin(COV_ORDER)]
    f = gate_flags(b)
    def label(row):
        fl = [gl.GATE_SHORT[g] for g in gl.GATES if row[g]]
        if len(fl) == 1:
            return fl[0] + " only"
        if "half_lag_consistency" in [g for g in gl.GATES if row[g]]:
            return "half_lag + others"
        return "2+ (no half_lag)"
    b = b.assign(cat=f.apply(label, axis=1))
    cats = [c for c in ["half_lag only", "height_snr only", "vertex only", "pos_heights only", "n_peaks only",
                        "few_height only", "2+ (no half_lag)", "half_lag + others"] if c in set(b.cat)]
    colors = dict(zip(cats, plt.cm.tab10.colors))
    bins = [x for x in gl.BIN_ORDER if x in set(b.period_bin)]
    fig, axes = plt.subplots(1, len(COV_ORDER), figsize=(15, 3.8), sharey=True)
    for ax, ds in zip(axes, COV_ORDER):
        sub = b[b.dataset == ds]
        bottom = np.zeros(len(bins))
        for c in cats:
            v = np.array([((sub.period_bin == pb) & (sub.cat == c)).sum() for pb in bins])
            ax.bar(bins, v, bottom=bottom, color=colors[c], label=c)
            bottom += v
        nd = stars[(stars.dataset == ds) & stars.final_outcome.eq("none")].period_bin.value_counts()
        ax.plot(bins, [nd.get(pb, 0) for pb in bins], "k_", ms=18, mew=2, label="all declines")
        ax.set_title(f"{ds}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel(f"declined stars with a {definition} success candidate")
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Which gates reject the best {definition}-success candidate of declined stars (synthetic n=200 seed 0)",
                 fontsize=10)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=110)
    return fig


def fig_stat_distributions(stars, cands, datasets=COV_ORDER, noise_map=NOISE_FOR, out=None, title=""):
    """Histogram grid (statistic x coverage): per star, the best success
    candidate vs the best wrong/alias candidate, plus pb_scan noise light
    curves' best candidate, with the current threshold marked."""
    succ = gl.best_per_star(cands, cands.cls.eq("success"))
    bad = gl.best_per_star(cands, cands.cls.isin(["wrong", "alias"]))
    noise = gl.best_per_star(cands, cands.cls.eq("noise"))
    fig, axes = plt.subplots(len(STATS), len(datasets), figsize=(3.4 * len(datasets), 2.1 * len(STATS)))
    for j, ds in enumerate(datasets):
        for i, (col, lab, thcol) in enumerate(STATS):
            ax = axes[i, j]
            vals = [succ[succ.dataset == ds][col], bad[bad.dataset == ds][col]]
            names = ["success", "wrong/alias"]
            if noise_map and ds in noise_map:
                vals.append(noise[noise.dataset == noise_map[ds]][col])
                names.append(f"noise ({noise_map[ds]})")
            allv = pd.concat(vals).replace([np.inf, -np.inf], np.nan).dropna()
            if allv.empty:
                continue
            lo, hi = allv.quantile(0.01), allv.quantile(0.99)
            edges = np.linspace(lo, hi, 30)
            for v, nm, colr in zip(vals, names, [CLS_COLORS["success"], CLS_COLORS["wrong"], CLS_COLORS["noise"]]):
                v = v.replace([np.inf, -np.inf], np.nan).dropna().clip(lo, hi)
                ax.hist(v, bins=edges, histtype="step", density=True, lw=1.4, color=colr, label=f"{nm} ({len(v)})")
            th = {"mean_height_snr_value": 1.0, "half_lag_consistency_value": 0.0, "half_lag_contrast": 0.1,
                  "frac_vertex_in_window_value": 0.8, "frac_positive_heights_value": 0.8,
                  "few_peaks_height_value": 0.4}[col]
            ax.axvline(th, color="k", ls="--", lw=1)
            if i == 0:
                ax.set_title(ds, fontsize=10)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and j == len(datasets) - 1:
                ax.legend(fontsize=6)
            elif i == 0:
                ax.legend(fontsize=6)
    fig.suptitle(title or "Gate statistics: best success vs best wrong/alias candidate per star, and noise "
                 "(dashed = current threshold; contrast 0.1 = strong-contrast route; 0.4 applies to few-peak only)",
                 fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=110)
    return fig


def fig_scaling(stars, cands, out=None):
    """Coverage scaling. Left/middle: the same star's matched success candidate,
    short-coverage vs full-year value of height SNR and few-peak ACF height.
    Right: ACF height at P on pb_scan noise vs P, by n_sectors (what the
    few-peak guard has to reject)."""
    c = cands[cands.outcome.eq("correct") & cands.dataset.isin(COV_ORDER)]
    b = gl.best_per_star(c, c.P_fit.notna())
    f = b[b.dataset == "full"].set_index("star_id")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    mk = {"consecutive-2": "o", "gapped-2": "s", "single": "^"}
    for ds in ["consecutive-2", "gapped-2", "single"]:
        s = b[b.dataset == ds].set_index("star_id")
        j = s.join(f, rsuffix="_f", how="inner")
        j = j[(j.P_fit / j.P_fit_f - 1).abs() < 0.05]
        for ax, col in zip(axes[:2], ["mean_height_snr_value", "few_peaks_height_value"]):
            ax.scatter(j[f"{col}_f"], j[col], s=14, marker=mk[ds], alpha=0.7, label=f"{ds} ({len(j)})")
    for ax, col, th, lab in zip(axes[:2], ["mean_height_snr_value", "few_peaks_height_value"], [1.0, 0.4],
                                ["mean height SNR", "ACF height at P"]):
        lim = (0, 4) if col == "mean_height_snr_value" else (-0.2, 1.05)
        ax.plot(lim, lim, color="0.6", lw=0.8)
        ax.axhline(th, color="k", ls="--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{lab}, full year")
        ax.set_ylabel(f"{lab}, short coverage")
        ax.legend(fontsize=7)
    ax = axes[2]
    nz = cands[cands.cls.eq("noise")]
    for ds, colr in zip(["noise-1s", "noise-2s", "noise-13s"], ["C3", "C1", "C0"]):
        s = nz[nz.dataset == ds]
        ax.scatter(s.P_fit, s.few_peaks_height_value, s=6, alpha=0.4, color=colr, label=ds)
        fp = s[s.few_peaks.astype(bool)]
        ax.scatter(fp.P_fit, fp.few_peaks_height_value, s=30, facecolor="none", edgecolor=colr)
    sc = cands[cands.cls.eq("success") & cands.few_peaks.astype(bool) & cands.dataset.isin(COV_ORDER)]
    ax.scatter(sc.P_fit, sc.few_peaks_height_value, s=14, marker="x", color="k", alpha=0.6,
               label="success cands with few_peaks (synthetic)")
    ax.axhline(0.4, color="k", ls="--", lw=1)
    ax.axhline(0.25, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("candidate P_fit [d]")
    ax.set_ylabel("ACF height at P")
    ax.set_title("noise candidates (circled = few_peaks)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=110)
    return fig

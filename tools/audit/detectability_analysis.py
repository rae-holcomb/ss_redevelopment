"""T26 analysis: join per-star detectability to dev_medium outcomes, compute metrics and figures.

Inputs:
- ``reports/2026-11-10_t26-detectability/data/detectability_{fullyear,assigned}.csv``
  (from ``tools/audit/detectability.py``);
- the T10 scorecard runs ``runs/t10_dev_medium_{fullyear,assigned}/{stars,candidates}.csv.gz``
  (main checkout; gitignored).

Steps (``prepare`` needs the runs, and the drive only for the example light curves):
1. Per star: final outcome (success = correct + diffrot; wrong = wrong + half + double;
   declined = none), the "correct period anywhere in the candidates" ceiling, and a
   chance level for that ceiling (truth shuffled among stars of the same period bin,
   5 shuffles), all written to ``data/joined_<mode>.csv``.
2. Threshold scan and per-period-bin metrics on the detectable subset
   (``data/threshold_scan.csv``, ``metrics.json``).
3. Example light curves saved to ``data/examples.npz`` (needs the drive).

``figures`` regenerates every figure from ``data/`` only (the notebook calls it).

Usage::

    $PY tools/audit/detectability_analysis.py prepare [--threshold 5]
    $PY tools/audit/detectability_analysis.py figures [--threshold 5]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "claude_files"))

import argparse  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluation as ev  # noqa: E402

REPORT = REPO / "reports" / "2026-11-10_t26-detectability"
DATA, FIG = REPORT / "data", REPORT / "figures"
RUNS = Path("/Users/raeholcomb/Documents/GitHub/ss_redevelopment/runs")
MODES = ("fullyear", "assigned")
PBINS = ("<1d", "1-3d", "3-10d", "10-30d", ">30d")
STAT = "snr_rot"  # = D_band * sqrt(n_cycles_obs); see detectability.py
STAT_LABEL = r"detectability $D_{\rm band}\sqrt{N_{\rm cyc}}$"
SNR_EDGES = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, np.inf])
COL = dict(success="#2a9d4f", wrong="#d1495b", declined="#8d99ae", ceiling="#3a6ea5", chance="#9fb7d6")
N_SHUFFLE = 5
SEED = 20261110


def outcome_class(final_outcome: pd.Series) -> pd.Series:
    """Map scorecard final_outcome to success / wrong (incl. aliases) / declined."""
    return np.where(final_outcome.isin(ev.SUCCESS_OUTCOMES), "success",
                    np.where(final_outcome.eq("none") | final_outcome.isna(), "declined", "wrong"))


def success_mask(p: np.ndarray, p_true: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 tol: float = ev.REL_TOL) -> np.ndarray:
    """Vectorised 'is a success' (correct or diffrot) test, same rule as evaluation.classify_period."""
    with np.errstate(invalid="ignore"):
        correct = np.abs(p - p_true) / p_true <= tol
        inband = (p >= lo * (1 - tol)) & (p <= hi * (1 + tol))
    return np.nan_to_num(correct | inband, nan=False).astype(bool)


def candidate_ceiling(cands: pd.DataFrame, stars: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Per star: any candidate a success (real truth), and the mean chance rate over shuffled truths."""
    truth = stars.set_index("star_id")[["true_period", "period_lo", "period_hi", "eval_pbin"]]
    c = cands[["star_id", "P_eval", "outcome"]].copy()
    tr = truth.loc[c.star_id]
    real = success_mask(c.P_eval.to_numpy(), tr.true_period.to_numpy(), tr.period_lo.to_numpy(), tr.period_hi.to_numpy())
    # cross-check against the scorecard's own per-candidate outcome
    assert np.array_equal(real, c.outcome.isin(ev.SUCCESS_OUTCOMES).to_numpy()), "vectorised rule != evaluation"
    out = pd.DataFrame({"star_id": c.star_id, "hit": real}).groupby("star_id").hit.any().rename("ceiling").to_frame()
    chance = np.zeros(len(out))
    for _ in range(N_SHUFFLE):
        sh = truth.copy()
        for _, g in truth.groupby("eval_pbin"):
            perm = rng.permutation(len(g))
            sh.loc[g.index, ["true_period", "period_lo", "period_hi"]] = g[["true_period", "period_lo", "period_hi"]].to_numpy()[perm]
        ts = sh.loc[c.star_id]
        hit = success_mask(c.P_eval.to_numpy(), ts.true_period.to_numpy(), ts.period_lo.to_numpy(), ts.period_hi.to_numpy())
        chance += pd.Series(hit, index=c.star_id.to_numpy()).groupby(level=0).any().reindex(out.index).to_numpy()
    out["ceiling_chance"] = chance / N_SHUFFLE
    return out.reset_index()


def prepare_joined(mode: str) -> pd.DataFrame:
    """Join detectability, scorecard outcomes and the candidate ceiling for one coverage mode."""
    d = pd.read_csv(DATA / f"detectability_{mode}.csv")
    s = pd.read_csv(RUNS / f"t10_dev_medium_{mode}" / "stars.csv.gz")
    c = pd.read_csv(RUNS / f"t10_dev_medium_{mode}" / "candidates.csv.gz", usecols=["star_id", "P_eval", "outcome"])
    keep = ["star_id", "eval_pbin", "true_period", "period_lo", "period_hi", "final_period", "final_outcome",
            "noise_classification", "searchable", "coverage_class", "n_sectors", "decline_kind", "flux_amp"]
    x = d.merge(s[keep], on="star_id", how="left")
    x = x.merge(candidate_ceiling(c, s, np.random.default_rng(SEED)), on="star_id", how="left")
    x["ceiling"] = x.ceiling.astype("boolean").fillna(False).astype(bool)
    x["ceiling_chance"] = x.ceiling_chance.fillna(0.0)
    x["outcome3"] = outcome_class(x.final_outcome)
    x[STAT] = x.D_band_sqrt_ncyc
    return x


def scan_thresholds(x: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Trade-off of the detectable cut: subset success, completeness of successes, below-cut behaviour."""
    rows = []
    n_succ = (x.outcome3 == "success").sum()
    for t in grid:
        det = x[STAT] >= t
        a, b = x[det], x[~det]
        rows.append(dict(threshold=t, frac_detectable=det.mean(), n_detectable=int(det.sum()),
                         success_det=(a.outcome3 == "success").mean() if len(a) else np.nan,
                         wrong_det=(a.outcome3 == "wrong").mean() if len(a) else np.nan,
                         declined_det=(a.outcome3 == "declined").mean() if len(a) else np.nan,
                         ceiling_det=a.ceiling.mean() if len(a) else np.nan,
                         success_kept=(a.outcome3 == "success").sum() / n_succ,
                         success_below=(b.outcome3 == "success").mean() if len(b) else np.nan,
                         wrong_below=(b.outcome3 == "wrong").mean() if len(b) else np.nan,
                         ceiling_excess_below=(b.ceiling.mean() - b.ceiling_chance.mean()) if len(b) else np.nan))
    return pd.DataFrame(rows)


def subset_metrics(x: pd.DataFrame) -> dict:
    """success / wrong / declined / ceiling (+ chance) for a set of stars."""
    return dict(n=int(len(x)), success=float((x.outcome3 == "success").mean()),
                wrong=float((x.outcome3 == "wrong").mean()), declined=float((x.outcome3 == "declined").mean()),
                ceiling_any_candidate=float(x.ceiling.mean()), ceiling_chance=float(x.ceiling_chance.mean()))


def metrics_by_bin(x: pd.DataFrame, thr: float) -> dict:
    """Per period bin and overall metrics for all / detectable / undetectable stars."""
    out = {}
    det, quiet = x[STAT] >= thr, x.noise_classification == "quiet"
    subsets = (("all", x), ("detectable", x[det]), ("undetectable", x[~det]),
               ("quiet", x[quiet]), ("detectable_quiet", x[det & quiet]), ("undetectable_quiet", x[~det & quiet]),
               ("detectable_quiet_searchable", x[det & quiet & x.searchable.astype(bool)]))
    for name, sub in subsets:
        out[name] = {"overall": subset_metrics(sub)}
        for b in PBINS:
            out[name][b] = subset_metrics(sub[sub.eval_pbin == b])
    return out


def pick_examples(x: pd.DataFrame, thr: float) -> dict:
    """Three example stars (full-year): detectable-but-declined, undetectable-but-wrong, borderline."""
    rng = np.random.default_rng(SEED)
    nolong = x[x.eval_pbin != ">30d"]
    a = nolong[(nolong.outcome3 == "declined")].sort_values(STAT, ascending=False).head(10)
    wrong = x[(x.outcome3 == "wrong") & (x.final_outcome == "wrong") & (x[STAT] < thr)]
    b = wrong[wrong.noise_classification == "quiet"].sort_values(STAT).head(3)
    b2 = wrong[wrong.noise_classification != "quiet"].sort_values(STAT).head(10)
    c = nolong[(nolong[STAT] > thr / 1.5) & (nolong[STAT] < thr * 1.5)]
    return {"detectable_declined": a.iloc[rng.integers(len(a))].star_id,
            "undetectable_wrong": b.iloc[rng.integers(len(b))].star_id,
            "undetectable_wrong_contaminated": b2.iloc[rng.integers(len(b2))].star_id,
            "borderline": c.iloc[rng.integers(len(c))].star_id}


def save_examples(x: pd.DataFrame, picks: dict, root: str | None) -> None:
    """Save (t, f-1, s-1) on the pipeline's cadences plus candidate periods for the example stars."""
    import detectability as det
    import smarts_data as sd
    rootp = sd.data_root(root)
    det.check_stacks_readable(rootp)
    man = pd.read_csv(REPO / "datasets" / "dev_medium.csv")
    stacks = np.load(det.stack_paths(rootp)["stacks"], mmap_mode="r")
    c = pd.read_csv(RUNS / "t10_dev_medium_fullyear" / "candidates.csv.gz",
                    usecols=["star_id", "period", "P_eval", "method", "rank", "outcome", "passed_gates"])
    arrs, meta = {}, {}
    for key, sid in picks.items():
        r = x[x.star_id == sid].iloc[0]
        m = man[man.star_id == sid].iloc[0]
        g = np.asarray(stacks[int(r.stack_row), int(r.stack_which)], float)
        t, s, n, f = det.star_series(REPO / "data" / "smarts_dev" / m.path, g, None)
        cc = c[c.star_id == sid]
        arrs[f"{key}_t"], arrs[f"{key}_f"], arrs[f"{key}_s"] = t, f, s
        arrs[f"{key}_cand"] = cc.P_eval.to_numpy(float)
        meta[key] = dict(star_id=sid, true_period=float(r.true_period), period_lo=float(r.period_lo),
                         period_hi=float(r.period_hi), final_period=float(r.final_period) if pd.notna(r.final_period) else None,
                         final_outcome=str(r.final_outcome), snr_rot=float(r[STAT]), D_band=float(r.D_band),
                         amp_proxy=float(r.amp_proxy), eval_pbin=r.eval_pbin,
                         noise_classification=str(r.noise_classification),
                         n_cand=int(len(cc)), n_cand_success=int(cc.outcome.isin(ev.SUCCESS_OUTCOMES).sum()))
    np.savez_compressed(DATA / "examples.npz", **arrs)
    (DATA / "examples.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- figures (data/ only)

def load_joined() -> dict:
    """Read the joined per-star tables for both modes."""
    return {m: pd.read_csv(DATA / f"joined_{m}.csv") for m in MODES}


def fig_outcome_vs_snr(J: dict, thr: float, path: Path) -> None:
    """Outcome fractions vs binned detectability, per period bin (columns) and coverage mode (rows)."""
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.6), sharex=True, sharey=True)
    centers = np.sqrt(SNR_EDGES[:-1].clip(0.5) * np.r_[SNR_EDGES[1:-1], 256])
    for i, m in enumerate(MODES):
        x = J[m]
        for j, b in enumerate(PBINS):
            ax = axes[i, j]
            g = x[x.eval_pbin == b]
            k = np.digitize(g[STAT], SNR_EDGES[1:-1])
            for lab in ("success", "wrong", "declined"):
                fr = [np.mean(g.outcome3[k == q] == lab) if (k == q).sum() >= 3 else np.nan for q in range(len(centers))]
                ax.plot(centers, fr, "o-", color=COL[lab], label=lab, ms=4)
            ce = [g.ceiling[k == q].mean() if (k == q).sum() >= 3 else np.nan for q in range(len(centers))]
            ch = [g.ceiling_chance[k == q].mean() if (k == q).sum() >= 3 else np.nan for q in range(len(centers))]
            ax.plot(centers, ce, "s--", color=COL["ceiling"], ms=3, lw=1, label="correct P in candidates")
            ax.plot(centers, ch, ":", color=COL["chance"], lw=1.5, label="… chance level")
            cnt = [(k == q).sum() for q in range(len(centers))]
            for cx, n in zip(centers, cnt):
                if n:
                    ax.text(cx, 1.03, str(n), ha="center", fontsize=6, color="0.4")
            ax.axvline(thr, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.1)
            if i == 0:
                ax.set_title(b)
            if j == 0:
                ax.set_ylabel(f"{m}\nfraction of stars")
            if i == 1:
                ax.set_xlabel(STAT_LABEL)
    axes[0, 0].legend(fontsize=7, loc="center left")
    fig.suptitle(f"dev_medium outcome vs detectability (dashed line: proposed threshold {thr:g}; grey numbers = stars per bin)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_distributions(J: dict, thr: float, path: Path) -> None:
    """Detectability distribution per period bin, both coverage modes overlaid."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True)
    bins = np.logspace(-1.5, 3, 37)
    for j, b in enumerate(PBINS):
        ax = axes[j]
        for m, c in zip(MODES, ("#264653", "#e76f51")):
            v = J[m].loc[J[m].eval_pbin == b, STAT].clip(bins[0], bins[-1])
            frac = (v >= thr).mean()
            ax.hist(v, bins=bins, histtype="step", lw=1.6, color=c, label=f"{m}: {frac:.0%} ≥ thr")
        ax.axvline(thr, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(b)
        ax.set_xlabel(STAT_LABEL)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("stars")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_vs_proxy(J: dict, auc: pd.DataFrame, path: Path) -> None:
    """Detectability vs the T1 catalog amplitude proxy, coloured by outcome, plus AUC comparison."""
    x = J["fullyear"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, v, lab in ((axes[0], STAT, STAT_LABEL), (axes[1], "D_band", r"$D_{\rm band}$ (band-limited S/N amplitude ratio)")):
        for o in ("declined", "wrong", "success"):
            g = x[x.outcome3 == o]
            ax.scatter(g.amp_proxy, g[v], s=7, alpha=0.6, color=COL[o], label=o)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"catalog amplitude proxy $A\,\sin i\,\cos(\bar\lambda)$ (T1)")
        ax.set_ylabel(lab)
        from scipy.stats import spearmanr
        ax.set_title(f"full-year, Spearman ρ = {spearmanr(x.amp_proxy, x[v])[0]:.2f}")
        ax.legend(fontsize=7)
    ax = axes[2]
    w = 0.38
    ii = np.arange(len(auc))
    for k, m in enumerate(MODES):
        ax.barh(ii + (k - 0.5) * w, auc[m], height=w, label=m, color=("#264653", "#e76f51")[k])
    ax.set_yticks(ii, auc.index)
    ax.set_xlim(0.3, 1)
    ax.axvline(0.5, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("AUC for predicting pipeline success")
    ax.legend(fontsize=8)
    ax.set_title("which statistic tracks recovery best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_tradeoff(scans: dict, thr: float, path: Path) -> None:
    """Success on the detectable subset, completeness, and below-threshold behaviour vs threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, m in zip(axes, MODES):
        s = scans[m]
        ax.plot(s.threshold, s.success_det, color=COL["success"], lw=2, label="success rate of detectable stars")
        ax.plot(s.threshold, s.success_kept, color="#264653", lw=2, label="fraction of all successes that are 'detectable'")
        ax.plot(s.threshold, s.frac_detectable, color="0.5", lw=1.5, ls="-.", label="fraction of stars detectable")
        ax.plot(s.threshold, s.success_below, color=COL["success"], lw=1, ls=":", label="success rate below threshold")
        ax.plot(s.threshold, s.wrong_below, color=COL["wrong"], lw=1, ls=":", label="wrong rate below threshold")
        ax.plot(s.threshold, s.ceiling_excess_below, color=COL["ceiling"], lw=1, ls="--",
                label="below threshold: candidate ceiling − chance")
        ax.axvline(thr, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel(f"threshold on {STAT_LABEL}")
        ax.set_title(f"dev_medium {m}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _acf_even(t: np.ndarray, f: np.ndarray, max_lag: float) -> tuple:
    """Gap-aware ACF on the 30-min grid via the pipeline's acf_utils.compute_acf."""
    from acf_utils import compute_acf
    dt = 1800 / 86400.0
    k = np.round((t - t[0]) / dt).astype(int)
    grid = np.full(k.max() + 1, np.nan)
    grid[k] = f
    tt = t[0] + np.arange(len(grid)) * dt
    res = compute_acf(tt, grid, max_lag_frac=min(0.7, max_lag / (tt[-1] - tt[0])))
    return res[0], res[1]


def fig_examples(path_prefix: Path) -> list:
    """One figure per example star: light curve + injected signal, ACF, and periodograms with periods marked."""
    from astropy.timeseries import LombScargle
    E = np.load(DATA / "examples.npz")
    meta = json.loads((DATA / "examples.json").read_text())
    titles = {"detectable_declined": "Detectable star the pipeline declined",
              "undetectable_wrong": "Undetectable star (quiet noise source) with a confident wrong answer",
              "undetectable_wrong_contaminated": "Undetectable star on a contaminated noise source, wrong answer",
              "borderline": "Borderline star (near the proposed threshold)"}
    out = []
    for key, mt in meta.items():
        t, f, s, cand = E[f"{key}_t"], E[f"{key}_f"], E[f"{key}_s"], E[f"{key}_cand"]
        P, Pf = mt["true_period"], mt["final_period"]
        fig, axes = plt.subplots(1, 3, figsize=(17, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
        ax = axes[0]
        ax.plot(t, f * 1e3, ",", color="0.55", label="pipeline flux (signal × noise) − 1")
        nb = max(1, int(round(min(P / 10, 1.0) / (1800 / 86400.0))))
        if nb > 1:
            m = len(t) // nb * nb
            ax.plot(t[:m].reshape(-1, nb).mean(1), f[:m].reshape(-1, nb).mean(1) * 1e3, ".", ms=1.5, color="0.2",
                    label=f"binned ×{nb}")
        ax.plot(t, s * 1e3, ".", ms=0.7, color="#d1495b", label="injected signal s − 1 (= f / g − 1)")
        ax.set_xlabel("time (d)")
        ax.set_ylabel("ppt")
        ax.legend(fontsize=7, markerscale=6, loc="lower left")
        ax.set_title(f"{mt['star_id']}  P={P:.3g} d  [{mt['eval_pbin']}]  D√N={mt['snr_rot']:.2g}  "
                     f"noise: {mt['noise_classification']}", fontsize=9)
        ax = axes[1]
        max_lag = min(max(4 * P, 3 * (Pf or 0), 2.0), 0.7 * (t[-1] - t[0]))
        lags, acf = _acf_even(t, f, max_lag)
        lags_s, acf_s = _acf_even(t, s, max_lag)
        ax.plot(lags, acf, color="0.2", lw=1, label="ACF of pipeline flux")
        ax.plot(lags_s, acf_s, color="#d1495b", lw=1, alpha=0.7, label="ACF of injected signal")
        ax.set_xlim(0, max_lag)
        ax.set_xlabel("lag (d)")
        ax.legend(fontsize=7)
        ax = axes[2]
        fr = np.geomspace(1 / (0.7 * (t[-1] - t[0])), 23.0, 6000)
        ax.plot(1 / fr, LombScargle(t, f).power(fr), color="0.2", lw=0.8, label="LS of pipeline flux")
        ax.plot(1 / fr, LombScargle(t, s).power(fr), color="#d1495b", lw=0.8, alpha=0.7, label="LS of injected signal")
        ax.set_xscale("log")
        ax.set_xlabel("period (d)")
        ax.set_ylabel("LS power")
        ax.legend(fontsize=7)
        for a in axes[1:]:
            for cp in cand[np.isfinite(cand)]:
                if a is axes[2] or cp <= max_lag:
                    a.axvline(cp, color="#3a6ea5", lw=0.5, alpha=0.35)
            a.axvspan(mt["period_lo"] * 0.9, mt["period_hi"] * 1.1, color=COL["success"], alpha=0.15)
            a.axvline(P, color=COL["success"], lw=1.8, label="true P")
            if Pf:
                a.axvline(Pf, color=COL["wrong"], lw=1.8, ls="--", label="pipeline answer")
        axes[2].legend(fontsize=7, loc="upper left")
        fig.suptitle(f"{titles[key]}: outcome = {mt['final_outcome']}, final P = "
                     f"{Pf if Pf is None else round(Pf, 3)} d; {mt['n_cand_success']}/{mt['n_cand']} candidates within "
                     f"the success window (thin blue = all candidates; green band = success window)", fontsize=9)
        fig.tight_layout()
        p = Path(f"{path_prefix}_{key}.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        out.append(p)
    return out


def auc_table(J: dict) -> pd.DataFrame:
    """AUC of each candidate statistic for predicting pipeline success, per coverage mode."""
    from sklearn.metrics import roc_auc_score
    names = {"snr_rot": "D_band·√N_cyc (chosen)", "D_peak": "D_peak (LS peak S/N)", "D_band": "D_band",
             "sig_rms": "injected RMS only", "amp_proxy": "catalog amp proxy (T1)", "noise_rms": "noise RMS only"}
    rows = {}
    for m in MODES:
        x = J[m]
        y = x.outcome3 == "success"
        rows[m] = {lab: roc_auc_score(y[x[v].notna()], x[v].dropna()) for v, lab in names.items()}
    return pd.DataFrame(rows)


def make_figures(thr: float) -> None:
    """Regenerate every report figure from data/."""
    FIG.mkdir(exist_ok=True)
    J = load_joined()
    scans = {m: pd.read_csv(DATA / f"threshold_scan_{m}.csv") for m in MODES}
    fig_outcome_vs_snr(J, thr, FIG / "outcome_vs_detectability.png")
    fig_distributions(J, thr, FIG / "detectability_distributions.png")
    fig_vs_proxy(J, auc_table(J), FIG / "detectability_vs_amp_proxy.png")
    fig_tradeoff(scans, thr, FIG / "threshold_tradeoff.png")
    if (DATA / "examples.npz").exists():
        fig_examples(FIG / "example")


def main():
    """CLI: prepare (needs runs + drive for examples) or figures (data/ only)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "figures"])
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--no-examples", action="store_true")
    a = ap.parse_args()
    if a.step == "prepare":
        J = {m: prepare_joined(m) for m in MODES}
        grid = np.geomspace(0.5, 200, 60)
        metrics = {"threshold": a.threshold, "statistic": STAT, "rel_tol": ev.REL_TOL}
        for m, x in J.items():
            x.to_csv(DATA / f"joined_{m}.csv", index=False)
            scan_thresholds(x, grid).to_csv(DATA / f"threshold_scan_{m}.csv", index=False)
            metrics[m] = metrics_by_bin(x, a.threshold)
        metrics["auc_success"] = auc_table(J).round(4).to_dict()
        (REPORT / "metrics.json").write_text(json.dumps(metrics, indent=1))
        if not a.no_examples:
            picks = pick_examples(J["fullyear"], a.threshold)
            save_examples(J["fullyear"], picks, a.data_root)
    make_figures(a.threshold)


if __name__ == "__main__":
    main()

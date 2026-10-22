"""T16c analysis: bit-identity check, R/2 validation per coverage class,
min_valid_frac 0.25 vs 0.3, per-method recall by coverage.

Reads scorecard outputs in runs/t16c/{before,after,mvf025}/<run>/ and writes
tables to data/ and figures to figures/ (next to this file).

R per star is the scorecard's `acf_lag_max` (the Stage 2 ACF's last lag):
compute_acf truncates at the first lag where the valid-pair fraction drops
below 0.3, so this equals coverage.searchable_range's R except where the
0.7 B max_lag_frac cap binds first (near-contiguous data, where the two
differ by < 1 cadence; checked in check_reach_proxy()).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs" / "t16c"
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "claude_files"))
import evaluation as ev  # noqa: E402

SYN = ["syn_single", "syn_consecutive-2", "syn_gapped-2", "syn_3-5", "syn_6-12"]
RUN_NAMES = ["dm_full", "dm_assigned"] + SYN
CLASS_ORDER = ["single", "consecutive-2", "gapped-2", "consecutive-3-5", "gapped-3-5",
               "consecutive-6-12", "gapped-6-12", "full"]
PR_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, np.inf]
PR_LABELS = ["0-.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5", ".5-.6", ".6-.8", ".8-1", "1-1.5", ">1.5"]


def load(variant: str, run: str):
    """(per_star, per_candidate, metrics) for one run, or None if missing."""
    d = RUNS / variant / run
    if not (d / "metrics.json").exists():
        return None
    ps = pd.read_csv(d / "per_star.csv")
    pc = pd.read_csv(d / "per_candidate.csv") if (d / "per_candidate.csv").exists() else None
    return ps, pc, json.loads((d / "metrics.json").read_text())


def add_reach_cols(ps: pd.DataFrame) -> pd.DataFrame:
    """Add R (= acf_lag_max), P_lo (low end of the success band), true P_lo/R and reported P/R."""
    ps = ps.copy()
    ps["R"] = ps["acf_lag_max"]
    ps["P_lo"] = ps["period_lo"].fillna(ps["true_period"])
    ps["truth_over_R"] = ps["P_lo"] / ps["R"]
    ps["reported_over_R"] = ps["final_period"] / ps["R"]
    ps["truth_bin"] = pd.cut(ps["truth_over_R"], PR_BINS, labels=PR_LABELS, right=False)
    ps["reported_bin"] = pd.cut(ps["reported_over_R"], PR_BINS, labels=PR_LABELS, right=False)
    ps["success"] = ps["final_outcome"].isin(ev.SUCCESS_OUTCOMES)
    ps["wrong"] = ps["final_outcome"] == ev.WRONG
    ps["alias"] = ps["final_outcome"].isin([ev.HALF, ev.DOUBLE])
    ps["answered"] = ps["final_outcome"] != ev.NONE
    return ps


# --------------------------------------------------------------------------
# 1. bit-identity
# --------------------------------------------------------------------------

def bit_identity(run: str) -> dict:
    """Compare before/after on one run.

    A star may change only if the before pool had a beyond-reach candidate
    (P0 > R/2 among the fitted/failed candidates). For every star, the
    after candidate rows must equal the before rows restricted to P0 <= R/2
    (same P0, P_fit, redchi, gates).
    """
    b, a = load("before", run), load("after", run)
    if b is None or a is None:
        return None
    bs, bc, _ = b
    as_, ac, _ = a
    m = bs.merge(as_, on="star_id", suffixes=("_b", "_a"))
    half_R = m.set_index("star_id")["acf_lag_max_b"] / 2
    bc = bc.copy()
    bc["beyond"] = bc["P0"] > bc["star_id"].map(half_R)
    n_beyond = bc.groupby("star_id")["beyond"].sum()
    m["n_beyond_before"] = m["star_id"].map(n_beyond).fillna(0).astype(int)
    same_P = (m["final_period_b"] == m["final_period_a"]) | (m["final_period_b"].isna() & m["final_period_a"].isna())
    m["changed"] = ~same_P | (m["final_outcome_b"] != m["final_outcome_a"])
    # candidate-level equality of in-reach candidates
    cols = ["P0", "P_fit", "redchi", "passed_gates", "failed_gates"]
    keyb = bc[~bc["beyond"]][["star_id"] + cols].sort_values(["star_id", "P0"]).reset_index(drop=True)
    keya = ac[["star_id"] + cols].sort_values(["star_id", "P0"]).reset_index(drop=True)
    cand_identical = keyb.equals(keya)
    changed = m[m["changed"]]
    out = dict(run=run, n_stars=len(m), n_with_beyond=int((m["n_beyond_before"] > 0).sum()),
               n_beyond_cands=int(bc["beyond"].sum()), n_cands_before=len(bc), n_cands_after=len(ac),
               n_changed=int(len(changed)),
               n_changed_without_beyond=int((changed["n_beyond_before"] == 0).sum()),
               inreach_candidates_identical=bool(cand_identical),
               fit_s_median_before=float(m["fit_seconds_b"].median()),
               fit_s_median_after=float(m["fit_seconds_a"].median()),
               fit_s_total_before=float(m["fit_seconds_b"].sum()),
               fit_s_total_after=float(m["fit_seconds_a"].sum()),
               star_s_median_before=float(m["seconds_b"].median()),
               star_s_median_after=float(m["seconds_a"].median()))
    changed = changed[["star_id", "coverage_class_b", "true_period_b", "acf_lag_max_b", "n_beyond_before",
                       "final_period_b", "final_outcome_b", "final_period_a", "final_outcome_a"]].copy()
    changed.insert(0, "run", run)
    return out, changed, m


def fit_time_by_class(m: pd.DataFrame, run: str) -> pd.DataFrame:
    """Median and total fit seconds before/after per coverage class."""
    g = m.groupby("coverage_class_b")
    t = pd.DataFrame({"n": g.size(), "fit_med_before": g["fit_seconds_b"].median(),
                      "fit_med_after": g["fit_seconds_a"].median(),
                      "fit_sum_before": g["fit_seconds_b"].sum(), "fit_sum_after": g["fit_seconds_a"].sum(),
                      "frac_stars_with_beyond": g["n_beyond_before"].apply(lambda s: (s > 0).mean())})
    t["fit_sum_ratio"] = t["fit_sum_after"] / t["fit_sum_before"]
    t.insert(0, "run", run)
    return t.reset_index().rename(columns={"coverage_class_b": "coverage_class"})


# --------------------------------------------------------------------------
# 2. R/2 validation
# --------------------------------------------------------------------------

def stack(variant: str, runs) -> pd.DataFrame:
    """Concatenate per_star tables (with reach columns) across runs."""
    parts = []
    for r in runs:
        x = load(variant, r)
        if x is not None:
            ps = add_reach_cols(x[0])
            ps["run"] = r
            parts.append(ps)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def rate_table(df: pd.DataFrame, by: str, bincol: str) -> pd.DataFrame:
    """success / wrong / alias / answered fraction per (by, bin)."""
    g = df.groupby([by, bincol], observed=True)
    return pd.DataFrame({"n": g.size(), "success": g["success"].mean(), "wrong": g["wrong"].mean(),
                         "alias": g["alias"].mean(), "answered": g["answered"].mean()}).reset_index()


def wilson(k, n, z=1.0):
    """Wilson binomial interval (z=1: ~68%) for k successes in n trials; NaN where n = 0."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        den = 1 + z ** 2 / n
        mid = (p + z ** 2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def plot_truth_curves(before: pd.DataFrame, after: pd.DataFrame, title: str, fname: str):
    """Success and wrong fraction vs true P_lo/R per coverage class, with ~68% Wilson bands.

    Draws the before (uncapped) run. If the after run's per-star outcomes are
    identical (the R/2 skip changed no answer), that is stated in the title
    instead of overplotting identical curves; otherwise after is drawn dashed.
    """
    classes = [c for c in CLASS_ORDER if c in set(before["coverage_class"])]
    same = (not after.empty and before.set_index(["run", "star_id"])["final_outcome"].sort_index()
            .equals(after.set_index(["run", "star_id"])["final_outcome"].sort_index()))
    nc = len(classes)
    ncol = (nc + 1) // 2
    fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 6.4), sharey=True, squeeze=False)
    x = np.arange(len(PR_LABELS))
    series = [(before, "-", "")] + ([] if same or after.empty else [(after, "--", " (after)")])
    for ax, c in zip(axes.flat, classes):
        for df, ls, suf in series:
            sub = df[df.coverage_class == c]
            g = sub.groupby("truth_bin", observed=False)
            n = g.size().reindex(PR_LABELS).fillna(0).to_numpy()
            for col, color, lab in (("success", "C2", "success"), ("wrong", "C3", "wrong")):
                k = g[col].sum().reindex(PR_LABELS).fillna(0).to_numpy()
                frac = np.where(n > 0, k / np.maximum(n, 1), np.nan)
                lo, hi = wilson(k, n)
                ax.plot(x, frac, ls, color=color, marker="o", ms=3, label=lab + suf)
                if not suf:
                    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
            if not suf:
                for xi, ni in zip(x, n.astype(int)):
                    ax.text(xi, 1.02, str(ni), ha="center", fontsize=6, color="0.4")
        ax.axvline(4.5, color="k", lw=0.8, ls=":")
        ax.set_title(c, fontsize=9, pad=12)
        ax.set_xticks(x, PR_LABELS, rotation=60, fontsize=7)
        ax.set_ylim(0, 1.08)
    for ax in axes.flat[nc:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="upper right")
    fig.supxlabel("true P_lo / R   (dotted line: R/2; grey numbers: stars per bin; bands: ~68% binomial)")
    note = ("  [after R/2 skip: identical outcomes for every star]" if same else
            "  [solid: before, uncapped; dashed: after, R/2 skip]")
    fig.suptitle(title + note, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def plot_reported_reliability(before: pd.DataFrame, title: str, fname: str):
    """Among answered stars (before, uncapped): fraction success vs wrong/alias by REPORTED P/R, per class."""
    classes = [c for c in CLASS_ORDER if c in set(before["coverage_class"])]
    ans = before[before.answered]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(PR_LABELS))
    for i, c in enumerate(classes):
        t = rate_table(ans[ans.coverage_class == c], "coverage_class", "reported_bin").set_index("reported_bin")
        t = t.reindex(PR_LABELS)
        ax[0].plot(x, t["success"], marker="o", ms=3, color=f"C{i}", label=c)
        ax[1].plot(x, t["n"].fillna(0), marker="o", ms=3, color=f"C{i}")
    for a in ax:
        a.axvline(4.5, color="k", lw=0.8, ls=":")
        a.set_xticks(x, PR_LABELS, rotation=60, fontsize=7)
        a.set_xlabel("reported P / R")
    ax[0].set_ylabel("precision: success / answered")
    ax[1].set_ylabel("answers in bin")
    ax[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. per-method recall
# --------------------------------------------------------------------------

def method_table(variant: str, runs) -> pd.DataFrame:
    """per_method rows joined with the star's coverage class and R; recall = success in the method's list."""
    parts = []
    for r in runs:
        d = RUNS / variant / r
        if not (d / "per_method.csv").exists():
            continue
        pm = pd.read_csv(d / "per_method.csv")
        ps = add_reach_cols(pd.read_csv(d / "per_star.csv"))[
            ["star_id", "coverage_class", "R", "truth_over_R", "truth_bin", "largest_gap_days",
             "longest_contiguous_days"]]
        x = pm.merge(ps, on="star_id")
        x["run"] = r
        parts.append(x)
    pm = pd.concat(parts, ignore_index=True)
    pm["recall"] = pm["outcome"].isin(ev.SUCCESS_OUTCOMES)
    pm["top1_over_true"] = pm["top1"] / pm["true_period"]
    return pm


def plot_method_recall(pm: pd.DataFrame, title: str, fname: str):
    """Stage 1 recall vs true P/R per method, one panel per coverage class."""
    classes = [c for c in CLASS_ORDER if c in set(pm["coverage_class"])]
    methods = sorted(pm["method"].unique())
    nc = len(classes)
    fig, axes = plt.subplots(2, (nc + 1) // 2, figsize=(3.4 * ((nc + 1) // 2), 6.4), sharey=True, squeeze=False)
    x = np.arange(len(PR_LABELS))
    for ax, c in zip(axes.flat, classes):
        sub = pm[pm.coverage_class == c]
        for i, mth in enumerate(methods):
            t = sub[sub.method == mth].groupby("truth_bin", observed=True)["recall"].mean().reindex(PR_LABELS)
            ax.plot(x, t, marker="o", ms=2.5, lw=1, color=plt.cm.tab20(i / max(len(methods) - 1, 1)), label=mth)
        ax.axvline(4.5, color="k", lw=0.8, ls=":")
        ax.set_title(c, fontsize=9)
        ax.set_xticks(x, PR_LABELS, rotation=60, fontsize=7)
        ax.set_ylim(0, 1.02)
    for ax in axes.flat[nc:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=5, loc="upper right")
    fig.supxlabel("true P_lo / R")
    fig.supylabel("Stage 1 recall (success anywhere in the method's list)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


MVF_RUNS = ["dm_assigned", "syn_single", "syn_consecutive-2", "syn_gapped-2", "dm_full"]


def mvf_compare():
    """min_valid_frac 0.25 (mvf025) vs 0.3 (after): per-run outcome changes and P/R curves.

    Both use the T16c code (R/2 skip with R at the frozen 0.3 definition), so the
    only difference is how far the ACF the fit sees extends past R.
    """
    rows, parts = [], []
    for r in MVF_RUNS:
        a, m = load("after", r), load("mvf025", r)
        if a is None or m is None:
            continue
        x = add_reach_cols(a[0]).merge(add_reach_cols(m[0]), on="star_id", suffixes=("_30", "_25"))
        ch = x[x.final_outcome_30 != x.final_outcome_25]
        rows.append(dict(run=r, n=len(x), success_30=x.success_30.mean(), success_25=x.success_25.mean(),
                         wrong_30=x.wrong_30.mean(), wrong_25=x.wrong_25.mean(),
                         alias_30=x.alias_30.mean(), alias_25=x.alias_25.mean(), n_changed=len(ch),
                         gained_success=int((~ch.success_30 & ch.success_25).sum()),
                         lost_success=int((ch.success_30 & ~ch.success_25).sum()),
                         new_wrong=int((~ch.wrong_30 & ch.wrong_25).sum()),
                         fixed_wrong=int((ch.wrong_30 & ~ch.wrong_25).sum())))
        ch = ch[["star_id", "coverage_class_30", "true_period_30", "truth_over_R_30", "final_period_30",
                 "final_outcome_30", "final_period_25", "final_outcome_25"]].copy()
        ch.insert(0, "run", r)
        parts.append(ch)
    if not rows:
        return None
    t = pd.DataFrame(rows)
    t.to_csv(DATA / "mvf025_summary.csv", index=False)
    pd.concat(parts).to_csv(DATA / "mvf025_changed_stars.csv", index=False)
    # success / wrong vs true P/R, pooled over short-coverage runs
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    x = np.arange(len(PR_LABELS))
    for ax, runs, ttl in ((axes[0], ["syn_single", "syn_consecutive-2", "syn_gapped-2"], "synthetic 1-2 sectors"),
                          (axes[1], ["dm_assigned"], "dev_medium assigned")):
        for v, ls in (("after", "-"), ("mvf025", "--")):
            d = stack(v, runs)
            if d.empty:
                continue
            g = d.groupby("truth_bin", observed=False)
            lab = "0.3" if v == "after" else "0.25"
            ax.plot(x, g.success.mean().reindex(PR_LABELS), ls, color="C2", marker="o", ms=3,
                    label=f"success, min_valid_frac {lab}")
            ax.plot(x, g.wrong.mean().reindex(PR_LABELS), ls, color="C3", marker="o", ms=3,
                    label=f"wrong, min_valid_frac {lab}")
        ax.axvline(4.5, color="k", lw=0.8, ls=":")
        ax.set_xticks(x, PR_LABELS, rotation=60, fontsize=7)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("true P_lo / R")
    axes[0].legend(fontsize=7)
    fig.suptitle("compute_acf min_valid_frac 0.25 vs 0.3 (both with the R/2 skip)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "mvf025_vs_030.png", dpi=130)
    plt.close(fig)
    return t


def main():
    """Run every analysis whose inputs exist; write tables and figures."""
    summ, changed_all, tclass = [], [], []
    for r in RUN_NAMES:
        x = bit_identity(r)
        if x is None:
            continue
        s, ch, m = x
        summ.append(s)
        changed_all.append(ch)
        tclass.append(fit_time_by_class(m, r))
    if summ:
        pd.DataFrame(summ).to_csv(DATA / "bit_identity_summary.csv", index=False)
        pd.concat(changed_all).to_csv(DATA / "changed_stars.csv", index=False)
        pd.concat(tclass).to_csv(DATA / "fit_time_by_class.csv", index=False)
        print(pd.DataFrame(summ).to_string())

    for label, runs in (("dm_assigned", ["dm_assigned"]), ("dm_full", ["dm_full"]), ("synthetic", SYN)):
        b, a = stack("before", runs), stack("after", runs)
        if b.empty:
            continue
        rate_table(b, "coverage_class", "truth_bin").to_csv(DATA / f"truth_curve_before_{label}.csv", index=False)
        if not a.empty:
            rate_table(a, "coverage_class", "truth_bin").to_csv(DATA / f"truth_curve_after_{label}.csv", index=False)
        plot_truth_curves(b, a, f"{label}: success and wrong rate vs true P/R, by coverage class", f"truth_curve_{label}.png")
        plot_reported_reliability(b, f"{label} (before, uncapped fit): precision vs reported P/R",
                                  f"reported_precision_{label}.png")
        pm = method_table("before", runs)
        pm.to_csv(DATA / f"per_method_{label}.csv.gz", index=False)
        plot_method_recall(pm, f"{label}: Stage 1 recall vs true P/R by method and coverage class",
                           f"method_recall_{label}.png")


if __name__ == "__main__":
    main()
    t = mvf_compare()
    if t is not None:
        print(t.round(3).to_string())

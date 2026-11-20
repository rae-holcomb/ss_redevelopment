"""P44 figures and summary tables, from saved data only (data/ and the committed _v8 per-star tables).

fig_sweep.png        purity-completeness curve on dev_medium_b_clean (success vs wrong over the cutoff sweep,
                     old and new mean_height_snr definitions, both modes), chosen point marked
fig_snr_hist.png     new-definition mean_height_snr of correct / alias / wrong / noise-only candidates on B
fig_bins.png         per-period-bin success / alias / wrong, _v8 vs local_tol_trim_l0x, dev_medium_clean and B,
                     both modes, Wilson 68% intervals
fig_gates.png        what blocks the closest correct candidate (best_success_failed_gates), before vs after
data/summary.csv     headline numbers per (set, mode, run); data/lost13.csv the P43 lag-0 losses
"""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA, FIG = HERE / "data", HERE / "figures"
S, A = {"correct", "diffrot"}, {"half", "double"}
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
VNEW = "local_tol_trim_l0x"
SETS = {"dm": ("dev_medium_clean", "dm"), "b": ("dev_medium_b_clean", "b")}
COL = {"old": "#555555", "new": "#d62728"}


def wilson(k, n, z=1.0):
    """Wilson interval (z=1: ~68%) for k of n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def per_star(set_name, mode, run):
    """Per-star table: run 'old' = committed _v8 baseline tables, 'new' = this task's scorecard."""
    if run == "old":
        return pd.read_csv(ROOT / f"benchmarks/baselines/{set_name}_{mode}_v8_tables/per_star.csv")
    tag = {"dev_medium_clean": "dm", "dev_medium_b_clean": "b"}[set_name] + {"fullyear": "fy", "assigned": "as"}[mode]
    return pd.read_csv(DATA / f"per_star/{VNEW}_{tag}.csv.gz")


def fig_sweep(chosen):
    sw = pd.read_csv(DATA / "sweep_B.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mode in zip(axes, ["fullyear", "assigned"]):
        for v, lab, col in [("local_tol_trim", "old: lag-0 tooth in", COL["old"]),
                            (VNEW, "new: lag-0 tooth out", COL["new"])]:
            t = sw[(sw["mode"] == mode) & (sw.variant == v)].sort_values("cutoff")
            ax.plot(t.wrong, t.success, "-o", ms=2.5, color=col, label=lab)
            for cut in (0.5, 1.0, 1.5, 2.0):
                r = t[np.isclose(t.cutoff, cut)]
                if len(r):
                    ax.annotate(f"{cut:g}", (r.wrong.iloc[0], r.success.iloc[0]), fontsize=7, color=col,
                                xytext=(3, -9), textcoords="offset points")
        o = sw[(sw["mode"] == mode) & (sw.variant == "local_tol_trim") & np.isclose(sw.cutoff, 1.0)].iloc[0]
        ax.plot(o.wrong, o.success, "s", ms=9, mfc="none", mec="k", label=f"_v8 (old, 1.0): {o.success:.3f} / {o.wrong:.3f}")
        c = chosen[mode]
        ax.plot(c["wrong"], c["success"], "*", ms=15, color=COL["new"], mec="k",
                label=f"rule's pick ({mode}): {c['cutoff']:g} -> {c['success']:.3f} / {c['wrong']:.3f}")
        if mode == "assigned":
            f = sw[(sw["mode"] == mode) & (sw.variant == VNEW) & np.isclose(sw.cutoff, chosen["chosen"])].iloc[0]
            ax.plot(f.wrong, f.success, "D", ms=8, color="gold", mec="k",
                    label=f"chosen (full-year) {chosen['chosen']:g}: {f.success:.3f} / {f.wrong:.3f}")
        ax.axvline(c["target_wrong"], color="k", ls=":", lw=0.8)
        ax.set_xlabel("wrong fraction (all stars)")
        ax.set_ylabel("success fraction (all stars)")
        ax.set_title(f"dev_medium_b_clean {mode}: mean_height_snr cutoff sweep", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sweep.png", dpi=130)
    plt.close(fig)


def cand_classes(c):
    """correct / alias / wrong (star has a correct or alias candidate) / noise-only (it has none)."""
    good = c.groupby("star_id").outcome.apply(lambda o: o.isin(S | A).any())
    cls = np.where(c.outcome.isin(S), "correct", np.where(c.outcome.isin(A), "alias",
                   np.where(c.star_id.map(good), "wrong (star has a real candidate)", "noise-only (star has none)")))
    return pd.Series(cls, index=c.index)


def fig_snr_hist(chosen):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    order = [("correct", "#2ca02c"), ("alias", "#9467bd"), ("wrong (star has a real candidate)", "#ff7f0e"),
             ("noise-only (star has none)", "#7f7f7f")]
    bins = np.linspace(-1, 6, 71)
    for j, mode in enumerate(["fullyear", "assigned"]):
        c = pd.read_csv(DATA / f"cands_B_{mode}.csv.gz")
        c = c[c.variant == VNEW].copy()
        c["cls"] = cand_classes(c)
        for i, (sub, lab) in enumerate([(c, "all fitted candidates"),
                                        (c[c.failed_ex_snr.fillna("") == ""], "candidates passing every other gate")]):
            ax = axes[i, j]
            for k, col in order:
                x = sub.loc[sub.cls == k, "snr_gate"].clip(bins[0], bins[-1]).dropna()
                ax.hist(x, bins=bins, histtype="step", lw=1.5, color=col, density=True, label=f"{k} (n={len(x)})")
            ax.axvline(chosen["chosen"], color="k", ls="--", lw=1, label=f"chosen cutoff {chosen['chosen']:g}")
            ax.axvline(1.0, color="k", ls=":", lw=0.8, label="old cutoff 1.0")
            ax.set_title(f"B {mode}: {lab}", fontsize=9)
            ax.set_xlabel("mean_height_snr, lag-0 tooth out")
            ax.set_ylabel("density")
            ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_snr_hist.png", dpi=130)
    plt.close(fig)


def fig_bins():
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
    rows = []
    for j, (set_name, mode) in enumerate([(s, m) for s in ("dev_medium_clean", "dev_medium_b_clean")
                                          for m in ("fullyear", "assigned")]):
        for run, off in (("old", -0.12), ("new", 0.12)):
            d = per_star(set_name, mode, run)
            for i, (lab, cats) in enumerate([("success", S), ("alias", A), ("wrong", {"wrong"})]):
                ks, ns = [], []
                for b in BINS:
                    sub = d[d.period_bin == b]
                    ks.append(sub.final_outcome.isin(cats).sum())
                    ns.append(len(sub))
                p = np.array(ks) / np.maximum(ns, 1)
                lo, hi = np.array([wilson(k, n) for k, n in zip(ks, ns)]).T
                x = np.arange(len(BINS)) + off
                axes[i, j].errorbar(x, p, yerr=[p - lo, hi - p], fmt="o", color=COL[run], ms=4, capsize=2,
                                    label="_v8 (local_tol_trim)" if run == "old" else VNEW)
                axes[i, j].set_ylabel(lab)
                for b, k, n in zip(BINS, ks, ns):
                    rows.append(dict(set=set_name, mode=mode, run=run, metric=lab, bin=b, k=k, n=n))
            axes[0, j].set_title(f"{set_name} {mode}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels(BINS)
        ax.set_xlabel("true period bin")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_bins.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(DATA / "bins.csv", index=False)


def fig_gates():
    gates = ["n_peaks", "frac_positive_heights", "mean_height_snr", "frac_vertex_tol", "half_lag_consistency",
             "few_peaks_height"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for j, (set_name, mode) in enumerate([(s, m) for s in ("dev_medium_clean", "dev_medium_b_clean")
                                          for m in ("fullyear", "assigned")]):
        ax = axes[j]
        for run, off in (("old", -0.2), ("new", 0.2)):
            d = per_star(set_name, mode, run)
            blocked = d[~d.final_outcome.isin(S) & d.best_success_failed_gates.notna()]
            cnt = [blocked.best_success_failed_gates.str.split(";").apply(lambda g: gt in g).sum() for gt in gates]
            ax.barh(np.arange(len(gates)) + off, cnt, height=0.4, color=COL[run],
                    label=("_v8" if run == "old" else VNEW) + f" ({len(blocked)} stars)")
        ax.set_yticks(range(len(gates)))
        ax.set_yticklabels(gates, fontsize=8)
        ax.set_title(f"{set_name} {mode}", fontsize=9)
        ax.set_xlabel("stars whose closest correct candidate fails the gate")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_gates.png", dpi=130)
    plt.close(fig)


def summary():
    rows = []
    for set_name in ("dev_medium_clean", "dev_medium_b_clean"):
        for mode in ("fullyear", "assigned"):
            for run in ("old", "new"):
                d = per_star(set_name, mode, run)
                o = d.final_outcome
                ans = (o != "none").sum()
                rows.append(dict(set=set_name, mode=mode, run=run, n=len(d), success=o.isin(S).mean(),
                                 wrong=(o == "wrong").mean(), alias=o.isin(A).mean(), none=(o == "none").mean(),
                                 precision=o.isin(S).sum() / max(ans, 1)))
            a = per_star(set_name, mode, "old").set_index("star_id").final_outcome
            b = per_star(set_name, mode, "new").set_index("star_id").final_outcome.reindex(a.index)
            rows[-1].update(gained=int((b.isin(S) & ~a.isin(S)).sum()), lost=int((a.isin(S) & ~b.isin(S)).sum()),
                            new_wrong=int(((b == "wrong") & (a != "wrong")).sum()),
                            fixed_wrong=int(((a == "wrong") & (b != "wrong")).sum()))
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "summary.csv", index=False)
    # the 13 P43 star-modes lost to the lag-0 tooth (best candidate failed mean_height_snr/n_peaks after
    # dropping more windows than lsq), and whether they come back
    l13 = []
    for mode in ("fullyear", "assigned"):
        lost = pd.read_csv(ROOT / f"reports/2026-11-16_p43-clean-rerun/data/lost_{mode}.csv")
        rej = lost.best_failed.fillna("").str.contains("mean_height_snr|n_peaks") & (lost.best_n_dropped > lost.lsq_n_dropped)
        new = per_star("dev_medium_clean", mode, "new").set_index("star_id").final_outcome
        old = per_star("dev_medium_clean", mode, "old").set_index("star_id").final_outcome
        for _, r in lost[rej].iterrows():
            l13.append(dict(mode=mode, star_id=r.star_id, true_period=r.true_period, v7=r.out_local_screen2,
                            v8=old.get(r.star_id), l0x=new.get(r.star_id)))
    l13 = pd.DataFrame(l13)
    l13.to_csv(DATA / "lost13.csv", index=False)
    return out, l13


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    chosen = json.load(open(DATA / "chosen_cutoff.json"))
    fig_sweep(chosen)
    fig_snr_hist(chosen)
    have_sc = all((DATA / f"per_star/{VNEW}_{t}.csv.gz").exists() for t in ("dmfy", "dmas", "bfy", "bas"))
    if have_sc:
        fig_bins()
        fig_gates()
        out, l13 = summary()
        pd.set_option("display.width", 200)
        print(out.round(3).to_string(index=False))
        print(l13.to_string(index=False))
        print("lost-13 back:", int(l13.l0x.isin(S).sum()), "of", len(l13), "| already back under _v8:",
              int(l13.v8.isin(S).sum()))

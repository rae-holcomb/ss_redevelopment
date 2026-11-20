"""P59 figures and tables, from saved data only (data/ and the committed _v9 per-star tables).

fig_frontier.png  purity-completeness on dev_medium_b_clean: every swept combination (dots) and the upper
                  envelope, old (l0x) vs new (l0x2) gate definitions, both modes; _v9 and the pick marked
fig_map.png       2-D maps on B full-year (new definition): success and wrong over (min_peaks_required, SNR
                  cutoff) at the chosen frac_positive_heights, and over (frac_positive_heights, SNR) at the
                  chosen min_peaks_required; cells within the wrong budget outlined
fig_bins.png      per-period-bin success / alias / wrong, _v9 vs l0x2, dev_medium_clean and B, both modes
fig_gates.png     what blocks the closest correct candidate, before vs after
data/summary.csv  headline numbers
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA, FIG = HERE / "data", HERE / "figures"
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("p44_figures", ROOT / "reports/2026-11-20_p44-lag0-out/figures.py")
F44 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F44)  # P44's helpers (wilson, S, A, BINS); a module named "figures" would be this file

S, A = F44.S, F44.A
BINS = F44.BINS
NEW, OLD = "local_tol_trim_l0x2", "local_tol_trim_l0x"
COL = {OLD: "#555555", NEW: "#d62728"}
LAB = {OLD: "old gates (l0x: lag-0 tooth in n_peaks / frac_pos)", NEW: "new gates (l0x2: lag-0 tooth out)"}


def per_star(set_name, mode, run):
    """run 'old' = committed _v9 tables (default local_tol_trim_l0x), 'new' = this task's l0x2 scorecard."""
    if run == "old":
        return pd.read_csv(ROOT / f"benchmarks/baselines/{set_name}_{mode}_v9_tables/per_star.csv")
    tag = {"dev_medium_clean": "dm", "dev_medium_b_clean": "b"}[set_name] + {"fullyear": "fy", "assigned": "as"}[mode]
    v = NEW if run == "new" else NEW + "b"  # 'alt' = the declared alternative local_tol_trim_l0x2b
    return pd.read_csv(DATA / f"per_star/{v}_{tag}.csv.gz")


def envelope(t):
    """Upper envelope of success vs wrong: best success at each wrong level or below."""
    t = t.sort_values(["wrong", "success"], ascending=[True, False])
    best, pts = -1, []
    for w, s in zip(t.wrong, t.success):
        if s > best:
            pts.append((w, s))
            best = s
    return np.array(pts)


def fig_frontier(chosen):
    sw = pd.read_csv(DATA / "sweep_B.csv")
    fig, axes2 = plt.subplots(2, 2, figsize=(11, 9))
    zoom = {"fullyear": ((0.015, 0.06), (0.24, 0.30)), "assigned": ((0.008, 0.035), (0.14, 0.19))}
    for row, axes in enumerate(axes2):
        for ax, mode in zip(axes, ["fullyear", "assigned"]):
            for v in (OLD, NEW):
                t = sw[(sw["mode"] == mode) & (sw.variant == v)]
                ax.plot(t.wrong, t.success, ".", ms=3, alpha=0.35, color=COL[v])
                e = envelope(t)
                ax.step(e[:, 0], e[:, 1], where="post", color=COL[v], lw=1.6, label=LAB[v] + f" ({len(t)} combos)")
            b = sw[(sw["mode"] == mode) & (sw.variant == OLD) & (sw.mp == 4) & np.isclose(sw.fp, 0.8)
                   & np.isclose(sw.snr, 1.02)].iloc[0]
            ax.plot(b.wrong, b.success, "s", ms=9, mfc="none", mec="k",
                    label=f"v9 default (4, 0.8, 1.02): {b.success:.3f} / {b.wrong:.3f}")
            c = chosen["chosen"]
            p = sw[(sw["mode"] == mode) & (sw.variant == NEW) & (sw.mp == c["mp"]) & np.isclose(sw.fp, c["fp"])
                   & np.isclose(sw.snr, c["snr"])].iloc[0]
            ax.plot(p.wrong, p.success, "*", ms=15, color=COL[NEW], mec="k",
                    label=f"pick ({c['mp']}, {c['fp']:g}, {c['snr']:g}): {p.success:.3f} / {p.wrong:.3f}")
            q = sw[(sw["mode"] == mode) & (sw.variant == NEW) & (sw.mp == 3) & np.isclose(sw.fp, 0.6)
                   & np.isclose(sw.snr, 1.06)].iloc[0]
            ax.plot(q.wrong, q.success, "P", ms=10, color="gold", mec="k",
                    label=f"declared alternative l0x2b (3, 0.6, 1.06): {q.success:.3f} / {q.wrong:.3f}")
            ob = chosen[mode]["old_best"]
            ax.plot(ob["wrong"], ob["success"], "D", ms=7, color="0.6", mec="k",
                    label=f"old gates re-tuned ({ob['mp']}, {ob['fp']:g}, {ob['snr']:g}): "
                          f"{ob['success']:.3f} / {ob['wrong']:.3f}")
            ax.axvline(chosen[mode]["target_wrong"], color="k", ls=":", lw=0.8)
            ax.set_xlabel("wrong fraction (all stars)")
            ax.set_ylabel("success fraction (all stars)")
            ax.set_title(f"dev_medium_b_clean {mode}" + (" (zoom)" if row else ""), fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6.5, loc="lower right")
            if row:
                ax.set_xlim(*zoom[mode][0])
                ax.set_ylim(*zoom[mode][1])
    fig.tight_layout()
    fig.savefig(FIG / "fig_frontier.png", dpi=130)
    plt.close(fig)


def _map(ax, t, xcol, ycol, val, target, title, fmt="{:d}"):
    piv = t.pivot_table(index=ycol, columns=xcol, values=val)
    wr = t.pivot_table(index=ycol, columns=xcol, values="wrong")
    im = ax.imshow(np.ma.masked_invalid(piv.values), origin="lower", aspect="auto", cmap="viridis" if val == "n_success" else "magma_r")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{x:g}" for x in piv.columns], fontsize=7)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{y:g}" for y in piv.index], fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if not np.isfinite(piv.values[i, j]):
                continue  # combination not in the coarse or fine grid
            ok = wr.values[i, j] <= target + 1e-12
            ax.text(j, i, fmt.format(int(piv.values[i, j])), ha="center", va="center", fontsize=6.5,
                    color="w" if not ok else "k",
                    bbox=dict(boxstyle="square,pad=0.15", fc="#ffffffcc", ec="none") if ok else None)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)


def fig_map(chosen):
    sw = pd.read_csv(DATA / "sweep_B.csv")
    t = sw[(sw["mode"] == "fullyear") & (sw.variant == NEW)]
    c = chosen["chosen"]
    tgt = chosen["fullyear"]["target_wrong"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    a = t[np.isclose(t.fp, c["fp"])]
    b = t[t.mp == c["mp"]]
    _map(axes[0, 0], a, "snr", "mp", "n_success", tgt, f"successes (of 1000), frac_pos = {c['fp']:g}")
    _map(axes[0, 1], a, "snr", "mp", "n_wrong", tgt, f"wrongs, frac_pos = {c['fp']:g}")
    _map(axes[1, 0], b, "snr", "fp", "n_success", tgt, f"successes, min_peaks = {c['mp']}")
    _map(axes[1, 1], b, "snr", "fp", "n_wrong", tgt, f"wrongs, min_peaks = {c['mp']}")
    fig.suptitle(f"dev_medium_b_clean full-year, new gates (l0x2): white-boxed cells meet wrong <= {tgt:.3f} "
                 f"(_v9); pick ({c['mp']}, {c['fp']:g}, {c['snr']:g})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_map.png", dpi=130)
    plt.close(fig)


def fig_bins():
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
    for j, (set_name, mode) in enumerate([(s, m) for s in ("dev_medium_clean", "dev_medium_b_clean")
                                          for m in ("fullyear", "assigned")]):
        for run, off, col in (("old", -0.12, COL[OLD]), ("new", 0.12, COL[NEW])):
            d = per_star(set_name, mode, run)
            for i, (lab, cats) in enumerate([("success", S), ("alias", A), ("wrong", {"wrong"})]):
                ks = [d[d.period_bin == b].final_outcome.isin(cats).sum() for b in BINS]
                ns = [(d.period_bin == b).sum() for b in BINS]
                p = np.array(ks) / np.maximum(ns, 1)
                lo, hi = np.array([F44.wilson(k, n) for k, n in zip(ks, ns)]).T
                axes[i, j].errorbar(np.arange(len(BINS)) + off, p, yerr=[p - lo, hi - p], fmt="o", color=col, ms=4,
                                    capsize=2, label="v9 default (l0x)" if run == "old" else NEW)
                axes[i, j].set_ylabel(lab)
            axes[0, j].set_title(f"{set_name} {mode}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels(BINS)
        ax.set_xlabel("true period bin")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_bins.png", dpi=130)
    plt.close(fig)


def fig_gates():
    gates = ["n_peaks", "frac_positive_heights", "mean_height_snr", "frac_vertex_tol", "half_lag_consistency",
             "few_peaks_height"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for j, (set_name, mode) in enumerate([(s, m) for s in ("dev_medium_clean", "dev_medium_b_clean")
                                          for m in ("fullyear", "assigned")]):
        ax = axes[j]
        for run, off, col in (("old", -0.2, COL[OLD]), ("new", 0.2, COL[NEW])):
            d = per_star(set_name, mode, run)
            bl = d[~d.final_outcome.isin(S) & d.best_success_failed_gates.notna()]
            cnt = [bl.best_success_failed_gates.str.split(";").apply(lambda g: gt in g).sum() for gt in gates]
            ax.barh(np.arange(len(gates)) + off, cnt, height=0.4, color=col,
                    label=("v9 default" if run == "old" else NEW) + f" ({len(bl)} stars)")
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
            for run in ("old", "new", "alt"):
                o = per_star(set_name, mode, run).final_outcome
                rows.append(dict(set=set_name, mode=mode, run=run, n=len(o), success=o.isin(S).mean(),
                                 wrong=(o == "wrong").mean(), alias=o.isin(A).mean(), none=(o == "none").mean(),
                                 precision=o.isin(S).sum() / max((o != "none").sum(), 1)))
                if run == "old":
                    continue
                a = per_star(set_name, mode, "old").set_index("star_id").final_outcome
                b = per_star(set_name, mode, run).set_index("star_id").final_outcome.reindex(a.index)
                rows[-1].update(gained=int((b.isin(S) & ~a.isin(S)).sum()), lost=int((a.isin(S) & ~b.isin(S)).sum()),
                                new_wrong=int(((b == "wrong") & (a != "wrong")).sum()),
                                fixed_wrong=int(((a == "wrong") & (b != "wrong")).sum()))
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "summary.csv", index=False)
    return out


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    chosen = json.load(open(DATA / "chosen.json"))
    fig_frontier(chosen)
    fig_map(chosen)
    if all((DATA / f"per_star/{NEW}_{t}.csv.gz").exists() for t in ("dmfy", "dmas", "bfy", "bas")):
        fig_bins()
        fig_gates()
        pd.set_option("display.width", 200)
        print(summary().round(3).to_string(index=False))

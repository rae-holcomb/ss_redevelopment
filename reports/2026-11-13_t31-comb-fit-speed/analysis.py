#!/usr/bin/env python3
"""T31 report: collect run metrics / timings into data/ and draw the figures.

Usage
-----
    python reports/2026-11-13_t31-comb-fit-speed/analysis.py <scratch dir> [--recommended NAME] [--top N1,N2,N3]

<scratch dir> holds:
    runs/<name>_dmfy, runs/<name>_dmas    scorecard runs (dev_medium full-year / assigned), where <name> is
                                          main (pre-T31 code), A (Part A), A_rep (Part A rerun) or
                                          v_<comb variant>[__<cwt variant>] (Part B, on top of Part A)
    prof/prof_<comb>_<cwt>_<fy|as>_r<k>.csv   benchmarks/profile_pipeline.py outputs (15 stars, serial)
    identity_partA.csv                    Part A identity checks (written by the T31 session)

Writes data/*.csv and figures/*.png next to this file.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
STAGES = ["load", "descriptors", "acf_long", "acf_short", "guess_acf_fft", "guess_acf_fft_highpass",
          "guess_acf_fft_short", "guess_lombscargle", "guess_lombscargle_short", "guess_pairwise_histogram",
          "guess_wavelet", "fit", "gate_records"]


def run_name_to_variant(name: str) -> tuple:
    """Map a run folder stem to (comb variant, cwt variant)."""
    if name in ("main", "A", "A_rep"):
        return "lsq", "full"
    body = name[2:]
    comb, _, cwt = body.partition("__")
    return comb, (cwt or "full")


def collect_metrics(scratch: Path) -> pd.DataFrame:
    """One row per (run, mode): final success/outcome fractions, per-bin success, wavelet per-method metrics."""
    rows = []
    for d in sorted(glob.glob(str(scratch / "runs" / "*_dm??"))):
        d = Path(d)
        if not (d / "metrics.json").exists():
            continue
        name, mode = d.name.rsplit("_", 1)
        m = json.loads((d / "metrics.json").read_text())
        comb, cwt = run_name_to_variant(name)
        r = dict(run=name, mode={"dmfy": "fullyear", "dmas": "assigned"}[mode], comb=comb, cwt=cwt)
        for k in ("success", "correct", "diffrot", "alias", "wrong", "none"):
            r[k] = m["final"]["overall"][k]
        for b in BINS:
            r[f"success_{b}"] = m["final"]["by_bin"][b]["success"]
            r[f"n_{b}"] = m["final"]["by_bin"][b]["n"]
        w = m["methods"]["wavelet"]["overall"]
        r.update(wavelet_recall=w["recall"], wavelet_top1_success=w["top1_success"],
                 wavelet_median_rank=w["median_rank"])
        r["wall_seconds"] = m["config"]["wall_seconds"]
        rows.append(r)
    return pd.DataFrame(rows)


def collect_profiles(scratch: Path) -> pd.DataFrame:
    """Per-star stage seconds from every profile CSV, tagged with comb / cwt variant, mode and round."""
    frames = []
    for f in sorted(glob.glob(str(scratch / "prof" / "prof_*.csv"))):
        stem = Path(f).name[len("prof_"):-len(".csv")]
        parts = stem.split("_")
        rnd = int(parts[-1][1:])
        mode = parts[-2]
        rest = "_".join(parts[:-2])
        # comb variant names contain underscores; cwt names are from a fixed list
        for cwt in ("dec_n100", "full", "lean", "dec", "n100"):
            if rest.endswith("_" + cwt):
                comb = rest[: -len(cwt) - 1]
                break
        df = pd.read_csv(f)
        df["comb"], df["cwt"], df["mode"], df["round"] = comb, cwt, {"fy": "fullyear", "as": "assigned"}[mode], rnd
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


TIMING_ROUNDS = (2, 3)  # round 1 (03:43-03:54) ran ~40% slow machine-wide (post-login Spotlight indexing):
# even acf_long, which no variant touches, took 0.020 s/star instead of 0.014. Rounds 2-3 ran on a quiet machine.


def timing_summary(prof: pd.DataFrame) -> pd.DataFrame:
    """Mean seconds per star (over the 15 stars and TIMING_ROUNDS) for every stage and the total, per variant and mode."""
    cols = [c for c in STAGES if c in prof.columns] + ["total"]
    prof = prof[prof["round"].isin(TIMING_ROUNDS)]
    g = prof.groupby(["comb", "cwt", "mode"])[cols].mean().reset_index()
    return g


def fig_frontier(met, tim, out, main_label="lsq"):
    """Success (top) and wrong rate (bottom) vs serial s/star, one point per comb variant, both modes; main marked.

    Final answers don't depend on the CWT variant (the wavelet isn't pooled), so each comb variant's run is used
    whichever CWT variant rode along; its time is the comb variant's cwt=full profile.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex="col")
    for col, mode in enumerate(("fullyear", "assigned")):
        t = tim[(tim["cwt"] == "full") & (tim["mode"] == mode)].set_index("comb")
        mm = met[(met["mode"] == mode) & (met["run"].str.startswith("v_") | (met["run"] == "A"))]
        mm = mm[mm["comb"].isin(t.index)].copy()
        mm["x"] = [t.loc[c, "total"] for c in mm["comb"]]
        mm = mm.sort_values("x")
        base = mm[mm["run"] == "A"].iloc[0]
        for row, metric in enumerate(("success", "wrong")):
            ax = axes[row, col]
            ax.axhline(base[metric], color="crimson", lw=0.8, ls=":")
            for i, (_, r) in enumerate(mm.iterrows()):
                is_main = r["run"] == "A"
                ax.scatter(r["x"], r[metric], s=130 if is_main else 50, marker="*" if is_main else "o",
                           color="crimson" if is_main else ("tab:green" if r["comb"] == "local_screen2" else "tab:blue"),
                           zorder=3)
                dy = (10 if i % 2 == 0 else -14)
                ax.annotate("main (lsq)" if is_main else r["comb"], (r["x"], r[metric]), textcoords="offset points",
                            xytext=(4, dy), fontsize=7.5)
            ax.set_ylabel({"success": "final success rate (correct + diffrot)", "wrong": "final wrong rate"}[metric])
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f"dev_medium {mode} (n=1000)")
            else:
                ax.set_xlabel("serial seconds per star (scorecard path, 15-star set, nice 15)")
    fig.suptitle("Accuracy vs time per comb-fit variant (red star / dotted line = current main; green = recommended); "
                 "lsq_lsmr (6.4 / 4.0 s, not scored) is off the plot")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_fit_frontier(met, tim, out):
    """Success vs comb-fit-stage seconds only (the part the variants change)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, mode in zip(axes, ("fullyear", "assigned")):
        t = tim[(tim["cwt"] == "full") & (tim["mode"] == mode)].set_index("comb")
        mm = met[(met["mode"] == mode) & (met["run"].str.startswith("v_") | (met["run"] == "A"))]
        for _, r in mm.iterrows():
            if r["comb"] not in t.index:
                continue
            x = t.loc[r["comb"], "fit"]
            is_main = r["run"] == "A"
            ax.scatter(x, r["success"], s=110 if is_main else 50, marker="*" if is_main else "o",
                       color="crimson" if is_main else "tab:purple", zorder=3)
            ax.annotate("main (lsq)" if is_main else r["comb"], (x, r["success"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
        ax.set_xlabel("comb-fit stage seconds per star (serial)")
        ax.set_ylabel("final success rate")
        ax.set_title(f"dev_medium {mode}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_by_bin(met, variants, out):
    """Success by period bin, main vs the listed variants, both modes (with per-bin n)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    runs = ["A"] + list(variants)
    w = 0.8 / len(runs)
    for ax, mode in zip(axes, ("fullyear", "assigned")):
        for i, run in enumerate(runs):
            r = met[(met["run"] == run) & (met["mode"] == mode)]
            if r.empty:
                continue
            r = r.iloc[0]
            vals = [r[f"success_{b}"] for b in BINS]
            ax.bar(np.arange(5) + (i - (len(runs) - 1) / 2) * w, vals, w,
                   label="main (lsq)" if run == "A" else run_name_to_variant(run)[0],
                   color="crimson" if run == "A" else None)
        r0 = met[(met["run"] == "A") & (met["mode"] == mode)].iloc[0]
        ax.set_xticks(range(5), [f"{b}\n(n={int(r0[f'n_{b}'])})" for b in BINS])
        ax.set_title(f"dev_medium {mode}")
        ax.set_ylabel("final success rate")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_stage_breakdown(tim, rec, out):
    """Stacked per-stage serial seconds per star, main vs the recommended variant, both modes."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    cols = [c for c in STAGES if c in tim.columns]
    labels, bottoms = [], []
    rows = []
    for mode in ("fullyear", "assigned"):
        for comb in ("main", "lsq", rec):
            r = tim[(tim["comb"] == comb) & (tim["cwt"] == "full") & (tim["mode"] == mode)]
            if r.empty:
                continue
            lab = {"main": "pre-T31 main", "lsq": "Part A (lsq)"}.get(comb, comb)
            rows.append((f"{lab}\n{mode}", r.iloc[0]))
    cmap = plt.get_cmap("tab20")
    for i, (lab, r) in enumerate(rows):
        b = 0
        for j, c in enumerate(cols):
            ax.barh(i, r[c], left=b, color=cmap(j), label=c if i == 0 else None)
            b += r[c]
        ax.text(b + 0.02, i, f"{b:.2f} s", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("serial seconds per star (mean over 15 stars x rounds)")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def gain_loss(scratch: Path, run: str, mode: str) -> pd.DataFrame:
    """Per-star main (A) vs run `run` (folder stem, e.g. v_local__dec): outcomes and success changes."""
    suf = {"fullyear": "dmfy", "assigned": "dmas"}[mode]
    a = pd.read_csv(scratch / "runs" / f"A_{suf}" / "per_star.csv")
    b = pd.read_csv(scratch / "runs" / f"{run}_{suf}" / "per_star.csv")
    m = a.merge(b, on="star_id", suffixes=("_main", "_var"))
    ok = {"correct", "diffrot"}
    m["succ_main"] = m["final_outcome_main"].isin(ok)
    m["succ_var"] = m["final_outcome_var"].isin(ok)
    same_p = (m["final_period_main"] == m["final_period_var"]) | (m["final_period_main"].isna() & m["final_period_var"].isna())
    m["change"] = np.where(m["succ_main"] & ~m["succ_var"], "lost",
                           np.where(~m["succ_main"] & m["succ_var"], "gained", np.where(same_p, "", "period_moved")))
    m["mode"] = mode
    keep = ["star_id", "mode", "true_period_main", "period_bin_main", "coverage_class_main", "final_period_main",
            "final_outcome_main", "decline_gates_main", "final_period_var", "final_outcome_var", "decline_gates_var",
            "change"]
    return m[keep].rename(columns={"true_period_main": "true_period", "period_bin_main": "period_bin",
                                   "coverage_class_main": "coverage_class"})


def fig_examples(gl: pd.DataFrame, rec: str, kind: str, out: Path, n: int = 4):
    """Light curve + ACF for up to n stars whose `change` is `kind` (gained / lost / period_moved), periods marked."""
    import scorecard as sc
    from types import SimpleNamespace
    from acf_utils import compute_acf
    sel = gl[gl["change"] == kind].sort_values("true_period")
    if sel.empty:
        return 0
    idx = np.linspace(0, len(sel) - 1, min(n, len(sel))).round().astype(int)
    sel = sel.iloc[np.unique(idx)]
    fig, axes = plt.subplots(len(sel), 2, figsize=(13, 2.6 * len(sel)), squeeze=False)
    for (ax_lc, ax_acf), (_, r) in zip(axes, sel.iterrows()):
        args = SimpleNamespace(synthetic=None, manifest=ROOT / "datasets" / "dev_medium.csv", seed=0,
                               data_root=str(ROOT / "data" / "smarts_dev"),
                               ignore_manifest_sectors=r["mode"] == "fullyear", coverage=sc.cov.FULL)
        task = next(t for t in sc.build_tasks(args) if t["star_id"] == r["star_id"])
        _, t, f, *_ = sc._load(task)
        ax_lc.plot(t, f, ",", color="k")
        ax_lc.set_title(f"{r['star_id'][-6:]} ({r['mode']}, {r['coverage_class']})  P_true={r['true_period']:.2f} d",
                        fontsize=9)
        ax_lc.set_xlabel("time [d]")
        lags, acf = compute_acf(t, f)
        ax_acf.plot(lags, acf, color="0.3", lw=0.8)
        P = r["true_period"]
        for k in range(1, int(lags[-1] / P) + 1):
            ax_acf.axvline(k * P, color="green", lw=0.8, alpha=0.6, ls="--" if k > 1 else "-")
        for col, colr, lab in (("final_period_main", "crimson", "main"), ("final_period_var", "tab:blue", rec)):
            v = r[col]
            if np.isfinite(v):
                ax_acf.axvline(v, color=colr, lw=1.6, alpha=0.8)
        ax_acf.set_xlim(0, min(lags[-1], 6 * max(P, np.nan_to_num(r["final_period_main"]),
                                                np.nan_to_num(r["final_period_var"]))))
        ax_acf.set_title(f"ACF: green=P_true (+harmonics), red=main {r['final_period_main']:.3g} d "
                         f"({r['final_outcome_main']}), blue={rec} {r['final_period_var']:.3g} d "
                         f"({r['final_outcome_var']})", fontsize=8)
        dg = [f"{lab} declined: {r[c]}" for c, lab in (("decline_gates_main", "main"), ("decline_gates_var", rec))
              if isinstance(r[c], str) and r[c]]
        if dg:
            ax_acf.text(0.99, 0.97, "\n".join(dg), transform=ax_acf.transAxes, ha="right", va="top", fontsize=7,
                        bbox=dict(fc="white", alpha=0.8, lw=0))
        ax_acf.set_xlabel("lag [d]")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return len(sel)


def fig_cwt(met, tim, out):
    """Wavelet per-method recall (any rank) vs wavelet-stage seconds, one point per CWT variant, both modes."""
    pairs = {"full": "A", "lean": "v_local_screen1__lean", "dec": "v_local__dec", "n100": "v_lsq_tol3__n100",
             "dec_n100": "v_lsq_tol4__dec_n100"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, mode in zip(axes, ("fullyear", "assigned")):
        for cwt, run in pairs.items():
            t = tim[(tim["comb"] == "lsq") & (tim["cwt"] == cwt) & (tim["mode"] == mode)]
            r = met[(met["run"] == run) & (met["mode"] == mode)]
            if t.empty or r.empty:
                continue
            x, y = t.iloc[0]["guess_wavelet"], r.iloc[0]["wavelet_recall"]
            ax.scatter(x, y, s=110 if cwt == "full" else 50, marker="*" if cwt == "full" else "o",
                       color="crimson" if cwt == "full" else "tab:orange", zorder=3)
            ax.annotate(f"{cwt} (top-1 {r.iloc[0]['wavelet_top1_success']:.3f})", (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
        ax.set_xlabel("guess_wavelet seconds per star (serial)")
        ax.set_ylabel("wavelet recall (success at any rank)")
        ax.set_title(f"dev_medium {mode}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_screen(scratch: Path, out: Path):
    """How many gates each candidate fails under the local fit, split by whether the lsq fit passes it.

    The local_screen variants refit with lsq every candidate at or left of their threshold (k = 1, 2).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    rows = []
    for ax, (mode, suf) in zip(axes, (("fullyear", "dmfy"), ("assigned", "dmas"))):
        a = pd.read_csv(scratch / "runs" / f"A_{suf}" / "per_candidate.csv", low_memory=False)
        b = pd.read_csv(scratch / "runs" / f"v_local__dec_{suf}" / "per_candidate.csv", low_memory=False)
        a, b = a[a.beyond_reach != True], b[b.beyond_reach != True]  # noqa: E712
        j = a.merge(b, on=["star_id", "P0", "method"], suffixes=("_a", "_b"))
        nf = j["failed_gates_b"].fillna("").astype(str).map(lambda x: 0 if x in ("", "nan") else len(x.split(";")))
        bins = np.arange(-0.5, nf.max() + 1.5)
        for passed, col, lab in ((False, "0.6", "lsq fit fails gates"), (True, "tab:green", "lsq fit passes gates")):
            sel = nf[j["passed_gates_a"] == passed]
            ax.hist(sel, bins=bins, color=col, alpha=0.8, label=f"{lab} (n={len(sel)})", log=True)
        for k, ls in ((1, "--"), (2, "-")):
            ax.axvline(k + 0.5, color="crimson", ls=ls, label=f"local_screen{k} refits <= {k}")
            rows.append(dict(mode=mode, k=k, refits_per_star=float((nf <= k).sum() / j["star_id"].nunique()),
                             recall_lsq_pass=float(((nf <= k) & j["passed_gates_a"]).sum() / j["passed_gates_a"].sum()),
                             recall_main_selected=float(((nf <= k) & (j["selected_a"] == True)).sum()  # noqa: E712
                                                        / (j["selected_a"] == True).sum())))  # noqa: E712
        ax.set_xlabel("gates failed under the local fit")
        ax.set_ylabel("candidates (log)")
        ax.set_title(f"dev_medium {mode}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows)


GATES = ("n_peaks", "frac_positive_heights", "mean_height_snr", "frac_vertex_in_window", "half_lag_consistency",
         "few_peaks_height")


def fig_gate_rates(scratch: Path, out: Path) -> pd.DataFrame:
    """Fraction of fitted candidates failing each gate, main (lsq) vs local, both modes."""
    rows = []
    for mode, suf in (("fullyear", "dmfy"), ("assigned", "dmas")):
        for run, lab in (("A", "main (lsq)"), ("v_local__dec", "local"), ("v_local_screen2", "local_screen2")):
            c = pd.read_csv(scratch / "runs" / f"{run}_{suf}" / "per_candidate.csv", low_memory=False)
            c = c[(c.beyond_reach != True) & c.error.isna()]  # noqa: E712
            for g in GATES:
                rows.append(dict(mode=mode, variant=lab, gate=g, fail_rate=1 - c[g + "_pass"].astype(bool).mean()))
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, mode in zip(axes, ("fullyear", "assigned")):
        d = df[df["mode"] == mode]
        for i, (lab, col) in enumerate((("main (lsq)", "crimson"), ("local", "tab:orange"), ("local_screen2", "tab:green"))):
            v = d[d["variant"] == lab].set_index("gate").loc[list(GATES), "fail_rate"]
            ax.bar(np.arange(len(GATES)) + (i - 1) * 0.27, v, 0.27, label=lab, color=col)
        ax.set_xticks(range(len(GATES)), [g.replace("_", "\n", 1) for g in GATES], fontsize=7.5)
        ax.set_title(f"dev_medium {mode}: fitted candidates failing each gate")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("fraction of candidates failing")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return df


def fig_identity(ident: pd.DataFrame, out: Path):
    """Part A identity-check table rendered as a figure (green = identical, orange = wavelet-only diffs)."""
    fig, ax = plt.subplots(figsize=(13, 0.45 * len(ident) + 1.0))
    ax.axis("off")
    cells = ident.astype(str).values.tolist()
    tab = ax.table(cellText=cells, colLabels=list(ident.columns), loc="center", cellLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.4)
    for i, row in enumerate(ident.itertuples(index=False), start=1):
        colr = "#d9f2d9" if str(row.verdict).startswith("identical") else "#fde3c4"
        for j in range(len(ident.columns)):
            tab[i, j].set_facecolor(colr)
    for j in range(len(ident.columns)):
        tab[0, j].set_facecolor("#dddddd")
    tab.auto_set_column_width(list(range(len(ident.columns))))
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    """Collect data and draw every report figure."""
    p = argparse.ArgumentParser()
    p.add_argument("scratch", type=Path)
    p.add_argument("--recommended", default="local_screen2", help="comb variant for the stage breakdown")
    p.add_argument("--top", default="v_local_screen2,v_local__dec,v_lsq_tol3__n100",
                   help="comma-separated run names for the by-bin figure")
    p.add_argument("--examples", default="v_local__dec", help="run whose gained/lost stars are plotted")
    a = p.parse_args()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    met = collect_metrics(a.scratch)
    met.to_csv(HERE / "data" / "variant_metrics.csv", index=False)
    prof = collect_profiles(a.scratch)
    prof.to_csv(HERE / "data" / "profile_per_star.csv", index=False)
    tim = timing_summary(prof)
    tim.to_csv(HERE / "data" / "timing_summary.csv", index=False)
    fig_frontier(met, tim, HERE / "figures" / "fig_frontier.png")
    fig_fit_frontier(met, tim, HERE / "figures" / "fig_frontier_fit_stage.png")
    fig_stage_breakdown(tim, a.recommended, HERE / "figures" / "fig_stage_breakdown.png")
    fig_cwt(met, tim, HERE / "figures" / "fig_cwt_variants.png")
    fig_by_bin(met, [v for v in a.top.split(",") if v], HERE / "figures" / "fig_success_by_bin.png")
    fig_screen(a.scratch, HERE / "figures" / "fig_screen_threshold.png").to_csv(HERE / "data" / "screen_recall.csv",
                                                                               index=False)
    fig_gate_rates(a.scratch, HERE / "figures" / "fig_gate_fail_rates.png").to_csv(HERE / "data" / "gate_fail_rates.csv",
                                                                                index=False)
    for run in (a.examples, "v_" + a.recommended):
        gl = pd.concat([gain_loss(a.scratch, run, m) for m in ("fullyear", "assigned")])
        gl[gl["change"] != ""].to_csv(HERE / "data" / f"changed_stars_{run}.csv", index=False)
        label = run_name_to_variant(run)[0]
        kinds = ("lost", "gained", "period_moved") if run == "v_" + a.recommended else ("lost", "gained")
        for kind in kinds:
            k = fig_examples(gl, label, kind, HERE / "figures" / f"fig_examples_{label}_{kind}.png", n=5)
            print(run, kind, k)
    ident = a.scratch / "identity_partA.csv"
    df = pd.read_csv(ident)
    df.to_csv(HERE / "data" / "identity_partA.csv", index=False)
    fig_identity(df, HERE / "figures" / "fig_identity_partA.png")
    print(met[["run", "mode", "success", "wrong", "wavelet_recall"]].to_string())
    print(tim[["comb", "cwt", "mode", "fit", "guess_wavelet", "total"]].round(3).to_string())


if __name__ == "__main__":
    main()

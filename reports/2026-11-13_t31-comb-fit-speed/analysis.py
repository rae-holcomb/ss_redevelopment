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


def timing_summary(prof: pd.DataFrame) -> pd.DataFrame:
    """Mean seconds per star (over stars and rounds) for every stage and the total, per variant and mode."""
    cols = [c for c in STAGES if c in prof.columns] + ["total"]
    g = prof.groupby(["comb", "cwt", "mode"])[cols].mean().reset_index()
    return g


def fig_frontier(met, tim, out, main_label="lsq"):
    """Success vs serial s/star, one point per comb variant (cwt=full), both modes; main marked."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, mode in zip(axes, ("fullyear", "assigned")):
        t = tim[(tim["cwt"] == "full") & (tim["mode"] == mode)].set_index("comb")
        mm = met[(met["mode"] == mode) & (met["cwt"] == "full") & met["run"].str.startswith("v_")]
        mm = pd.concat([mm, met[(met["mode"] == mode) & (met["run"] == "A")]])
        for _, r in mm.iterrows():
            if r["comb"] not in t.index:
                continue
            x = t.loc[r["comb"], "total"]
            is_main = r["run"] == "A"
            ax.scatter(x, r["success"], s=110 if is_main else 50, marker="*" if is_main else "o",
                       color="crimson" if is_main else "tab:blue", zorder=3)
            ax.annotate("main (lsq)" if is_main else r["comb"], (x, r["success"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
        # fit-only time as a secondary marker row
        ax.set_xlabel("serial seconds per star (scorecard path, 15-star set, nice 15)")
        ax.set_ylabel("final success rate (correct + diffrot)")
        ax.set_title(f"dev_medium {mode}")
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy vs time per comb-fit variant (red star = current main)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_fit_frontier(met, tim, out):
    """Success vs comb-fit-stage seconds only (the part the variants change)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, mode in zip(axes, ("fullyear", "assigned")):
        t = tim[(tim["cwt"] == "full") & (tim["mode"] == mode)].set_index("comb")
        mm = met[(met["mode"] == mode) & (met["cwt"] == "full") & (met["run"].str.startswith("v_") | (met["run"] == "A"))]
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
    runs = ["A"] + [f"v_{v}" for v in variants]
    w = 0.8 / len(runs)
    for ax, mode in zip(axes, ("fullyear", "assigned")):
        for i, run in enumerate(runs):
            r = met[(met["run"] == run) & (met["mode"] == mode)]
            if r.empty:
                continue
            r = r.iloc[0]
            vals = [r[f"success_{b}"] for b in BINS]
            ax.bar(np.arange(5) + (i - (len(runs) - 1) / 2) * w, vals, w,
                   label="main (lsq)" if run == "A" else run[2:], color="crimson" if run == "A" else None)
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
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cols = [c for c in STAGES if c in tim.columns]
    labels, bottoms = [], []
    rows = []
    for mode in ("fullyear", "assigned"):
        for comb in ("lsq", rec):
            r = tim[(tim["comb"] == comb) & (tim["cwt"] == "full") & (tim["mode"] == mode)]
            if r.empty:
                continue
            rows.append((f"{'main' if comb == 'lsq' else comb}\n{mode}", r.iloc[0]))
    cmap = plt.get_cmap("tab20")
    for i, (lab, r) in enumerate(rows):
        b = 0
        for j, c in enumerate(cols):
            ax.barh(i, r[c], left=b, color=cmap(j), label=c if i == 0 else None)
            b += r[c]
        ax.text(b + 0.02, i, f"{b:.2f} s", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("serial seconds per star (mean over 15 stars x rounds)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def gain_loss(scratch: Path, rec: str, mode: str) -> pd.DataFrame:
    """Stars whose final outcome changes success status between main (A) and variant `rec`."""
    suf = {"fullyear": "dmfy", "assigned": "dmas"}[mode]
    a = pd.read_csv(scratch / "runs" / f"A_{suf}" / "per_star.csv")
    b = pd.read_csv(scratch / "runs" / f"v_{rec}_{suf}" / "per_star.csv")
    m = a.merge(b, on="star_id", suffixes=("_main", "_var"))
    ok = {"correct", "diffrot"}
    m["succ_main"] = m["final_outcome_main"].isin(ok)
    m["succ_var"] = m["final_outcome_var"].isin(ok)
    m["change"] = np.where(m["succ_main"] & ~m["succ_var"], "lost",
                           np.where(~m["succ_main"] & m["succ_var"], "gained", ""))
    m["mode"] = mode
    keep = ["star_id", "mode", "true_period_main", "period_bin_main", "coverage_class_main", "final_period_main",
            "final_outcome_main", "final_period_var", "final_outcome_var", "change"]
    return m[keep].rename(columns={"true_period_main": "true_period", "period_bin_main": "period_bin",
                                   "coverage_class_main": "coverage_class"})


def fig_examples(gl: pd.DataFrame, rec: str, kind: str, out: Path, n: int = 4):
    """Light curve + ACF for up to n stars that `rec` gained or lost vs main, periods marked."""
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
        ax_acf.set_xlabel("lag [d]")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return len(sel)


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
    p.add_argument("--recommended", default=None)
    p.add_argument("--top", default="", help="comma-separated variants for the by-bin figure")
    a = p.parse_args()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    met = collect_metrics(a.scratch)
    met.to_csv(HERE / "data" / "variant_metrics.csv", index=False)
    prof = collect_profiles(a.scratch)
    if not prof.empty:
        prof.to_csv(HERE / "data" / "profile_per_star.csv", index=False)
        tim = timing_summary(prof)
        tim.to_csv(HERE / "data" / "timing_summary.csv", index=False)
        fig_frontier(met, tim, HERE / "figures" / "fig_frontier.png")
        fig_fit_frontier(met, tim, HERE / "figures" / "fig_frontier_fit_stage.png")
        if a.recommended:
            fig_stage_breakdown(tim, a.recommended, HERE / "figures" / "fig_stage_breakdown.png")
    top = [v for v in a.top.split(",") if v]
    if top:
        fig_by_bin(met, top, HERE / "figures" / "fig_success_by_bin.png")
    if a.recommended:
        gl = pd.concat([gain_loss(a.scratch, a.recommended, m) for m in ("fullyear", "assigned")])
        gl.to_csv(HERE / "data" / f"gain_loss_{a.recommended}.csv", index=False)
        for kind in ("lost", "gained"):
            k = fig_examples(gl[gl["change"] != ""], a.recommended, kind, HERE / "figures" / f"fig_examples_{kind}.png",
                             n=6)
            print(kind, k)
    ident = a.scratch / "identity_partA.csv"
    if ident.exists():
        df = pd.read_csv(ident)
        df.to_csv(HERE / "data" / "identity_partA.csv", index=False)
        fig_identity(df, HERE / "figures" / "fig_identity_partA.png")
    print(met[["run", "mode", "success", "wavelet_recall", "wavelet_top1_success"]].to_string())


if __name__ == "__main__":
    main()

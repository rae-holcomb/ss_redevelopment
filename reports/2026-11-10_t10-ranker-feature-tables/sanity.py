"""sanity.py -- T10 sanity checks on built feature tables, with figures and small CSVs.

For each run (label=outdir): label balance per period bin, candidates per star, fraction
of stars with any relevance >= 1, gate-failure rates, and the final-answer parity check
(the selected row vs stars.final_period, and vs a scorecard per_star.csv when given).

Usage (repo root):
    python reports/2026-11-10_t10-ranker-feature-tables/sanity.py \
        --run dev_medium_assigned=runs/t10_dev_medium_assigned \
        --run dev_medium_fullyear=runs/t10_dev_medium_fullyear \
        --run train_1k=/Volumes/Data_Vault/smarts_data/extracted/t10_tables/train_1k \
        [--scorecard dev_medium_fullyear=<scorecard outdir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
import evaluation as ev  # noqa: E402
from comb_fit import GATE_NAMES  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
REL_COLORS = {0: "#bbbbbb", 1: "#2ca02c", 2: "#1f77b4"}


def load(outdir: Path) -> tuple:
    """(candidates, stars) DataFrames of one builder run."""
    return (pd.read_csv(outdir / "candidates.csv.gz", low_memory=False),
            pd.read_csv(outdir / "stars.csv.gz", low_memory=False))


def bins_present(s: pd.Series) -> list:
    """Period-bin labels in canonical order, keeping only those present."""
    return [b for b in BINS if b in set(s)] + sorted(set(s) - set(BINS))


def label_balance(c: pd.DataFrame) -> pd.DataFrame:
    """Rows per period bin x relevance (counts and fractions), fitted and unfit rows together."""
    t = c.groupby(["period_bin", "relevance"]).size().unstack(fill_value=0)
    t = t.reindex(bins_present(c.period_bin)).reindex(columns=[0, 1, 2], fill_value=0)
    t["n"] = t.sum(axis=1)
    for r in (0, 1, 2):
        t[f"frac_{r}"] = t[r] / t["n"]
    return t


def star_summary(c: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Per period bin: stars, mean rows, fraction with any relevance >= 1 (all / fitted), hand success."""
    s = s.copy()
    s["any_rel"] = s.max_relevance >= 1
    s["any_rel_fitted"] = s.n_relevant_fitted >= 1
    s["success"] = s.final_outcome.isin(ev.SUCCESS_OUTCOMES)
    g = s.groupby("period_bin")
    out = pd.DataFrame(dict(n_stars=g.size(), mean_rows=g.n_rows.mean(), frac_any_rel=g.any_rel.mean(),
                            frac_any_rel_fitted=g.any_rel_fitted.mean(), hand_success=g.success.mean(),
                            frac_no_rows=g.n_rows.apply(lambda x: (x == 0).mean())))
    out.loc["all"] = [len(s), s.n_rows.mean(), s.any_rel.mean(), s.any_rel_fitted.mean(), s.success.mean(),
                      (s.n_rows == 0).mean()]
    return out.reindex(bins_present(s.period_bin) + ["all"])


def gate_failure(c: pd.DataFrame) -> pd.DataFrame:
    """Per gate: fraction of fitted candidates failing it, for relevance >= 1 and relevance 0 rows."""
    fit = c[~c.fit_error.astype(bool) & ~c.beyond_reach.astype(bool)]
    rows = []
    for g in GATE_NAMES:
        col, en = f"gate_{g}_pass", f"gate_{g}_enabled"
        if col not in fit:
            continue
        on = fit[fit[en].astype(bool)] if en in fit else fit
        p = on[col].astype(bool)
        rel = on.relevance >= 1
        rows.append(dict(gate=g, fail_all=float((~p).mean()), fail_relevant=float((~p[rel]).mean()),
                         fail_irrelevant=float((~p[~rel]).mean()), n_relevant=int(rel.sum()),
                         sole_fail_relevant=float(((~p) & (on.n_failed_gates == 1) & rel).sum() / max(rel.sum(), 1))))
    pr = c.relevance >= 1
    rows.append(dict(gate="(fit_error)", fail_all=float(c.fit_error.astype(bool).mean()),
                     fail_relevant=float(c.fit_error.astype(bool)[pr].mean()),
                     fail_irrelevant=float(c.fit_error.astype(bool)[~pr].mean()), n_relevant=int(pr.sum())))
    rows.append(dict(gate="(beyond_reach)", fail_all=float(c.beyond_reach.astype(bool).mean()),
                     fail_relevant=float(c.beyond_reach.astype(bool)[pr].mean()),
                     fail_irrelevant=float(c.beyond_reach.astype(bool)[~pr].mean()), n_relevant=int(pr.sum())))
    return pd.DataFrame(rows)


def parity(c: pd.DataFrame, s: pd.DataFrame, scorecard_dir: Path | None) -> dict:
    """Does the selected row reproduce the final answer? Internal, and against a scorecard run."""
    sel = c[c.selected.astype(bool)].groupby("star_id").fitted_P.agg(["size", "first"])
    m = s.set_index("star_id").join(sel)
    succ = m.pipeline_success.astype(bool)
    out = dict(n_stars=len(m), n_success=int(succ.sum()),
               n_success_with_one_selected_row=int((m["size"][succ] == 1).sum()),
               n_declined_with_selected_row=int(m["size"][~succ].notna().sum()),
               max_rel_diff_selected_vs_final=float(np.nanmax(np.abs(m["first"][succ] / m.final_period[succ] - 1)))
               if succ.any() else None)
    if scorecard_dir is not None:
        ps = pd.read_csv(scorecard_dir / "per_star.csv").set_index("star_id")
        j = m.join(ps[["final_period", "final_outcome", "pipeline_success"]], rsuffix="_sc", how="inner")
        same_dec = (j.pipeline_success.astype(bool) == j.pipeline_success_sc.astype(bool))
        both = j.pipeline_success.astype(bool) & j.pipeline_success_sc.astype(bool)
        d = np.abs(j.final_period[both] / j.final_period_sc[both] - 1)
        out.update(scorecard_dir=str(scorecard_dir), n_joined=len(j), n_same_success_flag=int(same_dec.sum()),
                   n_both_success=int(both.sum()), n_period_identical=int((d == 0).sum()),
                   n_period_within_1e9=int((d < 1e-9).sum()), max_rel_diff=float(d.max()) if len(d) else None,
                   n_same_outcome=int((j.final_outcome == j.final_outcome_sc).sum()),
                   outcome_mismatches=j.loc[j.final_outcome != j.final_outcome_sc,
                                            ["final_outcome", "final_outcome_sc"]].value_counts().to_dict())
    return out


def figures(runs: dict, tables: dict):
    """The four sanity figures, one column per run."""
    names = list(runs)
    n = len(names)
    # 1. label balance per bin
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, k in zip(axes[0], names):
        lb = tables[k]["label_balance"]
        bottom = np.zeros(len(lb))
        for r in (0, 1, 2):
            ax.bar(lb.index, lb[f"frac_{r}"], bottom=bottom, color=REL_COLORS[r], label=f"relevance {r}")
            bottom += lb[f"frac_{r}"].to_numpy()
        for i, (b, nn) in enumerate(zip(lb.index, lb.n)):
            ax.text(i, 1.01, f"{nn}", ha="center", fontsize=7)
        ax.set_ylim(0, 1.08)
        ax.set_title(k, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0][0].set_ylabel("fraction of candidate rows")
    axes[0][0].legend(fontsize=7, loc="center left")
    fig.suptitle("Label balance per period bin (numbers = candidate rows)", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig_label_balance.png", dpi=110)
    # 2. candidates per star
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.2), squeeze=False)
    for ax, k in zip(axes[0], names):
        s = tables[k]["stars"]
        mx = int(s.n_rows.max())
        bins = np.arange(-0.5, mx + 1.5, 1)
        ax.hist([s.n_fit_candidates, s.n_fit_error + s.n_beyond_reach], bins=bins, histtype="step",
                label=["fitted", "fit error + beyond reach"], lw=1.4)
        ax.hist(s.n_rows, bins=bins, histtype="step", color="k", lw=1.4, label="all rows")
        ax.set_title(f"{k}: median {s.n_rows.median():.0f} rows/star", fontsize=9)
        ax.set_xlabel("candidates per star", fontsize=8)
        ax.tick_params(labelsize=8)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig_candidates_per_star.png", dpi=110)
    # 3. fraction of stars with any relevant candidate vs hand success
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
    for ax, k in zip(axes[0], names):
        ss = tables[k]["star_summary"].drop(index="all")
        x = np.arange(len(ss))
        ax.bar(x - 0.27, ss.frac_any_rel, 0.27, label="any row relevance >= 1", color="#2ca02c")
        ax.bar(x, ss.frac_any_rel_fitted, 0.27, label="... among fitted rows", color="#98df8a")
        ax.bar(x + 0.27, ss.hand_success, 0.27, label="hand arbitration success", color="#ff7f0e")
        ax.set_xticks(x, ss.index, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(k, fontsize=9)
    axes[0][0].set_ylabel("fraction of stars")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Ceiling for a re-ranker: stars with a relevant candidate vs current success", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig_any_relevant.png", dpi=110)
    # 4. gate failure
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, k in zip(axes[0], names):
        gf = tables[k]["gate_failure"]
        y = np.arange(len(gf))
        ax.barh(y + 0.2, gf.fail_relevant, 0.4, label="relevance >= 1 rows", color="#2ca02c")
        ax.barh(y - 0.2, gf.fail_irrelevant, 0.4, label="relevance 0 rows", color="#bbbbbb")
        ax.set_yticks(y, gf.gate, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("fraction failing (gates: of fitted rows)", fontsize=8)
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig_gate_failure.png", dpi=110)


def main(argv=None):
    """Compute every sanity table, write data/sanity_<run>_*.csv + data/parity.json, draw figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=outdir")
    ap.add_argument("--scorecard", action="append", default=[], help="label=scorecard outdir for parity")
    a = ap.parse_args(argv)
    runs = dict(r.split("=", 1) for r in a.run)
    scs = dict(r.split("=", 1) for r in a.scorecard)
    (HERE / "figures").mkdir(exist_ok=True)
    tables, par = {}, {}
    for k, d in runs.items():
        c, s = load(Path(d))
        tables[k] = dict(label_balance=label_balance(c), star_summary=star_summary(c, s),
                         gate_failure=gate_failure(c), stars=s)
        for name in ("label_balance", "star_summary", "gate_failure"):
            tables[k][name].to_csv(HERE / "data" / f"sanity_{k}_{name}.csv")
        par[k] = parity(c, s, Path(scs[k]) if k in scs else None)
        par[k].update(n_rows=len(c), n_errors=int(s.error.notna().sum()))
        print(f"== {k}\n", tables[k]["star_summary"].round(3).to_string(), "\n",
              tables[k]["gate_failure"].round(3).to_string(), "\n", json.dumps(par[k], indent=1, default=str))
    (HERE / "data" / "parity.json").write_text(json.dumps(par, indent=2, default=str))
    figures(runs, tables)


if __name__ == "__main__":
    main()

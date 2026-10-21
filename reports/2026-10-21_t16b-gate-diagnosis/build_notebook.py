#!/usr/bin/env python3
"""
build_notebook.py -- write notebook.ipynb for the T16b report (then execute it
with tools/run_notebook.py). The notebook regenerates every README figure and
table from data/ (written by make_data.py and examples.py).
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

cells = []


def md(s):
    """Append a markdown cell."""
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    """Append a code cell."""
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# T16b: which reliability gates over-decline on short coverage?

Regenerates every figure and table in `README.md` from `data/`:
- `data/stars.csv.gz`, `data/candidates.csv.gz` (from `make_data.py`, which collects the instrumented
  scorecard runs and the pb_scan noise candidate dumps);
- `data/examples.{json,npz}` (from `examples.py`).

Datasets: synthetic n=200 seed 0 under `--coverage full / consecutive-2 / gapped-2 / single` (coverage seed 0;
same stars each time), dev_medium with assigned coverage, pb_scan noise (40 light curves per 1/2/13 sectors).
Held-out validation: synthetic seed 1 (single, gapped-2) and pb_scan noise with seed offset 1500.
""")

code("""
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import gate_lib as gl, figs, rules
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
stars, cands = gl.load_data()
stars.groupby("dataset").size()
""")

md("## 0. Instrumentation is bit-identical (final periods and outcomes)")
code("""
print(open("data/bit_identity_compare_runs.txt").read())
""")

md("""
## 1. Outcomes per dataset
`noise-*` rows: pb_scan noise light curves (`noise` = the pipeline answered, i.e. a false positive).
""")
code("""
cls = np.where(stars.final_outcome.eq("none"), "none", gl.cand_class(stars.final_outcome))
overview = pd.crosstab(stars.dataset, cls)
overview["n"] = overview.sum(axis=1)
overview
""")

md("""
## 2. Declined stars that had a success candidate in the fitted pool, under three definitions
- **loose**: the candidate's fitted P scores success (`correct` or `diffrot`), which is what the brief asked for.
  For differential rotators the active-band range can span several times P_eq, so this over-counts.
- **strict**: fitted P within REL_TOL of P_eq (`correct`).
- **matched** (synthetic coverage runs only): within 3% of the same star's full-year final period, for stars
  the full-year run measured, i.e. "the short-coverage pool contains the answer the full-year run gives".
""")
code("""
rows = {}
for d in ("loose", "strict", "matched"):
    b = figs.rescue_candidates(stars, cands, d)
    rows[d] = b.groupby("dataset").size()
dec = stars[stars.final_outcome.eq("none")].groupby("dataset").size().rename("declined")
rescue_counts = pd.concat([dec, pd.DataFrame(rows)], axis=1).fillna(0).astype(int)
rescue_counts = rescue_counts.loc[[d for d in figs.COV_ORDER + ["dev_medium", "s1-single", "s1-gapped-2"] if d in rescue_counts.index]]
rescue_counts.to_csv("data/table_rescue_counts.csv")
rescue_counts
""")

md("""
How wide are the active-latitude bands of the stars whose only "success" candidates are loose?
(band width = (P_hi - P_lo) / P_eq; any period in that range scores success)
""")
code("""
lo = figs.rescue_candidates(stars, cands, "loose")
st = figs.rescue_candidates(stars, cands, "strict")
only_loose = lo[~lo.set_index(["dataset", "star_id"]).index.isin(st.set_index(["dataset", "star_id"]).index)]
w = only_loose.merge(stars[["dataset", "star_id", "period_lo", "period_hi"]], on=["dataset", "star_id"])
w["band_width"] = (w.period_hi - w.period_lo) / w.true_period
w.groupby("dataset").band_width.describe(percentiles=[.25, .5, .75]).round(2)
""")

md("## 3. Gate attribution: coverage x period bin x gate")
md("""
For each declined star, take its best (lowest-redchi) success candidate and list the gates it failed.
`any` = candidates failing that gate (with or without others); `sole` = failing only that gate (relaxing that gate alone
would make the candidate pass, though it may still lose to another passing candidate).
""")
code("""
fig = figs.fig_decline_attribution(stars, cands, "loose", out="figures/fig_attribution_loose.png")
""")
code("""
fig = figs.fig_decline_attribution(stars, cands, "strict", out="figures/fig_attribution_strict.png")
""")
code("""
tabs = {}
for d in ("loose", "strict", "matched"):
    b = figs.rescue_candidates(stars, cands, d)
    t = figs.gate_table(b[b.dataset.isin(figs.COV_ORDER + ["dev_medium"])], by=("dataset", "period_bin"))
    t.to_csv(f"data/table_gates_{d}_by_bin.csv")
    tabs[d] = t
tabs["loose"]
""")
code("""
tabs["strict"]
""")
code("""
b = figs.rescue_candidates(stars, cands, "strict")
figs.gate_table(b[b.dataset.isin(figs.COV_ORDER + ["dev_medium"])], by=("dataset",))
""")

md("""
### Strict rescue candidates: is their ACF structured at all?
Mean half-lag contrast of the strict rescue candidates. |contrast| < 0.02 is where pb_scan noise combs sit
(90th percentile of noise best candidates: ~0.01-0.02).
""")
code("""
st = figs.rescue_candidates(stars, cands, "strict")
st["structured"] = pd.cut(st.half_lag_contrast, [-np.inf, 0.02, 0.1, np.inf],
                          labels=["noise-like (<0.02)", "weak (0.02-0.1)", "structured (>=0.1)"])
pd.crosstab([st.dataset], st.structured).loc[lambda d: d.index.isin(figs.COV_ORDER + ["dev_medium"])]
""")
code("""
cols = ["dataset", "star_id", "true_period", "P_fit", "failed_gates", "mean_height_snr_value",
        "half_lag_consistency_value", "half_lag_contrast", "frac_vertex_in_window_value",
        "frac_positive_heights_value", "few_peaks_height_value", "n_teeth_available"]
st[st.half_lag_contrast >= 0.1].sort_values(["dataset", "true_period"])[cols].round(3)
""")

md("""
### Stage 1 recall by coverage (context)
Fraction of stars where a pipeline-pool method (pairwise_histogram, lombscargle, acf_fft) lists a success period
(loose), and where any method does, vs final success. The gap between recall and final success is the arbitration +
gate loss; the gap below recall is a Stage 1 loss that no gate change can fix.
""")
code("""
meth = pd.read_csv("data/methods.csv.gz")
meth["succ"] = meth.outcome.isin(gl.SUCCESS)
pool = meth[meth.method.isin(["pairwise_histogram", "lombscargle", "acf_fft"])].groupby(["dataset", "star_id"]).succ.any()
anym = meth.groupby(["dataset", "star_id"]).succ.any()
s = stars.set_index(["dataset", "star_id"])
s = s[s.index.get_level_values(0).isin(figs.COV_ORDER + ["dev_medium"])]
rec = pd.DataFrame({"pool_recall": pool, "any_recall": anym}).join(s[["period_bin", "final_outcome"]], how="inner")
rec["final_success"] = rec.final_outcome.isin(gl.SUCCESS)
recall_tab = rec.groupby(["dataset", "period_bin"])[["any_recall", "pool_recall", "final_success"]].mean().round(2).unstack(0)
recall_tab = recall_tab.reindex([b for b in gl.BIN_ORDER if b in recall_tab.index])
recall_tab.to_csv("data/table_recall_by_coverage.csv")
recall_tab
""")

md("## 4. Gate statistic distributions by coverage class")
md("""
Per star: the best success candidate (blue) vs the best wrong/alias candidate (red); grey = best candidate of each
pb_scan noise light curve with the matching number of sectors. Dashed = current threshold. (`frac vertex` /
`frac positive heights` are discrete; the few-peak ACF height only gates few-peak candidates.)
""")
code("""
syn = cands[cands.dataset.isin(figs.COV_ORDER + ["noise-1s", "noise-2s", "noise-13s"])]
fig = figs.fig_stat_distributions(stars, syn, out="figures/fig_stat_distributions_synth.png")
""")
code("""
dm = cands[cands.dataset.eq("dev_medium")].merge(stars[["dataset", "star_id", "coverage_class"]],
                                                   on=["dataset", "star_id"])
dm["dataset"] = dm.coverage_class.map(lambda c: c if c in ("full", "single") else
                                          ("2 sectors" if c.endswith("-2") else ("3-5" if "3-5" in c else "6-12")))
fig = figs.fig_stat_distributions(stars, dm, datasets=["full", "6-12", "3-5", "2 sectors", "single"], noise_map=None,
                                  out="figures/fig_stat_distributions_dev_medium.png",
                                  title="dev_medium (assigned coverage), by coverage class: best success vs best wrong/alias "
                                        "candidate per star (real-data 'wrong' labels: see T20 caveat)")
""")

md("## 5. Does any gate statistic scale with coverage?")
md("""
Left/middle: the same synthetic star's matched `correct` candidate, short coverage vs full year.
Right: ACF height at P for every fitted candidate on pb_scan noise (circled: few-peak candidates, the only ones
the 0.4 guard applies to) and synthetic few-peak success candidates; dotted = 0.25.
""")
code("""
fig = figs.fig_scaling(stars, cands[cands.dataset.isin(figs.COV_ORDER + ["noise-1s", "noise-2s", "noise-13s"])],
                       out="figures/fig_scaling.png")
""")
code("""
c = cands[cands.outcome.eq("correct") & cands.dataset.isin(figs.COV_ORDER)]
b = gl.best_per_star(c, c.P_fit.notna()); f = b[b.dataset == "full"].set_index("star_id")
out = {}
for ds in ["consecutive-2", "gapped-2", "single"]:
    j = b[b.dataset == ds].set_index("star_id").join(f, rsuffix="_f", how="inner")
    j = j[(j.P_fit / j.P_fit_f - 1).abs() < 0.05]
    out[ds] = {f"{col} ratio short/full ({q})": (j[col] / j[f"{col}_f"]).quantile(q)
               for col in ["mean_height_snr_value", "few_peaks_height_value", "half_lag_contrast"] for q in (0.25, 0.5)}
    out[ds]["n"] = len(j)
pd.DataFrame(out).round(3)
""")
code("""
# when does the few-peak guard apply? share of success candidates that are few_peaks, by coverage and period bin
s = cands[cands.cls.eq("success") & cands.dataset.isin(figs.COV_ORDER)].merge(
    stars[["dataset", "star_id", "period_bin"]], on=["dataset", "star_id"])
pd.crosstab(s.dataset, s.period_bin, values=s.few_peaks.astype(bool), aggfunc="mean").round(2)[
    [b for b in gl.BIN_ORDER if b in set(s.period_bin)]]
""")

md("## 6. Examples")
md("""
Blue = comb of the best success candidate, grey = lowest-redchi candidate; solid verticals at kP, dotted at
(k-1/2)P. The box lists the best success candidate's gate values.
""")
code("""
recs, arr = figs.load_examples()
fig = figs.fig_examples(recs, arr, out="figures/fig_examples.png")
""")

md("## 7. Offline trade-offs of alternative gate rules")
md("""
Replay: the selected candidate is the lowest-redchi candidate passing the rule (the pipeline's factor-of-2 alias
guard is not replayed; check below that the current-rule replay reproduces the real outcomes). Counts are deltas vs
the current rule; `d_noise` = extra pb_scan noise light curves that would get an answer (out of 40 per set).
Rules: `rules.py`. **Tuning sets** (seed 0, pb_scan noise offset 0); the PROPOSED2 rule was refined on these.
""")
code("""
gl.replay_agreement(stars, cands)
""")
code("""
TUNE = figs.COV_ORDER + ["noise-13s", "noise-2s", "noise-1s"]
t = gl.tradeoff(stars[stars.dataset.isin(TUNE)], cands[cands.dataset.isin(TUNE)], rules.RULES)
def show(t, ds_order):
    p = t.pivot_table(index="rule", columns="dataset", values=["d_success", "d_wrong", "d_alias", "d_noise"],
                      aggfunc="sum", sort=False)
    keep = [(m, d) for m in ["d_success", "d_wrong", "d_alias"] for d in ds_order if not d.startswith(("noise", "v-noise"))]
    keep += [("d_noise", d) for d in ds_order if d.startswith(("noise", "v-noise"))]
    return p[[k for k in keep if k in p.columns]]
trade_tune = show(t, TUNE)
trade_tune.to_csv("data/table_tradeoff_tuning.csv")
trade_tune
""")
md("**Held-out validation** (synthetic seed 1, pb_scan noise seed offset 1500; nothing was tuned on these).")
code("""
VAL = ["s1-single", "s1-gapped-2", "v-noise-13s", "v-noise-2s", "v-noise-1s"]
tv = gl.tradeoff(stars[stars.dataset.isin(VAL)], cands[cands.dataset.isin(VAL)], rules.RULES)
trade_valid = show(tv, VAL)
trade_valid.to_csv("data/table_tradeoff_validation.csv")
trade_valid
""")
code("""
base_valid = tv[tv.rule == "current"].set_index("dataset")[["success", "alias", "wrong", "noise", "none"]]
base_valid
""")

md("""
## 8. dev_medium (assigned coverage)
Real SMARTS light curves. Caveat (T20): some SMARTS noise galaxies may carry real periodic signals, so real-data
`wrong` labels may be partly unreliable; gate-firing counts are still valid.
""")
code("""
d = stars[stars.dataset.eq("dev_medium")].copy()
d["cls"] = np.where(d.final_outcome.eq("none"), "none", gl.cand_class(d.final_outcome))
pd.crosstab(d.coverage_class, d.cls, margins=True)
""")
code("""
b = figs.rescue_candidates(stars, cands, "strict")
b = b[b.dataset.eq("dev_medium")]
gt = figs.gate_table(b, by=("coverage_class",))
gt.to_csv("data/table_gates_strict_dev_medium_by_class.csv")
gt
""")
code("""
b["structured"] = pd.cut(b.half_lag_contrast, [-np.inf, 0.02, 0.1, np.inf],
                         labels=["noise-like (<0.02)", "weak (0.02-0.1)", "structured (>=0.1)"])
pd.crosstab(b.coverage_class, b.structured, margins=True)
""")
code("""
dmv = stars[stars.dataset.eq("dev_medium")]
td = gl.tradeoff(dmv, cands[cands.dataset.eq("dev_medium")], rules.RULES)
trade_dm = td.pivot_table(index="rule", values=["d_success", "d_wrong", "d_alias"], aggfunc="sum", sort=False)
trade_dm.to_csv("data/table_tradeoff_dev_medium.csv")
trade_dm
""")
code("""
# dev_medium by coverage class for the proposed rule
r = gl.replay(cands[cands.dataset.eq("dev_medium")], rules.PROPOSED2, stars=dmv)
cur = gl.replay(cands[cands.dataset.eq("dev_medium")], {}, stars=dmv)
r = r.merge(dmv[["dataset", "star_id", "coverage_class"]], on=["dataset", "star_id"])
cur = cur.merge(dmv[["dataset", "star_id", "coverage_class"]], on=["dataset", "star_id"])
(gl.outcome_counts(r, by=("coverage_class",)) - gl.outcome_counts(cur, by=("coverage_class",))).drop(columns="noise")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "ss_openblas", "display_name": "Python (ss_openblas)", "language": "python"}
nbf.write(nb, HERE / "notebook.ipynb")
print("wrote notebook.ipynb")

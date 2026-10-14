"""Write notebook.ipynb for the T15 report (run once; then execute with tools/run_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# T15: every guess_* returns [] instead of raising\n\n"
       "Companion notebook to `README.md`. Regenerates every figure in `figures/` from "
       "`benchmarks/baselines/synth200_s0.json` (before), `runs/t15_after/` (after) and the "
       "wavelet tables in `data/` (written by `make_data.py`)."),
    code('''import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path.cwd()
while not (REPO / "claude_files").exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO / "claude_files"))
REPORT = REPO / "reports" / "2026-10-14_t15-no-raise-all-guess-methods"
FIG, DATA = REPORT / "figures", REPORT / "data"
FIG.mkdir(exist_ok=True)
import evaluation as ev

before = json.loads((REPO / "benchmarks/baselines/synth200_s0.json").read_text())
after = json.loads((REPO / "runs/t15_after/metrics.json").read_text())
per_method = pd.read_csv(REPO / "runs/t15_after/per_method.csv")
w_before = pd.read_csv(DATA / "wavelet_head.csv")
w_after = pd.read_csv(DATA / "wavelet_t15.csv")
spectra = json.loads((DATA / "wavelet_head.spectra.json").read_text())'''),
    md("## Error rate per method, before vs after\n\nBefore = committed baseline (main at 1a076f0 "
       "plus T4); after = this branch. Only `wavelet` raised on this suite."),
    code('''methods = sorted(after["methods"])
tab = pd.DataFrame({
    "before": [before["methods"][m]["overall"]["error_rate"] for m in methods],
    "after": [after["methods"][m]["overall"]["error_rate"] for m in methods],
}, index=methods)
print(tab)
fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(len(methods))
ax.bar(x - 0.2, tab["before"], 0.4, label="before (baseline)", color="#c44e52")
ax.bar(x + 0.2, tab["after"], 0.4, label="after (T15)", color="#4c72b0")
for xi, v in zip(x, tab["after"]):
    ax.text(xi + 0.2, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x, methods, rotation=30, ha="right")
ax.set_ylabel("error rate (fraction of stars that raised)")
ax.set_title("synthetic:n=200,seed=0 -- Stage-1 error rate by method")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "error_rate_by_method.png", dpi=130)'''),
    md("## Wavelet: error rate and recall by period bin"),
    code('''bins = list(before["methods"]["wavelet"]["by_bin"])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, key in zip(axes, ["error_rate", "recall"]):
    b = [before["methods"]["wavelet"]["by_bin"][k][key] for k in bins]
    a = [after["methods"]["wavelet"]["by_bin"][k][key] for k in bins]
    x = np.arange(len(bins))
    ax.bar(x - 0.2, b, 0.4, label="before", color="#c44e52")
    ax.bar(x + 0.2, a, 0.4, label="after", color="#4c72b0")
    ax.set_xticks(x, bins); ax.set_title(f"wavelet {key}"); ax.set_xlabel("true-period bin")
axes[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "wavelet_error_recall_by_bin.png", dpi=130)
print(pd.DataFrame({k: [before["methods"]["wavelet"]["by_bin"][b][k] for b in bins] +
                    [after["methods"]["wavelet"]["by_bin"][b][k] for b in bins]
                    for k in ["error_rate", "recall"]},
                   index=[f"before {b}" for b in bins] + [f"after {b}" for b in bins]))'''),
    md("## Which stars raised in wavelet, and why\n\n`make_data.py` reran `guess_wavelet` exactly as the "
       "scorecard calls it, once with the HEAD code and once with this branch."),
    code('''m = w_before.merge(w_after, on="star_id", suffixes=("_head", "_t15"))
raised = m[m.error_head.notna()]
print(raised[["star_id", "true_period_head", "snr_head", "inclination_head", "error_head", "n_candidates_t15", "error_t15"]].to_string(index=False))
print("\\nstars whose candidate count changed outside the raised set:",
      int(((m.n_candidates_head != m.n_candidates_t15) & m.error_head.isna()).sum()))'''),
    code('''ids = list(spectra)
ncol = 4
nrow = int(np.ceil(len(ids) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow), squeeze=False)
for ax, sid in zip(axes.flat, ids):
    s = spectra[sid]
    ax.loglog(s["periods"], s["gwps"], color="k", lw=1)
    ax.axhline(s["threshold"], color="#c44e52", ls="--", lw=1, label="median + 3 sigma_MAD")
    ax.axvline(s["true_period"], color="#4c72b0", lw=1, label="P_true")
    ax.set_title(f"{sid[-5:]}  P={s['true_period']:.2f} d", fontsize=9)
    ax.set_xlabel("period [d]", fontsize=8)
for ax in list(axes.flat)[len(ids):]:
    ax.axis("off")
axes[0, 0].legend(fontsize=6, frameon=False)
fig.suptitle("GWPS of stars where guess_wavelet raised at HEAD (now returns [])", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "wavelet_raised_gwps.png", dpi=130)'''),
    md("## Final answer: before vs after\n\n`wavelet` is not in the pipeline's pooled methods "
       "(`pairwise_histogram, lombscargle, acf_fft`), so final outcomes should be unchanged."),
    code('''keys = ["correct", "diffrot", "half", "double", "wrong", "none", "success"]
print(pd.DataFrame({"before": [before["final"]["overall"][k] for k in keys],
                    "after": [after["final"]["overall"][k] for k in keys]}, index=keys))
fb = pd.read_csv(DATA / "final_before_after.csv")
print("\\nper-star final outcome, before (rows) vs after (cols):")
print(pd.crosstab(fb.final_outcome_before, fb.final_outcome_after))
rel = (fb.final_period_after - fb.final_period_before).abs() / fb.final_period_before
print("stars with any final-period difference: %d, max relative diff %.2e" % ((rel > 0).sum(), rel.max()))
print("n_pooled differences:", int((fb.n_pooled_before != fb.n_pooled_after).sum()))
print("\\nStage-1 per-method diff vs the pre-T4 v2 run:")
print(pd.read_csv(DATA / "stage1_diff_by_method.csv").to_string(index=False))
print("\\nmedian s/star (wall, per worker) before %.2f, after %.2f" % (before["runtime"]["median_seconds"],
                                                  after["runtime"]["median_seconds"]))'''),
]
nb.metadata["kernelspec"] = {"name": "ss_local", "display_name": "Python (ss_local)", "language": "python"}
nbf.write(nb, __file__.replace("build_notebook.py", "notebook.ipynb"))

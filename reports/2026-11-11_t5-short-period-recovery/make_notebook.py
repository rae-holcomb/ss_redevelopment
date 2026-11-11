"""Generate the T5 report notebook: every README figure, redrawn from data/ via figures.py."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# T5: short-period recovery

Redraws every figure in `README.md` from the committed tables in `data/` (the drawing code is in
`figures.py`). The tables come from the scripts listed in the README's *Reproduce* section.""")
code("""import matplotlib.pyplot as plt
import pandas as pd
import figures as F
pd.set_option('display.width', 200)
d = F.load_phase1()""")

md("""## 1. Diagnosis: where the true sub-day peak sits
Rank (by raw power) of the true-period peak for detectable <1 d stars, and how many spectral peaks compete
for the 10 slots as the baseline grows.""")
code("F.fig_rank(d); plt.show()")
md("""Example stars, both coverage modes: light curve, ACF, ACF-FFT spectrum, Lomb-Scargle. Red = true
period (band = diffrot range or ±10%), grey triangles = old top-10, green = picks the hybrid ranking adds.""")
code("""for s in ['smarts-tess-v1.0-002390', 'smarts-tess-v1.0-001208', 'smarts-tess-v1.0-000718', 'smarts-tess-v1.0-002919']:
    F.fig_diag(s); plt.show()""")

md("## 2. Stage-1 recall before / after the ranking fix")
code("F.fig_stage1_before_after(d); plt.show()")
code("F.fig_stage1_all_bins(d); plt.show()")
code("""t = d.groupby(['mode', 'eval_pbin'])[['old_pool', 'new_pool']].mean()
t.join(d[d.detectable].groupby(['mode', 'eval_pbin'])[['old_pool', 'new_pool']].mean(), rsuffix='_detectable').round(3)""")

md("## 3. Scorecard: final outcomes and fitted-candidate ceiling, before vs after")
code("F.fig_scorecard_bins(); plt.show()")
code("pd.read_csv(F.DATA / 'scorecard_bins.csv').round(3)")
md("Why the newly fitted sub-day success candidates are still declined: gate failures on success candidates.")
code("F.fig_gates(); plt.show()")
code("pd.read_csv(F.DATA / 'gates_success_candidates.csv').round(3)")

md("## 4. Interleaved timing of the ranking change")
code("""b = pd.read_csv(F.DATA / 'bench_ranking.csv')
per = b.groupby(['star_id', 'rank_by'])[['gen_seconds', 'fit_seconds', 'n_tried']].median().unstack()
tot = per['gen_seconds'] + per['fit_seconds']
print(per.median().round(4)); print('median paired total ratio hybrid/power:', round((tot['hybrid'] / tot['power']).median(), 3))""")

md("## 5. Phase 2: optional methods on top of the fixed pool")
code("""s, fp = F.phase2_summary()
F.fig_tradeoff(s); plt.show()
F.fig_fp(s, fp); plt.show()""")
code("s.set_index(['mode', 'method']).T.round(3)")
code("fp.round(4)")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "ss_openblas", "display_name": "Python (ss_openblas)", "language": "python"}
nbf.write(nb, "reports/2026-11-11_t5-short-period-recovery/notebook.ipynb")
print("wrote notebook")

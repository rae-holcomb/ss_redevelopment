"""Build notebook.ipynb for the P43 report (regenerates every README figure from data/ and named runs/)."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# P43: recovering `local_tol`'s decayed-tail losses\nRegenerates the README figures. Inputs: `data/` "
       "(variant metrics, harness summary, local_tol candidate table, lost-star tables, timing) and the scorecard "
       "runs named in `analysis.py` (`runs/p43/sc/`, `benchmarks/baselines/*_v7_tables`); the example panels refit "
       "stars from `data/smarts_dev` with the Stage-1 cache in `runs/p43/stage1_cache/`."),
    code("import sys, pandas as pd\nfrom pathlib import Path\nfrom IPython.display import Image, display\n"
         "HERE = Path.cwd() if (Path.cwd() / 'analysis.py').exists() else Path('reports/2026-11-15_p43-decayed-tail')\n"
         "sys.path.insert(0, str(HERE))\nimport analysis as A\npd.set_option('display.width', 200)"),
    md("## Diagnosis: what fails on local_tol's lost stars\nOne row per correct candidate of T32's 30 lost "
       "star-modes (`diag_lost2.py`). `hs` = sign of each tooth's fitted height, `vs` = tooth within tolerance."),
    code("display(pd.read_csv(A.DATA / 'diag_lost_candidates.csv').head(40))"),
    md("## Scorecard and harness tables"),
    code("display(pd.read_csv(A.DATA / 'variant_metrics.csv').round(3))\n"
         "display(pd.read_csv(A.DATA / 'harness_summary.csv').round(3))"),
    md("## Success / alias / wrong by period bin (scorecards)"),
    code("per_star = A.load_per_star()\nA.fig_by_bin(per_star, [v for v in A.REFS + A.VARIANTS if (v, 'fullyear') in per_star])\n"
         "display(Image(str(A.FIG / 'fig_by_bin.png')))"),
    md("## frac_vertex_tol: all teeth vs positive-height teeth"),
    code("c = pd.read_csv(A.DATA / 'cands_local_tol.csv.gz', dtype={'star_id': str})\nA.fig_fvt(c)\n"
         "display(Image(str(A.FIG / 'fig_fvt_all_vs_positive.png')))\ndisplay(pd.read_csv(A.DATA / 'fvt_stats.csv').round(3))"),
    md("## local_tol's lost stars (vs _v7): which each variant recovers"),
    code("for v in A.VARIANTS:\n    lt = A.lost_table(per_star, c, v); A.fig_lost_scatter(lt, v)\n"
         "    display(Image(str(A.FIG / f'fig_lost_scatter_{v}.png')))"),
    md("## Example stars (refits; needs data/smarts_dev and the Stage-1 cache)"),
    code("import numpy as np\nfor v in ('local_tol_tail', 'local_tol_norej_tail_snr0'):\n"
         "    ch = A.changed_vs_tol(per_star, v)\n"
         "    A.fig_examples(ch, v, ('gained',), f'fig_examples_{v}_recovered.png', n=4)\n"
         "    bad = ch[(ch.change == 'lost') | ((ch.change == 'other') & (ch.final_outcome_var == 'wrong'))].copy()\n"
         "    bad['change'] = np.where(bad.change == 'lost', 'lost', 'newly wrong')\n"
         "    A.fig_examples(bad, v, ('lost', 'newly wrong'), f'fig_examples_{v}_lost_or_wrong.png', n=2 if (bad.change == 'lost').any() else 4)\n"
         "    for k in ('recovered', 'lost_or_wrong'):\n        display(Image(str(A.FIG / f'fig_examples_{v}_{k}.png')))"),
    md("## drift_a_frac (the new ranker feature)"),
    code("A.fig_drift(c)\ndisplay(Image(str(A.FIG / 'fig_drift_a_frac.png')))\n"
         "display(pd.read_csv(A.DATA / 'drift_a_frac_summary.csv'))"),
    md("## Runtime"),
    code("tim = pd.read_csv(A.DATA / 'timing_summary.csv')\nA.fig_runtime(tim)\ndisplay(Image(str(A.FIG / 'fig_runtime.png')))\n"
         "display(tim.round(4))"),
]
nbf.write(nb, HERE / "notebook.ipynb")

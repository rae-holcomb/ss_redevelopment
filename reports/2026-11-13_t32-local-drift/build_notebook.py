"""Build notebook.ipynb for the T32 report (regenerates every README figure from data/ and named runs/)."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# T32: drift-robust local comb solver\nRegenerates the README figures. Inputs: `data/` (step-1 peaks, "
       "variant metrics, candidate subset from the harness) and the scorecard runs named in `analysis.py` "
       "(`runs/t32/sc/`, `benchmarks/baselines/*_v6w_tables`, `*_v7_tables`); the example panels read the FITS "
       "files in `data/smarts_dev`."),
    code("import sys, pandas as pd\nfrom pathlib import Path\nfrom IPython.display import Image, display\n"
         "HERE = Path.cwd() if (Path.cwd() / 'analysis.py').exists() else Path('reports/2026-11-13_t32-local-drift')\n"
         "sys.path.insert(0, str(HERE))\nimport analysis as A"),
    md("## Step 1: residual vs peak index n\nGains drift linearly out of the frozen n·P0 windows; none→wrong stars "
       "are offset from n = 0 and scatter randomly."),
    code("%run {HERE}/fig_step1.py\ndisplay(Image(str(A.FIG / 'fig_step1_residual_vs_n.png')))".replace("{HERE}", "$HERE")),
    md("## Variant table and frontier\nRates per variant (dev_medium, both modes), serial timing on T12's 15 stars."),
    code("tab = pd.read_csv(A.DATA / 'variant_metrics.csv')\n"
         "display(tab[['variant','mode','success','wrong','precision','gained','lost','none_to_wrong','s_production']].round(3))\n"
         "A.fig_frontier(tab)\ndisplay(Image(str(A.FIG / 'fig_frontier.png')))"),
    md("## Success and wrong by period bin"),
    code("per_star = A.load_per_star()\nA.fig_by_bin(per_star, [v for v in A.REFS + A.NEW if (v, 'fullyear') in per_star])\n"
         "display(Image(str(A.FIG / 'fig_by_bin.png')))"),
    md("## Gate statistics (candidate level), scatter threshold curve, drift coefficient"),
    code("c = pd.read_csv(A.DATA / 'cands_subset.csv.gz')\nA.fig_gate_stats(c); A.fig_scatter_curve(c); A.fig_drift_a(c)\n"
         "for f in ['fig_gate_stats', 'fig_scatter_threshold_curve', 'fig_drift_a']:\n    display(Image(str(A.FIG / f'{f}.png')))\n"
         "display(pd.read_csv(A.DATA / 'gate_stats.csv').round(3)); display(pd.read_csv(A.DATA / 'drift_a_summary.csv'))"),
    md("## Gained and lost stars: `local_tol` vs `local_screen2`"),
    code("ch = A.changed_stars(per_star, 'local_tol')\ndisplay(ch.groupby(['mode','change']).size())\n"
         "A.fig_examples(ch, 'local_tol', 'gained'); A.fig_examples(ch, 'local_tol', 'lost')\n"
         "display(Image(str(A.FIG / 'fig_examples_local_tol_gained.png'))); display(Image(str(A.FIG / 'fig_examples_local_tol_lost.png')))"),
]
nbf.write(nb, HERE / "notebook.ipynb")

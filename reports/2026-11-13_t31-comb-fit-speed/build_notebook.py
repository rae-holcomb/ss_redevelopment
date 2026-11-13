#!/usr/bin/env python3
"""Build notebook.ipynb for the T31 report: regenerates every README figure from runs/t31 + data/ via analysis.py."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# T31: wavelet determinism + result-changing comb-fit / CWT speed-ups\n\n"
       "Regenerates every figure in `README.md` from `runs/t31/` (scorecard outputs and `profile_pipeline.py` "
       "timings copied from the overnight session; see README *Reproduce* to rebuild them) and `data/`. "
       "All plotting code lives in `analysis.py`."),
    code("import sys\nfrom pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n"
         "HERE = Path.cwd()\nROOT = HERE.parents[1]\nsys.path.insert(0, str(HERE))\n"
         "import analysis as A\nSCR = ROOT / 'runs' / 't31'\nFIG = HERE / 'figures'\n"
         "met = A.collect_metrics(SCR)\ntim = A.timing_summary(A.collect_profiles(SCR))\n"
         "met[['run','mode','success','correct','diffrot','wrong','none','wavelet_recall','wavelet_top1_success']].round(3)"),
    md("## Part A: identity checks\nGreen rows are bit-identical; orange rows differ only in the last digits of "
       "`methods.wavelet.top1`."),
    code("ident = pd.read_csv(SCR / 'identity_partA.csv')\nA.fig_identity(ident, FIG / 'fig_identity_partA.png')\n"
         "display(Image(FIG / 'fig_identity_partA.png'))"),
    md("## Part B: timing (serial, 15-star set, rounds 2-3)\nMean seconds per star per stage; `main` = pre-T31 code, "
       "`lsq` = Part A code with the default comb fit."),
    code("tim[['comb','cwt','mode','fit','guess_wavelet','total']].round(3)"),
    md("### Accuracy vs time frontier\nTop: success; bottom: wrong rate. The red star / dotted line is current main."),
    code("A.fig_frontier(met, tim, FIG / 'fig_frontier.png')\ndisplay(Image(FIG / 'fig_frontier.png'))"),
    md("Same, against the comb-fit stage time only (the part the comb variants change)."),
    code("A.fig_fit_frontier(met, tim, FIG / 'fig_frontier_fit_stage.png')\n"
         "display(Image(FIG / 'fig_frontier_fit_stage.png'))"),
    md("### Success by period bin: main vs local_screen2 (recommended), local, lsq_tol3"),
    code("A.fig_by_bin(met, ['v_local_screen2', 'v_local__dec', 'v_lsq_tol3__n100'], FIG / 'fig_success_by_bin.png')\n"
         "display(Image(FIG / 'fig_success_by_bin.png'))"),
    md("### Per-stage time: pre-T31 main, Part A, local_screen2"),
    code("A.fig_stage_breakdown(tim, 'local_screen2', FIG / 'fig_stage_breakdown.png')\n"
         "display(Image(FIG / 'fig_stage_breakdown.png'))"),
    md("### The screen threshold\nGates failed under the local fit, for candidates the lsq fit passes (green) or "
       "fails (grey). local_screen2 refits everything left of the solid line with lsq."),
    code("rec = A.fig_screen(SCR, FIG / 'fig_screen_threshold.png')\ndisplay(rec)\n"
         "display(Image(FIG / 'fig_screen_threshold.png'))"),
    md("### Why `local` behaves differently: gate failure rates"),
    code("g = A.fig_gate_rates(SCR, FIG / 'fig_gate_fail_rates.png')\n"
         "display(g.pivot_table(index=['mode','gate'], columns='variant', values='fail_rate').round(3))\n"
         "display(Image(FIG / 'fig_gate_fail_rates.png'))"),
    md("### CWT variants: wavelet recall vs wavelet-stage time"),
    code("A.fig_cwt(met, tim, FIG / 'fig_cwt_variants.png')\ndisplay(Image(FIG / 'fig_cwt_variants.png'))"),
    md("## Example light curves\nStars `local` loses / gains vs main, and the one star whose final period moves under "
       "local_screen2 (needs `data/smarts_dev`)."),
    code("for run, kinds in (('v_local__dec', ('lost', 'gained')), ('v_local_screen2', ('lost', 'gained', 'period_moved'))):\n"
         "    gl = pd.concat([A.gain_loss(SCR, run, m) for m in ('fullyear', 'assigned')])\n"
         "    lab = A.run_name_to_variant(run)[0]\n"
         "    for k in kinds:\n"
         "        n = A.fig_examples(gl, lab, k, FIG / f'fig_examples_{lab}_{k}.png', n=5)\n"
         "        print(run, k, n)\n"
         "        if n:\n"
         "            display(Image(FIG / f'fig_examples_{lab}_{k}.png'))"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "ss_openblas", "display_name": "Python (ss_openblas)",
                                                               "language": "python"}})
nbf.write(nb, HERE / "notebook.ipynb")
print("wrote", HERE / "notebook.ipynb")

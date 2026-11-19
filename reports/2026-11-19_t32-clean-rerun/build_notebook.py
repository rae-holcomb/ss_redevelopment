"""Build notebook.ipynb: re-run each figure script from saved data / runs/t32_clean and show the figures."""
import json
cells = []
def md(s): cells.append(dict(cell_type="markdown", metadata={}, source=s))
def code(s): cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s))
md("# T32 clean rerun: robust local comb solver\n\nCompanion to `README.md`. Each section re-runs one figure script "
   "(from `data/` or the named `runs/t32_clean/` outputs) and shows the figure. Heavy steps (Stage-1 caches, Stage-2 "
   "screen, scorecards, profiling) are scripts; see README \"Reproduce\".")
code("import subprocess, sys, os\nfrom pathlib import Path\nfrom IPython.display import Image, display\n"
     "HERE = Path.cwd()\nPY = sys.executable\n"
     "def run(*args):\n    r = subprocess.run([PY, *args], cwd=HERE, capture_output=True, text=True)\n"
     "    print(r.stdout[-3000:], r.stderr[-2000:])\n")
md("## 1. Free vertex residual vs n (step 1)\nFrom `data/vertex_residuals.csv` (made by `diag_vertex.py`). Gains: early teeth "
   "on the line, late teeth walk out of the frozen seed windows. none→wrong: low acceptance and scatter from n = 1.")
code("run('fig_vertex.py'); display(Image('figures/fig_vertex_residual_vs_n.png'))")
md("## 2. Screen summary and outcomes by period bin\nStage-2 harness = scorecard per star (checked for local, local_screen2, "
   "local_seq, local_seq_trim, local_trim in both modes).")
code("run('summarize.py')\nimport pandas as pd\npd.read_csv('data/screen_summary.csv')")
code("run('fig_bins.py'); display(Image('figures/fig_by_bin_fy.png')); display(Image('figures/fig_by_bin_as.png'))")
md("## 3. Leader's lost stars: which gate fails")
code("run('fig_gates.py', 'local_seq_trim'); display(Image('figures/fig_leader_lost_gates.png'))")
md("## 4. Speed vs success, runtime bars")
code("run('fig_speed.py'); display(Image('figures/fig_speed_vs_success.png')); display(Image('figures/fig_runtime_bars.png'))")
md("## 5. Example stars (LC + ACF with windows, fitted teeth and vertices)\nNeeds the Stage-1 caches in `runs/t32_clean/cache_{fy,as}`.")
code("P = 'smarts-tess-v1.0-'\nsets = {\n"
     " 'fig_examples_gained.png': ('Gained (v7 declined/wrong, local_seq_trim correct)', ['004105:fy','046125:fy','107733:fy','202553:fy','012015:as']),\n"
     " 'fig_examples_lost.png': (\"Lost (v7 correct, local_seq_trim declines; blue = its candidate with v7's seed)\", ['026704:fy','130951:fy','215341:fy','236661:fy','028773:as']),\n"
     " 'fig_examples_newly_wrong.png': ('Newly wrong (v7 declined)', ['003031:fy','154599:fy','199874:as','434994:as']),\n"
     " 'fig_examples_avoided_wrong.png': ('Avoided wrongs (v7 wrong, local_seq_trim declines)', ['010041:fy','285035:fy','810997:fy','123698:as'])}\n"
     "for out, (title, stars) in sets.items():\n    run('fig_examples.py', 'local_seq_trim', title, out, *[P + s for s in stars])\n"
     "    display(Image('figures/' + out))")
nb = dict(cells=cells, metadata=dict(kernelspec=dict(name="ss_openblas", display_name="Python (ss_openblas)", language="python")),
          nbformat=4, nbformat_minor=5)
json.dump(nb, open("notebook.ipynb", "w"), indent=1)

#!/usr/bin/env python3
"""build_notebook.py -- (re)write notebook.ipynb for the T18 report. Execute it with
tools/run_notebook.py afterwards. Run from this folder."""
import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = [
    md("# T18: arbitration and reliability gates\n\n"
       "Regenerates every figure in `README.md` from `data/` (small tables extracted from the "
       "gitignored `runs/t18_cache_*` diagnostic caches by `make_data.py`) and from the scorecard / "
       "pb_scan output directories `runs/t18_{before,after}_*`. The example figure re-simulates five "
       "frozen example light curves (`data/example_params.json`) and runs the pipeline with the T18 gates "
       "off and on (~1 min)."),
    code("import json, sys\nfrom pathlib import Path\nimport pandas as pd\n"
         "sys.path.insert(0, str(Path('../../claude_files').resolve()))\n"
         "import determinism; determinism.pin_blas_threads()\n"
         "import matplotlib; matplotlib.rcParams['figure.max_open_warning'] = 0\n"
         "import figs\npd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 70)\n"
         "RUNS = Path('../../runs')"),
    md("## 1. Diagnosis: why seed-0 stars were lost (pre-T18)\n"
       "Categories: no fitted success candidate at all (a Stage 1 problem); the best success candidate "
       "failed one of the four comb-shape gates; or it passed every gate but another candidate was selected."),
    code("lost = pd.read_csv('data/lost_s0.csv')\nfigs.fig_lost_categories(lost);\n"
         "lost.groupby(['category', 'outcome']).size()"),
    md("Lost on selection: in every case the winner was a ~0.2-0.5 d comb with mean half-lag contrast ~0."),
    code("cols = ['star_id', 'P_true', 'snr', 'outcome', 'succ_P', 'succ_acf_half_lag_contrast', 'sel_P', "
         "'sel_acf_half_lag_contrast', 'sel_redchi', 'succ_redchi']\n"
         "lost[lost.category == 'lost_on_selection'][cols].round(4)"),
    md("Noise false positives on the pb_scan dev seeds (offset 500): selected candidate's gate values and features."),
    code("pd.read_csv('data/fp_pbdev.csv').round(3)"),
    md("## 2. The half-lag consistency gate\n"
       "Each point is a pre-T18 *selected* answer. x = the smallest of the terms h(kP) - h((k-1/2)P); "
       "y = their mean (`acf_half_lag_contrast`). The new gate rejects the shaded quadrant (some term <= 0 "
       "and mean < 0.1). Left: tuning data; right: held-out validation."),
    code("sel = pd.read_csv('data/selected_base.csv')\nfigs.fig_contrast_gate(sel);"),
    md("## 3. The few-peak ACF-height guard"),
    code("few = pd.read_csv('data/few_peak_cands.csv')\nfigs.fig_few_peak_guard(few);\n"
         "few.assign(group=few.kind.where(few.kind == 'noise', few.label)).groupby('group').acf_height_at_P.describe().round(3)"),
    md("## 4. Gate ablation (offline replay of the arbitration on the caches)\n"
       "`arbitration.replay` re-applies a gating / selection rule to the cached candidate fits; with the "
       "pre-T18 rule it reproduces every cached pipeline outcome (0 mismatches on all 5 caches), and with the "
       "recommended rule it reproduces the real post-change scorecards star by star. Thresholds were chosen on "
       "the dev rows only."),
    code("abl = pd.read_csv('data/ablation.csv')\n"
         "rules = ['base (pre-T18)', '+ half-lag gate only', '+ few-peak guard only', "
         "'RECOMMENDED: half-lag gate + few-peak guard', 'rec, height-SNR gate 0.7', 'rec, require >= 2 methods', "
         "'rec, factor-2 ratio 0.8', 'rec, factor-2 guard off', 'raw contrast gate hlc >= 0.02', "
         "'B: half-lag PREFERRED, else base gates', 'C: base gates, select max contrast', "
         "'D: half-lag preferred, else base gates + contrast > 0']\n"
         "short = {'RECOMMENDED: half-lag gate + few-peak guard': 'RECOMMENDED (new default)', "
         "'D: half-lag preferred, else base gates + contrast > 0': 'D: preferred, fallback contrast>0', "
         "'B: half-lag PREFERRED, else base gates': 'B: half-lag preferred', "
         "'C: base gates, select max contrast': 'C: select max contrast'}\n"
         "figs.fig_ablation(abl, rules, short);"),
    code("view = abl.pivot_table(index='rule', columns='dataset', values=['success', 'wrong', 'alias', 'none', 'fp_red'])\n"
         "view.loc[[r for r in abl.rule.unique()]].round(3)"),
    md("## 5. Before / after on the real benchmarks"),
    code("before = {f'synthetic:n=200,seed={s}': json.load(open(RUNS / f't18_before_s{s}/metrics.json')) for s in (0, 1)}\n"
         "after = {f'synthetic:n=200,seed={s}': json.load(open(RUNS / f't18_after_s{s}/metrics.json')) for s in (0, 1)}\n"
         "figs.fig_bins(before, after);\n"
         "pd.DataFrame({(k, w): d[k]['final']['overall'] for w, d in (('before', before), ('after', after)) for k in d}).round(3)"),
    code("rows = []\nfor t in ('eval', 'dev'):\n    for w in ('before', 'after'):\n"
         "        s = json.load(open(RUNS / f't18_{w}_pb_{t}/summary.json'))\n"
         "        rows.append(dict(set=t, run=w, **s['overall'], fp_red=s['noise_false_positive']['red'], "
         "fp_white=s['noise_false_positive']['white']))\npd.DataFrame(rows).round(3)"),
    md("pb_scan correct fraction by P/B (pooled over 1/2/13 sectors), evaluation seeds."),
    code("def pb(w):\n    r = pd.read_csv(RUNS / f't18_{w}_pb_eval/rows.csv'); s = r[r.kind == 'signal']\n"
         "    return s.groupby('P_over_B').outcome.apply(lambda o: (o == 'correct').mean())\n"
         "pd.DataFrame({'before': pb('before'), 'after': pb('after')}).round(3).T"),
    md("## 6. Example failure modes (frozen light curves; gates off = pre-T18, gates on = T18)\n"
       "Orange: the pre-T18 pick's fitted comb. Blue: the best success candidate (or another candidate). "
       "Right column: zoom on the pre-T18 pick, with its kP (solid) and (k-1/2)P (dotted) lags; the gate "
       "asks whether the ACF is higher at every solid line than at the dotted line before it."),
    code("fig, ex = figs.fig_examples()\nex"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "ss_openblas", "display_name": "Python (ss_openblas)", "language": "python"}})
nbf.write(nb, "notebook.ipynb")

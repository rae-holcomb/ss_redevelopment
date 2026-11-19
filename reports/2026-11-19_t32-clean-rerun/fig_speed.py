"""Runtime (profile_pipeline, 15 stars x 3 repeats, full-year, serial) and speed-vs-success figures."""
import json
from pathlib import Path
import pandas as pd
from style import plt, C, INK2
ROOT = Path(__file__).resolve().parents[2]
V = ["lsq", "local_screen2", "local", "local_trim", "local_seq", "local_seq_trim"]
PROD = ["load", "acf_long", "acf_short", "guess_acf_fft", "guess_lombscargle", "guess_pairwise_histogram"]
rows = []
for v in V:
    d = pd.read_csv(ROOT / f"runs/t32_clean/prof/{v}.csv")
    d = d[d.repeat >= 1]  # round 0 = warm-up
    m = d.mean(numeric_only=True)
    rows.append(dict(variant=v, fit=m.fit, features=m.features, production=m[PROD].sum() + m.fit,
                     scorecard=m.total - m.features, total_with_features=m.total,
                     other=m.total - m.features - m.fit))
t = pd.DataFrame(rows).round(4)
t.to_csv("data/timing_summary.csv", index=False)
print(t.to_string(index=False))
# official scorecard metrics (lsq = _v7 per star: local_screen2 reproduces it)
met = []
for v in V:
    for mode, base in [("fy", "fullyear"), ("as", "assigned")]:
        p = ROOT / (f"runs/t32_clean/sc/{v}_{mode}/metrics.json" if v != "lsq"
                    else f"benchmarks/baselines/dev_medium_clean_{base}_v7.json")
        f = json.loads(p.read_text())["final"]["overall"]
        met.append(dict(variant=v, mode=mode, success=f["success"], wrong=f["wrong"], alias=f["alias"]))
met = pd.DataFrame(met)
met.to_csv("data/official_metrics.csv", index=False)
x = t.set_index("variant")
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
for j, (mode, lab) in enumerate([("fy", "full-year"), ("as", "assigned")]):
    mm = met[met["mode"] == mode].set_index("variant")
    for i, k in enumerate(["success", "wrong"]):
        a = ax[i, j]
        for c, v in enumerate(V):
            a.plot(x.loc[v, "production"], mm.loc[v, k], "o", ms=9, color=C[c], mec="#fcfcfb", mew=2, label=v)
            a.annotate(v, (x.loc[v, "production"], mm.loc[v, k]), xytext=(8, -10 if v in ("local_seq", "local") else 4),
                       textcoords="offset points",
                       fontsize=7, color=INK2)
        a.set_ylabel(f"{k} rate ({lab})")
        if i == 1:
            a.set_xlabel("production-path s/star (full-year, serial)")

fig.suptitle("Speed vs accuracy, clean dev_medium (official scorecards; lsq = _v7 answers); production path = load + ACFs + 3 pooled methods + fit", fontsize=9)
fig.savefig("figures/fig_speed_vs_success.png")
fig, a = plt.subplots(figsize=(9, 3.4))
y = range(len(V))
a.barh(y, x.loc[V, "fit"], color=C[0], label="comb fit (Stage 2)", height=0.6)
a.barh(y, x.loc[V, "production"] - x.loc[V, "fit"], left=x.loc[V, "fit"], color=C[2], label="load + ACFs + pooled methods",
       height=0.6)
a.barh(y, x.loc[V, "scorecard"] - x.loc[V, "production"], left=x.loc[V, "production"], color="#c3c2b7",
       label="rest of scorecard path (wavelet, other methods)", height=0.6)
a.barh(y, x.loc[V, "features"], left=x.loc[V, "scorecard"], color=C[3], label="ml_features", height=0.6)
for i, v in enumerate(V):
    a.text(x.loc[v, "total_with_features"] + 0.02, i, f"fit {x.loc[v, 'fit']:.3f} s, production {x.loc[v, 'production']:.3f} s",
           va="center", fontsize=7, color=INK2)
a.set_yticks(list(y), V)
a.invert_yaxis()
a.set_xlabel("mean s/star (15 profile stars, full-year, repeats 2-3, serial, nice 15)")
a.legend(fontsize=7, loc="lower right")
a.set_xlim(0, x.total_with_features.max() * 1.45)
fig.savefig("figures/fig_runtime_bars.png")

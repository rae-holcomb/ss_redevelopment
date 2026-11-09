"""Follow-up to c07: period-binned thresholds (quiet sources' 95th percentile of min-half S/N within the same
log-period bin), and what the high-S/N quiet short-period peaks are."""
import sys, numpy as np, pandas as pd
r = pd.read_csv(sys.argv[1])
r["P"] = r.P_ref_d.fillna(r.ls_P1_d)
r["bin"] = pd.cut(r.P, [0.1, 0.5, 1.5, 5, 10, 20, 61])
q = r[~r.periodic]
thr = q.groupby("bin").min_snr.quantile(0.95)
print("quiet n per bin", q.groupby("bin").size().to_dict()); print("thr", thr.round(1).to_dict())
r["coh"] = r.min_snr > r.bin.map(thr).astype(float)
p = r[r.periodic]
print("periodic per bin: n, frac coherent (> quiet 95th in same bin):")
print(p.groupby("bin").agg(n=("coh", "size"), coh=("coh", "mean"), both_top=("both_top", "mean")).round(2).to_string())
print(p.groupby("classification").agg(n=("coh", "size"), coh=("coh", "mean")).round(2).to_string())
print("pipeline-only:", p[~p.ls_periodic].coh.mean().round(2), " ls_periodic:", p[p.ls_periodic].coh.mean().round(2))
print("P>=5 periodic coherent:", int(p[p.P >= 5].coh.sum()), "of", int((p.P >= 5).sum()))
qs = q[(q.P < 5) & (q.min_snr > 20)]
print("quiet, P<5, min-half S/N>20:", len(qs), "periods:", np.round(np.sort(qs.P.values), 3)[:40])

"""Poisson-binomial p-values of observed FFI confirmations against each source's own chance-match rate (from c06)."""
import sys, numpy as np, pandas as pd
r = pd.read_csv(sys.argv[1])
def pb_sf(ps, k):
    """P(X >= k) for a sum of independent Bernoulli(ps)."""
    d = np.zeros(len(ps) + 1); d[0] = 1
    for p in ps: d[1:] = d[1:] * (1 - p) + d[:-1] * p; d[0] *= 1 - p
    return d[k:].sum()
for (ctrl, cls), g in r.groupby(["control", "classification"]):
    for col, nul in (("real_m3", "null_m3"), ("real_m1", "null_m1")):
        k = int(g[col].sum()); print(f"{cls:34s} {col}: obs {k}/{len(g)} exp {g[nul].sum():.1f}  p(>=obs) = {pb_sf(g[nul].values, k):.2g}")

"""Spatial families: periodic noise sources with another periodic source within R arcmin at the same period
(within 2%, or P/2, 2P). Shared signals across separate apertures are direct evidence of a sky (contaminant)
origin, and they make 'sources' non-independent trials in the class statistics."""
import numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
S = pd.read_csv(f"{A}/docs/smarts_release_note/smarts_noise_sources.csv")
p = S[S.periodic].reset_index(drop=True)
ra, de, P = np.radians(p.ra_deg.values), np.radians(p.dec_deg.values), p.P_ref_d.values
sep = np.degrees(np.arccos(np.clip(np.sin(de[:, None]) * np.sin(de) + np.cos(de[:, None]) * np.cos(de) * np.cos(ra[:, None] - ra), -1, 1))) * 60
r = P[:, None] / P
same = (np.abs(r - 1) < 0.02) | (np.abs(r - 2) < 0.04) | (np.abs(r - 0.5) < 0.01)
for R in (2, 5, 10):
    adj = (sep < R) & same & ~np.eye(len(p), dtype=bool)
    n, lab = connected_components(adj, directed=False)
    sizes = pd.Series(lab).value_counts()
    fam = sizes[sizes > 1]
    print(f"R={R}': {int(adj.any(1).sum())} periodic sources in {len(fam)} families (sizes {fam.tolist()[:12]}); independent units = {n}")
    if R == 5:
        p["fam"] = lab
        p["in_fam"] = adj.any(1)
        print(p.groupby("classification").in_fam.agg(["size", "sum"]).to_string())
        big = p[p.fam.isin(fam.index)].groupby("fam").agg(n=("noise_tic", "size"), P=("P_ref_d", "median"),
              classes=("classification", lambda s: ";".join(sorted(set(s)))))
        print(big.sort_values("n", ascending=False).head(8).round(3).to_string())
# chance: same test with periods shuffled among sources
rng = np.random.default_rng(0); null = []
for _ in range(200):
    Ps = rng.permutation(P); rr = Ps[:, None] / Ps
    ss = (np.abs(rr - 1) < 0.02) | (np.abs(rr - 2) < 0.04) | (np.abs(rr - 0.5) < 0.01)
    null.append(int(((sep < 5) & ss & ~np.eye(len(p), dtype=bool)).any(1).sum()))
print("R=5' sources in families under period shuffle: mean", np.mean(null), "max", np.max(null))

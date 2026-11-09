"""Recompute Section 2 (corrupted segments) from the A+B stacks, and test sensitivity to the undocumented 2-d merge gap."""
import sys
import numpy as np, pandas as pd
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
sys.path.insert(0, f"{A}/tools/audit")
import smarts_release_products as rp
D = "/Volumes/Data_Vault/smarts_data/extracted/galaxy_stack"
S = np.load(f"{D}/stacks.npy", mmap_mode="r")
idx = pd.read_csv(f"{D}/stack_index.csv")
src = pd.read_csv(f"{A}/docs/smarts_release_note/smarts_noise_sources.csv").set_index("noise_tic")
rec = []
for i, tic in enumerate(idx.tic):
    g = np.asarray(S[i, 2], float)
    segs = rp.corrupted_segments(g)
    if not segs:
        continue
    dur = [(b - a + 1) / 48 for a, b in segs]
    inseg = np.zeros(g.size, bool)
    for a, b in segs: inseg[a:b + 1] = True
    rec.append(dict(tic=tic, nseg=len(segs), durs=dur, start=[a / 48 for a, _ in segs],
                    maxdev=np.nanmax(np.abs(g[inseg] - 1)), neg=bool((g[inseg] < 0).any()),
                    neg_any=bool((g < 0).any()), out_dev=np.nanmax(np.abs(g[~inseg] - 1)) if (~inseg).any() else np.nan,
                    frac_in=inseg.mean()))
r = pd.DataFrame(rec)
print("sources", len(r), "files", src.loc[r.tic, "n_files"].sum(), "flag agrees", set(r.tic) == set(src.index[src.has_corrupted_segment]))
alld = np.concatenate(r.durs.values)
print("segments", len(alld), "median dur (per segment)", np.median(alld), "per-source median of durations", r.durs.map(np.median).median())
print("nseg counts", r.nseg.value_counts().to_dict())
print("maxdev>1:", (r.maxdev > 1).sum(), "max", r.maxdev.max(), "neg in seg:", r.neg.sum(), "neg anywhere:", r.neg_any.sum())
print("max |g-1| outside segments, median/max over flagged:", r.out_dev.median(), r.out_dev.max())
st = np.concatenate(r.start.values)
print("segment start times (d) hist (30-d bins):", np.histogram(st, bins=np.arange(0, 391, 30))[0])
# sources NOT flagged but with many deviant cadences
cnt = []
for i, tic in enumerate(idx.tic):
    g = np.asarray(S[i, 2], float)
    cnt.append((tic, int((np.abs(g - 1) > 0.1).sum()), int((np.abs(g - 1) > 1).sum())))
c = pd.DataFrame(cnt, columns=["tic", "n01", "n1"]).set_index("tic")
nf = c[~c.index.isin(r.tic)]
print("unflagged sources with >=48 cadences |g-1|>0.1:", (nf.n01 >= 48).sum(), "with any |g-1|>1:", (nf.n1 > 0).sum(), "max n01 unflagged", nf.n01.max())
# sensitivity to merge gap and min frac
for gap in (0.5, 1, 2, 5):
    rp.CORRUPT_MERGE_D = gap
    n = sum(bool(rp.corrupted_segments(np.asarray(S[i, 2], float))) for i in range(len(idx)))
    print("merge gap", gap, "d ->", n, "sources")

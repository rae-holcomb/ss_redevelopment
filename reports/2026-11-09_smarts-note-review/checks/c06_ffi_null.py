"""Per-source chance-match null for the independent-FFI test: re-run ffi_lc_compare's exact test on each
source's own FFI light curves at random trial periods (drawn from the testable periodic P_ref values,
excluding trials within 5% of the source's P_ref, P_ref/2, 2P_ref). Also stricter variants of the
real test (top-1 peak; P only; >=2 sectors). Output: ffi_null.csv (per source)."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
sys.path.insert(0, f"{A}/tools/audit")
import ffi_lc_compare as fc
FD = "/Volumes/Data_Vault/smarts_data/extracted/ffi_check"
T = f"{A}/reports/2026-10-21_t20-noise-sources-phase2/data"
N_TRIAL = 40

def run(args):
    tic, P_ref, paths, trials = args
    lcs = [pd.read_csv(f"{FD}/lc/{p.split('/')[-1]}") for p in paths]
    def test(P):
        w = fc.detrend_window(P); T_, Y_ = [], []; secm = []
        for lc in lcs:
            t, y = fc.clean(lc.time.values, lc.flux.values, w)
            a = fc.analyse(t, y, P, w); secm.append(a["match"] != ""); T_.append(t); Y_.append(y)
        a = fc.analyse(np.concatenate(T_), np.concatenate(Y_), P, w)
        top1 = a["peaks"][:1]
        return dict(m3=a["match"] != "", m1=fc.match(top1, P) != "", mP=any(abs(x - P) / P < fc.MATCH_TOL for x in a["peaks"]),
                    nsec=sum(secm))
    real = test(P_ref)
    null = [test(P) for P in trials]
    return dict(tic=tic, **{f"real_{k}": v for k, v in real.items()},
                null_m3=np.mean([x["m3"] for x in null]), null_m1=np.mean([x["m1"] for x in null]),
                null_mP=np.mean([x["mP"] for x in null]), null_m3_2sec=np.mean([x["m3"] and x["nsec"] >= 2 for x in null]))

if __name__ == "__main__":
    f = pd.read_csv(f"{T}/ffi_compare_all.csv")
    idx = pd.read_csv(f"{FD}/ffi_index_ffi_selection_all.csv"); idx = idx[idx.status == "ok"]
    pool = f.loc[~f.control & f.testable, "P_ref"].values
    rng = np.random.default_rng(7)
    jobs = []
    for r in f[f.testable].itertuples():
        bad = lambda P: any(abs(P / q - 1) < 0.05 for q in (r.P_ref, r.P_ref / 2, 2 * r.P_ref))
        tr = [P for P in rng.choice(pool, 4 * N_TRIAL) if not bad(P)][:N_TRIAL]
        jobs.append((r.tic, r.P_ref, list(idx[idx.tic == r.tic].path), tr))
    with ProcessPoolExecutor(4) as ex:
        res = pd.DataFrame(list(ex.map(run, jobs)))
    res = res.merge(f[["tic", "classification", "control", "confirmed", "authors", "P_ref"]], on="tic")
    res.to_csv(sys.argv[1], index=False)
    print("real_m3 reproduces 'confirmed':", (res.real_m3 == res.confirmed).mean())
    g = res.groupby(["control", "classification"])
    print(pd.DataFrame(dict(n=g.size(), obs=g.real_m3.sum(), exp_null=g.null_m3.sum(), obs_top1=g.real_m1.sum(), exp_top1=g.null_m1.sum(),
                            obs_Ponly=g.real_mP.sum(), exp_Ponly=g.null_mP.sum(),
                            obs_2sec=g.apply(lambda d: (d.real_m3 & (d.real_nsec >= 2)).sum()), exp_2sec=g.null_m3_2sec.sum())).round(1).to_string())
    print("null rate by author:", res.groupby("authors").null_m3.mean().round(3).to_dict())
    res["Pbin"] = pd.cut(res.P_ref, [0, 0.5, 1, 2, 5])
    print("null rate by own P_ref bin (trial P differ):", res.groupby("Pbin").null_m3.mean().round(3).to_dict())

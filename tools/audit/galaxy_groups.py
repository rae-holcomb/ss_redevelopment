"""T20 phase 1: cluster the galaxy_pass noise fingerprints into noise-galaxy IDs (scales to 1M files).

Input: the per-bundle outputs of ``tools/audit/galaxy_pass.py``. Files that
share a noise light curve have (near-)identical fingerprints, so this is
template ("leader") clustering, not all-pairs. Two correlation levels matter
(measured on bundle 001, see the T20 phase-1 report):

- **identical noise light curve**: member-to-template r >= 0.99 for 95% of
  files (median 0.9997); a file with a strong injected signal can drop to
  r ~ 0.6, but it is then *alone* at that level;
- **sibling light curves**: some distinct noise light curves correlate with
  each other at 0.3-0.99 (families of 2-5, often bright galaxies; possibly
  one galaxy reduced twice or overlapping apertures; projecting out up to 30
  common modes does not remove it). Each sibling is internally identical
  (r ~ 1), so siblings show up as separate self-consistent clusters.
  Unrelated galaxies stay below ~0.5.

Steps:

1. **Build templates (pass 1)**, one bundle at a time: correlate the bundle's
   unit-normalized fingerprints with the current templates (one BLAS matrix
   product, n_bundle x n_templates). Files with r >= ``--link`` (0.99) join and
   update that template. The rest are correlated with each other (dense,
   <= ~5,700^2): components linked at r >= ``--link`` with >= 2 members seed
   new templates (a new noise light curve, possibly a sibling of an existing
   one). Leftover singletons seed a template only if they match nothing at
   r >= ``--assign`` (0.6); otherwise they are weak members (mostly
   signal-heavy files), left for pass 2. After the last bundle the pooled
   weak files are clustered once more: a sibling light curve that appears
   only once per bundle seeds its template here. A template is the running sum of its strong members, so
   the injected spot signals, which differ between members, average down
   while the shared noise adds up.
2. **Merge**: templates with r >= ``--merge`` (0.99) are the same light curve
   and are merged; single-member templates are folded into their best
   multi-member template if r >= ``--assign`` (a weak member that arrived
   before its galaxy's template existed).
3. **Assign (pass 2)**: every file is re-scored against the final templates;
   ``galaxy_id`` = best template if r >= ``--assign``, else -1, plus the
   best template regardless (``galaxy_best``) and the best and second-best r
   for auditing. Templates with < ``--min-members`` (2) strong members are
   dropped first: in a large run they are signal-dominated single files
   (short P, high activity), which would otherwise match themselves at r = 1.
4. **Siblings**: template pairs with r >= ``--sibling`` (0.5) are listed, and
   connected components of that graph give ``sibling_group``.

Cost is O(N x T x D) for N files, T ~ 2,000 templates, D = 17,519 (see the
report for timings).

The SIM_ID 800k pool split (T1) is *not* imposed; it is a validation: the
summary reports galaxies (and sibling groups) with members on both sides.

Outputs (``--outdir``, default ``<root>/extracted/galaxy_groups``):
``galaxy_assign.csv.gz`` (row, bundle, galaxy_id, galaxy_best, corr_best, corr_second, sim_id, ...),
``galaxy_table.csv`` (per galaxy: n_files, n_lt800k, n_ge800k, n_corrupt, ...),
``galaxy_siblings.csv`` (template pairs with r >= ``--sibling``),
``templates.npy`` (float32 unit templates) and ``groups_summary.json``.

Usage::

    $PY tools/audit/galaxy_groups.py                     # all finished bundles
    $PY tools/audit/galaxy_groups.py --bundles 1 90 180 --outdir /tmp/x
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "claude_files"))
import determinism  # noqa: E402


def _blas_threads_from_argv(default: int = 4) -> int:
    """Read ``--blas-threads`` before numpy is imported (BLAS reads it once, at load)."""
    a = sys.argv
    if "--blas-threads" in a:
        return int(a[a.index("--blas-threads") + 1])
    return default


if __name__ == "__main__":  # don't touch the env of importers (e.g. pytest)
    determinism.pin_blas_threads(_blas_threads_from_argv())

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix, csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

import galaxy_pass as gp  # noqa: E402
import smarts_data as sd  # noqa: E402

SIM_SPLIT = 800_000


def unit_rows(fp: np.ndarray) -> np.ndarray:
    """int8 fingerprints -> float32 rows of unit norm (all-zero rows stay zero)."""
    x = np.asarray(fp, dtype=np.float32)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(nrm > 0, nrm, 1)


def _normalize(S: np.ndarray) -> np.ndarray:
    """Unit-normalize template sums row-wise."""
    nrm = np.linalg.norm(S, axis=1, keepdims=True)
    return S / np.where(nrm > 0, nrm, 1)


def _components(C: np.ndarray, thresh: float) -> np.ndarray:
    """Connected-component labels of the graph with edges C[i, j] >= thresh (i != j)."""
    i, j = np.nonzero(np.triu(C >= thresh, 1))
    g = coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=C.shape)
    return connected_components(g, directed=False)[1]


def scatter_sum(idx: np.ndarray, X: np.ndarray, n_out: int) -> np.ndarray:
    """Row sums of ``X`` grouped by ``idx`` (0..n_out-1), as a dense float32 (n_out, D) array.

    A sparse one-hot product; ``np.add.at`` on (5,000 x 17,519) float32 took 4.3 s vs 0.27 s (measured 2026-10-21).
    """
    M = csr_matrix((np.ones(len(idx), np.float32), (idx, np.arange(len(idx)))),
                   shape=(n_out, len(idx)))
    return np.asarray(M @ X, dtype=np.float32)


def best_two(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row of a correlation matrix: (argmax, max, second max)."""
    if C.shape[1] == 0:
        z = np.full(C.shape[0], -1.0, np.float32)
        return np.full(C.shape[0], -1), z, z
    k = np.argmax(C, axis=1)
    c1 = C[np.arange(len(C)), k]
    if C.shape[1] == 1:
        return k, c1, np.full(len(C), -1.0, np.float32)
    c2 = np.partition(C, -2, axis=1)[:, -2]
    return k, c1, c2


def _pairs_above(X: np.ndarray, thresh: float, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i < j) with X[i] . X[j] >= thresh, computed in row blocks (no n x n matrix)."""
    ii, jj = [], []
    for s0 in range(0, len(X), block):
        C = X[s0:s0 + block] @ X.T
        i, j = np.nonzero(C >= thresh)
        keep = j > i + s0
        ii.append(i[keep] + s0)
        jj.append(j[keep])
    return (np.concatenate(ii), np.concatenate(jj)) if ii else (np.zeros(0, int),) * 2


def _seed_clusters(X: np.ndarray, link: float) -> np.ndarray:
    """Component labels of rows of X linked at r >= ``link`` (blockwise pair search)."""
    i, j = _pairs_above(X, link)
    g = coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(len(X), len(X)))
    return connected_components(g, directed=False)[1]


def build_templates(fps: list, assign: float, link: float, log=print) -> tuple[np.ndarray, np.ndarray]:
    """Pass 1: grow templates bundle by bundle, then seed from the pooled weak leftovers.

    Returns (float32 (T, D) template sums, int (T,) number of strong members per template).
    """
    D = fps[0].shape[1]
    S = np.zeros((0, D), np.float32)
    cnt = np.zeros(0, int)
    pending = []

    def add_seeds(Xs, lab):
        nonlocal S, cnt
        _, lab = np.unique(lab, return_inverse=True)
        S = np.vstack([S, scatter_sum(lab, Xs, lab.max() + 1)])
        cnt = np.concatenate([cnt, np.bincount(lab)])

    for b, fp in enumerate(fps):
        t0 = time.time()
        X = unit_rows(fp)
        k, c1, _ = best_two(X @ _normalize(S).T)
        strong = c1 >= link
        if strong.any():
            S += scatter_sum(k[strong], X[strong], len(S))
            cnt += np.bincount(k[strong], minlength=len(S))
        U = np.flatnonzero(~strong)
        n_seed = 0
        if len(U):
            lab = _seed_clusters(X[U], link)
            seed = (np.bincount(lab)[lab] >= 2) | (c1[U] < assign)
            if seed.any():
                add_seeds(X[U[seed]], lab[seed])
                n_seed = int(seed.sum())
            pending.append(np.asarray(fp)[U[~seed]])
        log(f"  pass1 chunk {b + 1}/{len(fps)}: {len(X)} files, {strong.sum()} strong, "
            f"{n_seed} seeding, {len(U) - n_seed} weak -> {len(S)} templates "
            f"({time.time() - t0:.1f} s)", flush=True)
    # Weak leftovers: files that matched a template only at assign <= r < link when they arrived.
    # Most are signal-heavy members; but a sibling light curve seen once per bundle never
    # seeds within a chunk, so pool them and seed any cluster of >= 2 that still isn't strong.
    P = unit_rows(np.vstack(pending)) if pending else np.zeros((0, D), np.float32)
    if len(P):
        _, c1, _ = best_two(P @ _normalize(S).T)
        P = P[c1 < link]
        lab = _seed_clusters(P, link)
        seed = np.bincount(lab)[lab] >= 2
        if seed.any():
            add_seeds(P[seed], lab[seed])
        log(f"  pass1 leftovers: {len(P)} weak files, {int(seed.sum())} seeding -> {len(S)} templates",
            flush=True)
    return S, cnt


def merge_templates(S: np.ndarray, cnt: np.ndarray, merge: float, assign: float
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Merge same-light-curve templates (r >= ``merge``), then fold single-member templates.

    A single-member template whose best multi-member template has r >= ``assign``
    is folded into it. Returns (sums, counts, merge stats).
    """
    Tn = _normalize(S)
    lab = _components(Tn @ Tn.T, merge)
    S2, cnt2 = scatter_sum(lab, S, lab.max() + 1), np.bincount(lab, weights=cnt).astype(int)
    single, multi = np.flatnonzero(cnt2 <= 1), np.flatnonzero(cnt2 > 1)
    n_fold = 0
    if len(single) and len(multi):
        Tn2 = _normalize(S2)
        k, c1, _ = best_two(Tn2[single] @ Tn2[multi].T)
        fold = c1 >= assign
        target = np.arange(len(S2))
        target[single[fold]] = multi[k[fold]]
        _, lab3 = np.unique(target, return_inverse=True)
        S2, cnt2 = scatter_sum(lab3, S2, lab3.max() + 1), np.bincount(lab3, weights=cnt2).astype(int)
        n_fold = int(fold.sum())
    return S2, cnt2, dict(n_merged_same_lc=int(len(S) - lab.max() - 1), n_singletons_folded=n_fold)


def assign_all(fps: list, Tn: np.ndarray, assign: float) -> tuple[np.ndarray, ...]:
    """Pass 2: (galaxy id or -1 below ``assign``, best template regardless, best r, second-best r)."""
    ks, c1s, c2s = [], [], []
    for fp in fps:
        k, c1, c2 = best_two(unit_rows(fp) @ Tn.T)
        ks.append(k)
        c1s.append(c1)
        c2s.append(c2)
    k, c1 = np.concatenate(ks), np.concatenate(c1s)
    return np.where(c1 >= assign, k, -1), k, c1, np.concatenate(c2s)


def relabel_by_size(gid: np.ndarray, best: np.ndarray, Tn: np.ndarray) -> tuple[np.ndarray, ...]:
    """Renumber galaxy IDs 0..G-1 by decreasing membership (drop empty templates); -1 stays -1.

    ``best`` (unthresholded best template) is mapped the same way; it is -1 only
    if that template ended up with no assigned members.
    """
    counts = np.bincount(gid[gid >= 0], minlength=len(Tn))
    order = np.argsort(-counts, kind="stable")
    order = order[counts[order] > 0]
    new = np.full(len(Tn), -1)
    new[order] = np.arange(len(order))
    return np.where(gid >= 0, new[np.maximum(gid, 0)], -1), new[best], Tn[order]


def sibling_pairs(Tn: np.ndarray, sibling: float) -> pd.DataFrame:
    """Template pairs (galaxy_id_a < galaxy_id_b) with correlation >= ``sibling``."""
    C = Tn @ Tn.T
    i, j = np.nonzero(np.triu(C >= sibling, 1))
    return pd.DataFrame(dict(galaxy_a=i, galaxy_b=j, corr=C[i, j])).sort_values("corr", ascending=False)


def galaxy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-galaxy summary from the per-file assignment table (needs galaxy_id, sim_id, max_abs_dev)."""
    g = df[df.galaxy_id >= 0].groupby("galaxy_id")
    return pd.DataFrame(dict(
        n_files=g.size(),
        n_lt800k=g.sim_id.apply(lambda s: int((s < SIM_SPLIT).sum())),
        n_ge800k=g.sim_id.apply(lambda s: int((s >= SIM_SPLIT).sum())),
        n_bundles=g.bundle.nunique(),
        n_corrupt=g.max_abs_dev.apply(lambda s: int((s > gp.OUTLIER_THRESH).sum())),
        corr_best_min=g.corr_best.min(),
        corr_best_median=g.corr_best.median(),
        p2p_ppm_median=g.p2p_ppm.median(),
        slow_rms_ppm_median=g.slow_rms_ppm.median(),
        period_min=g.period.min(),
        period_max=g.period.max(),
    )).reset_index()


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pass-dir", default=None, help="default: <root>/extracted/galaxy_pass")
    ap.add_argument("--outdir", default=None, help="default: <root>/extracted/galaxy_groups")
    ap.add_argument("--bundles", type=int, nargs="+", help="default: every finished bundle")
    ap.add_argument("--assign", type=float, default=0.6,
                    help="min r for a file to belong to a template (weak members)")
    ap.add_argument("--link", type=float, default=0.99,
                    help="r for 'same noise light curve' (template updates, seeding links)")
    ap.add_argument("--merge", type=float, default=0.99, help="template-template merge threshold")
    ap.add_argument("--sibling", type=float, default=0.5, help="template-template sibling threshold")
    ap.add_argument("--min-members", type=int, default=2,
                    help="drop templates with fewer strong members before pass 2")
    ap.add_argument("--blas-threads", type=int, default=4)
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    pass_dir = Path(args.pass_dir) if args.pass_dir else Path(root, sd.EXTRACT_DIR, gp.PASS_DIR)
    outdir = Path(args.outdir) if args.outdir else Path(root, sd.EXTRACT_DIR, "galaxy_groups")
    bundles = args.bundles or [n for n in range(1, sd.N_BUNDLES + 1) if gp.bundle_done(pass_dir, n)]
    t0 = time.time()
    stats, fps = gp.load_pass(pass_dir, bundles, fingerprints=True)
    print(f"{len(stats)} files from {len(bundles)} bundles", flush=True)

    t1 = time.time()
    S, cnt = build_templates(fps, args.assign, args.link)
    n_seeded = len(S)
    S, cnt, merge_stats = merge_templates(S, cnt, args.merge, args.assign)
    t2 = time.time()
    # Drop templates with < min_members strong members: at full-corpus scale every galaxy has
    # tens of members, and a lone template is a signal-dominated file (short P, high activity)
    # that would otherwise match itself at r = 1 and pose as a galaxy.
    keep = cnt >= args.min_members
    n_dropped = int((~keep).sum())
    S, cnt = S[keep], cnt[keep]
    gid, best, c1, c2 = assign_all(fps, _normalize(S), args.assign)
    gid, best, Tn = relabel_by_size(gid, best, _normalize(S))
    t3 = time.time()

    cat = np.load(Path(root, sd.EXTRACT_DIR, "catalog", "smarts_catalog.npz"))
    rows = stats.row.values
    df = pd.DataFrame(dict(row=rows, bundle=stats.bundle.values, galaxy_id=gid, galaxy_best=best,
                           corr_best=c1, corr_second=c2, sim_id=cat["SIM_ID"][rows],
                           period=cat["PERIOD"][rows], max_abs_dev=stats.max_abs_dev.values,
                           p2p_ppm=stats.p2p_ppm.values, slow_rms_ppm=stats.slow_rms_ppm.values))
    tab = galaxy_table(df)
    sib = sibling_pairs(Tn, args.sibling)
    g = coo_matrix((np.ones(len(sib)), (sib.galaxy_a, sib.galaxy_b)), shape=(len(Tn), len(Tn)))
    sgroup = connected_components(g, directed=False)[1]
    tab["sibling_group"] = sgroup[tab.galaxy_id]
    tab["max_sibling_corr"] = pd.concat([sib.groupby("galaxy_a")["corr"].max(),
                                         sib.groupby("galaxy_b")["corr"].max()]
                                        ).groupby(level=0).max().reindex(tab.galaxy_id).values
    straddle = tab[(tab.n_lt800k > 0) & (tab.n_ge800k > 0)]
    sg = tab.groupby("sibling_group")[["n_lt800k", "n_ge800k"]].sum()
    summ = dict(
        n_files=len(df), bundles=bundles,
        thresholds=dict(assign=args.assign, link=args.link, merge=args.merge, sibling=args.sibling),
        n_templates_seeded=n_seeded, **merge_stats, n_templates_dropped_lt_min_members=n_dropped,
        n_galaxies=int(len(tab)), n_unassigned=int((gid < 0).sum()),
        n_single_file_galaxies=int((tab.n_files == 1).sum()),
        n_galaxies_lt800k_only=int((tab.n_ge800k == 0).sum()),
        n_galaxies_ge800k_only=int((tab.n_lt800k == 0).sum()),
        n_galaxies_straddling_800k=int(len(straddle)),
        n_files_minority_side_of_straddlers=int(np.minimum(straddle.n_lt800k,
                                                           straddle.n_ge800k).sum()),
        n_sibling_pairs=int(len(sib)), n_sibling_groups_multi=int((tab.sibling_group.value_counts() > 1).sum()),
        n_sibling_groups_straddling_800k=int(((sg.n_lt800k > 0) & (sg.n_ge800k > 0)).sum()),
        group_size_quantiles={str(q): float(np.quantile(tab.n_files, q))
                              for q in (0, 0.01, 0.1, 0.5, 0.9, 0.99, 1)},
        corr_best_quantiles={str(q): float(np.quantile(c1, q))
                             for q in (0, 0.0001, 0.001, 0.01, 0.1, 0.5)},
        corr_second_quantiles={str(q): float(np.quantile(c2, q)) for q in (0.5, 0.99, 0.999, 1)},
        timing_s=dict(load=t1 - t0, pass1_and_merge=t2 - t1, pass2=t3 - t2,
                      total=time.time() - t0, blas_threads=args.blas_threads),
    )
    outdir.mkdir(parents=True, exist_ok=True)
    with sd.atomic_path(outdir / "galaxy_assign.csv.gz") as tmp:
        df.to_csv(tmp, index=False, compression="gzip")
    with sd.atomic_path(outdir / "galaxy_table.csv") as tmp:
        tab.to_csv(tmp, index=False)
    with sd.atomic_path(outdir / "galaxy_siblings.csv") as tmp:
        sib.to_csv(tmp, index=False)
    with sd.atomic_path(outdir / "templates.npy") as tmp:
        np.save(tmp, Tn.astype(np.float32))
    with sd.atomic_path(outdir / "groups_summary.json") as tmp:
        tmp.write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "bundles"}, indent=1))


if __name__ == "__main__":
    main()

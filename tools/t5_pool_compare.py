#!/usr/bin/env python3
"""T5 phase 2: compare candidate-pool configurations on rotators and non-rotators.

For each star the light curve is loaded and the ACFs computed once, every
Stage-1 generator needed by any configuration is run once (timed), and then
`fit_rotation_period` is run once per pool configuration (timed). Per star and
configuration it records the final answer (period or decline), the pooled raw
guesses, and every fitted candidate with its contributing methods, so the
analysis can count recall after the short-period cap, extra wrong candidates,
and false-positive answers on non-rotators.

Configurations (CONFIGS below) are lists of (label, generator name, kwargs).
Generators named "*_old" run the pre-T5 ranking (rank_by="power").

Resumable (`--resume` skips stars with a shard) and atomic (temp file + rename),
so an unplugged drive mid-run leaves no partial shards. The data root is checked
with a real read before any work starts.

Usage:
    python tools/t5_pool_compare.py --manifest datasets/dev_medium.csv --data-root data/smarts_dev \
        --ignore-manifest-sectors --outdir runs/t5_pool/rot_fy --resume
    python tools/t5_pool_compare.py --manifest datasets/negatives_v0.csv --only-set dev_medium \
        --data-root /Volumes/Data_Vault/smarts_data --outdir runs/t5_pool/neg --resume
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "claude_files"), str(ROOT / "benchmarks")]
import determinism  # noqa: E402

determinism.pin_blas_threads()

import argparse  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

N_GUESSES = {"acf_fft_highpass": 5}  # project convention (scorecard.N_GUESSES)
N_GUESSES_DEFAULT = 10

# generator label -> (guess_* function name, extra kwargs)
GENERATORS = {
    "pairwise_histogram": ("guess_pairwise_histogram", {}),
    "lombscargle_old": ("guess_lombscargle", {"rank_by": "power"}),
    "acf_fft_old": ("guess_acf_fft", {"rank_by": "power"}),
    "lombscargle": ("guess_lombscargle", {}),
    "acf_fft": ("guess_acf_fft", {}),
    "acf_fft_highpass": ("guess_acf_fft_highpass", {}),
    "acf_fft_short": ("guess_acf_fft_short", {}),
    "lombscargle_short": ("guess_lombscargle_short", {}),
}
CONFIGS = {
    "base_old": ["pairwise_histogram", "lombscargle_old", "acf_fft_old"],
    "base": ["pairwise_histogram", "lombscargle", "acf_fft"],
    "base+highpass": ["pairwise_histogram", "lombscargle", "acf_fft", "acf_fft_highpass"],
    "base+acf_short": ["pairwise_histogram", "lombscargle", "acf_fft", "acf_fft_short"],
    "base+ls_short": ["pairwise_histogram", "lombscargle", "acf_fft", "lombscargle_short"],
    "base_old+highpass": ["pairwise_histogram", "lombscargle_old", "acf_fft_old", "acf_fft_highpass"],
}


def check_data_root(root: Path, sample_rel: str) -> None:
    """Exit with a clear message unless `root/sample_rel` can actually be read."""
    p = root / sample_rel
    try:
        with open(p, "rb") as fh:
            fh.read(1024)
    except OSError as exc:
        raise SystemExit(f"Cannot read {p} ({exc}). Plug in the Data_Vault drive or fix --data-root.")


def _fitted_rows(res) -> list:
    """Compact per-candidate records (period, methods, passed gates) from an EnsembleResult."""
    rows = []
    for c in list(res.candidates) + list(res.failed_candidates):
        rows.append(dict(P=float(c.period) if c.period is not None else None,
                         methods="+".join(c.contributing_methods or (c.source_guess.method,)),
                         passed=bool(c.passed_gates), error=c.error is not None))
    return rows


def run_one(task: dict, outdir: str, configs: dict) -> str:
    """Process one star (module-level so it pickles): load, generate, fit each config, save a shard."""
    fn = Path(outdir) / f"{task['star_id']}.pkl"
    if fn.exists():
        return "skip"
    warnings.simplefilter("ignore")
    import guesses as G
    from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf
    from comb_fit import fit_rotation_period
    from preprocessing import load_smarts_fits

    rec = dict(star_id=task["star_id"], error=None)
    try:
        t_load = time.perf_counter()
        pre = load_smarts_fits(task["path"], **task.get("load_kwargs", {}))
        t, f = pre.time, pre.flux
        lags, acf, counts = compute_acf(t, f, return_counts=True)
        lags1, acf1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
        rec["load_seconds"] = time.perf_counter() - t_load
        rec["baseline"] = float(t[-1] - t[0])
        needed = sorted({g for c in configs.values() for g in c})
        gen, gen_s = {}, {}
        for name in needed:
            fname, kw = GENERATORS[name]
            t0 = time.perf_counter()
            try:
                gen[name] = G.two_view_guesses(getattr(G, fname), t, f, (lags1, acf1), (lags, acf),
                                               n_guesses=N_GUESSES.get(name, N_GUESSES_DEFAULT), **kw)
            except Exception:  # noqa: BLE001 -- a failed generator contributes nothing
                gen[name] = []
            gen_s[name] = time.perf_counter() - t0
        rec["gen_seconds"] = gen_s
        rec["raw"] = {k: [(g.P0, g.method) for g in v] for k, v in gen.items()}
        rec["configs"] = {}
        for cname, members in configs.items():
            pooled = [g for m in members for g in gen[m]]
            t0 = time.perf_counter()
            res = fit_rotation_period(lags, acf, pooled, acf_counts=counts)
            rec["configs"][cname] = dict(
                final_period=float(res.best_fit.P) if res.success else None,
                n_pooled=len(pooled), n_tried=res.n_candidates_tried,
                fit_seconds=time.perf_counter() - t0, fitted=_fitted_rows(res))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    tmp = fn.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(rec, fh)
    os.replace(tmp, fn)
    return "ok"


def build_tasks(args) -> list:
    """Task dicts (star_id, path, load_kwargs) from the manifest, honoring --only-set / sectors flags."""
    man = pd.read_csv(args.manifest)
    if args.only_set:
        man = man[man["eval_set"] == args.only_set]
    root = Path(args.data_root)
    tasks = []
    for _, r in man.iterrows():
        sec = r.get("sectors")
        use = (not args.ignore_manifest_sectors) and isinstance(sec, str) and sec.strip()
        kw = {"sectors": [int(s) for s in sec.split("-")]} if use else {}
        tasks.append(dict(star_id=str(r["star_id"]), path=str(root / r["path"]), load_kwargs=kw))
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def main(argv=None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--only-set", default=None)
    ap.add_argument("--ignore-manifest-sectors", action="store_true")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated subset of CONFIGS")
    ap.add_argument("--n-workers", type=int, default=determinism.default_n_workers())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    tasks = build_tasks(args)
    check_data_root(Path(args.data_root), Path(tasks[0]["path"]).relative_to(args.data_root).as_posix())
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.outdir.glob("*.pkl")):
        raise SystemExit(f"{args.outdir} has shards; pass --resume")
    configs = {k: CONFIGS[k] for k in args.configs.split(",")}
    work = partial(run_one, outdir=str(args.outdir), configs=configs)
    t0 = time.time()
    with ProcessPoolExecutor(args.n_workers, initializer=determinism.init_worker) as ex:
        for i, _ in enumerate(ex.map(work, tasks, chunksize=2), start=1):
            if i % 100 == 0:
                print(f"{i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(tasks)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

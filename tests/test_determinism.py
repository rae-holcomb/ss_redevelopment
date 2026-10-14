"""Bit-reproducibility of the pipeline (T14).

Runs Stage 1 (the pipeline's default guess methods) and Stage 2
(fit_rotation_period) on a few synthetic stars three ways -- in-process,
in-process again after perturbing the heap, and through a spawn-based
ProcessPoolExecutor -- and asserts every candidate list and final period is
bit-identical.

The heap perturbation matters: the original non-determinism came from Apple
Accelerate's alignment-dependent ddot (see claude_files/determinism.py), which
shows up whenever temporaries land at different addresses -- exactly what
random live allocations between runs provoke. The stars were chosen because
their final period varied run to run under Accelerate.

Requires an OpenBLAS-linked numpy (the ss_openblas env, environment.yml). On
any other BLAS the test is an expected failure (xfail) with the reason, so a
wrong environment is visible in the summary rather than silently passing.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

import determinism
from acf_utils import compute_acf
from comb_fit import fit_rotation_period
import guesses
from synthetic import make_suite

PIPELINE_METHODS = ("pairwise_histogram", "lombscargle", "acf_fft")
SUITE_N, SUITE_SEED = 40, 0
# Stars whose final period differed between repeated runs under Accelerate
# (T14 reproduction: 2-6 distinct values in 8 heap-perturbed repeats).
STAR_IDX = (4, 14, 38)


def _run_star(idx: int) -> dict:
    """Run Stage 1 + Stage 2 on star `idx` of the reference suite.

    Module-level so ProcessPoolExecutor can pickle it. Returns the exact
    candidate periods per method (or the exception name) and the final
    period, as Python floats so equality is bit-exact.
    """
    lc = make_suite(SUITE_N, seed=SUITE_SEED)[idx]
    lags, acf = compute_acf(lc.time, lc.flux)
    cands, pooled = {}, []
    for m in PIPELINE_METHODS:
        try:
            out = getattr(guesses, "guess_" + m)(lc.time, lc.flux, lags, acf, n_guesses=10)
        except Exception as exc:  # noqa: BLE001 -- a raise must also be reproducible
            cands[m] = type(exc).__name__
            continue
        cands[m] = tuple(float(g.P0) for g in out)
        pooled.extend(out)
    res = fit_rotation_period(lags, acf, pooled)
    final = float(res.best_fit.P) if res.best_fit is not None else None
    return dict(star_id=lc.star_id, candidates=cands, final_period=final, success=bool(res.success))


def _perturb_heap(rng: np.random.Generator, keep: list) -> None:
    """Allocate a random number of small, randomly sized arrays and keep them
    alive, so later temporaries land at different memory alignments."""
    keep.extend(np.empty(int(n)) for n in rng.integers(1, 64, size=int(rng.integers(5, 40))))


@pytest.mark.slow
def test_pipeline_bit_identical_across_runs_and_processes():
    """Same inputs -> bit-identical candidates and final periods, in-process,
    after heap perturbation, and in spawned pool workers."""
    if determinism.blas_backend() != "openblas":
        pytest.xfail(f"numpy BLAS is {determinism.blas_backend()!r}, not OpenBLAS: "
                     "results are known not to be bit-reproducible (use the ss_openblas env)")

    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        first = [_run_star(i) for i in STAR_IDX]
        rng, keep = np.random.default_rng(123), []
        second = []
        for i in STAR_IDX:
            _perturb_heap(rng, keep)
            second.append(_run_star(i))

    determinism.pin_blas_threads()  # inherited by the spawned workers
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"),
                             initializer=determinism.init_worker) as ex:
        pooled = list(ex.map(_run_star, STAR_IDX))

    for a, b, c in zip(first, second, pooled):
        assert a == b, f"{a['star_id']}: in-process repeat differs:\n{a}\n{b}"
        assert a == c, f"{a['star_id']}: process-pool run differs:\n{a}\n{c}"


def test_pin_blas_threads_sets_every_variable(monkeypatch):
    """pin_blas_threads overrides every BLAS/OpenMP thread variable, including
    ones already set in the user's shell."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    determinism.pin_blas_threads(1)
    import os
    for var in determinism.BLAS_THREAD_ENV_VARS:
        assert os.environ[var] == "1"


def test_default_n_workers_is_positive_and_bounded():
    """default_n_workers returns a usable pool size no larger than the
    logical CPU count."""
    import os
    n = determinism.default_n_workers()
    assert 1 <= n <= (os.cpu_count() or n)

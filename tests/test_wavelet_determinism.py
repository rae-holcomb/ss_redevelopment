"""Bit-reproducibility of guess_wavelet (T31 Part A, backlog P4).

Before T31, the wavelet used two numpy 1.26.4 SIMD loops whose last bits
varied with process / allocation state on the i5 (AVX-512): the float64
np.exp (Morlet envelope, Gaussian peak fits) and the complex np.abs (|c|^2).
The CWT, the GWPS and the wavelet candidates then varied run to run (T12 saw
2-4 distinct GWPS per star over 8 in-process calls on dev_small 031596,
419951, 107369; np.abs flipped in ~1 of 6 fresh processes). Now the kernel is
one complex exp (scalar libm loop), |c|^2 is re*re + im*im, and the Gaussian
uses guesses._det_exp.

Checks:
- the kernel equals an element-by-element reference bit for bit;
- repeated kernel builds under heap churn give identical bytes;
- repeated guess_wavelet calls on the three dev_small stars give identical
  GWPS and candidate bytes, in-process and in a spawned worker process
  (skipped when data/smarts_dev isn't present).
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import determinism
import guesses

ROOT = Path(__file__).resolve().parents[1]
DEV_DIR = ROOT / "data" / "smarts_dev"
STARS = ("031596", "419951", "107369")


def _reference_kernel(M: int, s: float, w0: float) -> np.ndarray:
    """Element-by-element Morlet kernel: numpy's complex exp on one-element arrays.

    Each element is computed in its own fresh array, so it can't depend on the
    element's position, array length or alignment. (numpy's complex exp calls
    the platform cexp, which can differ from CPython's cmath.exp by an ulp;
    both are deterministic scalar paths.)
    """
    x = (np.arange(0, M) - (M - 1.0) / 2.0) / s
    vals = np.empty(M, dtype=complex)
    for i, v in enumerate(x):
        vals[i] = np.exp(np.array([complex(-0.5 * (v * v), w0 * v)]))[0]  # array x**2 is x*x
    return np.sqrt(1.0 / s) * (vals * np.pi**(-0.25))


@pytest.mark.parametrize("M,s", [(3, 0.4), (20, 2.0), (102, 10.2), (731, 73.1), (5190, 1800.0)])
def test_morlet_kernel_matches_cmath_reference(M, s):
    """Each kernel element equals its stand-alone complex exp (no SIMD-path, alignment-dependent drift)."""
    k = guesses._morlet2_kernel(M, s, 6.0)
    assert k.tobytes() == _reference_kernel(M, s, 6.0).tobytes()


def test_morlet_kernel_repeatable_under_heap_churn():
    """Repeated builds of the same kernels give identical bytes while live allocations shift around."""
    rng = np.random.default_rng(0)
    scales = np.geomspace(0.5, 2000.0, 120)
    ref = [guesses._morlet2_kernel(max(int(min(10 * s, 5190)), 3), s, 6.0).tobytes() for s in scales]
    keep = []
    for _ in range(5):
        for i, s in enumerate(scales):
            keep.append(np.empty(int(rng.integers(1, 6000))))
            if len(keep) > 30:
                keep.pop(int(rng.integers(0, 30)))
            assert guesses._morlet2_kernel(max(int(min(10 * s, 5190)), 3), s, 6.0).tobytes() == ref[i]


def test_abs2_matches_rowwise_and_definition():
    """|c|^2 of the 2D CWT equals the per-row result and re*re + im*im exactly.

    The old np.abs(c)**2 failed the row-vs-2D comparison in ~1 of 6 fresh
    processes (numpy 1.26.4 complex abs SIMD loop), so this also guards that
    path from coming back.
    """
    rng = np.random.default_rng(2)
    t = np.arange(0, 54.0, 1 / 48)
    y = np.sin(2 * np.pi * t / 4.3) + rng.normal(0, 0.3, len(t))
    y -= y.mean()
    dt = t[1] - t[0]
    scales = guesses._period_to_morlet_scale(np.geomspace(4 * dt, 27.0, 60), dt, 6.0)
    W = guesses._cwt_morlet(y, scales, 6.0)
    P = guesses._abs2(W)
    for i in range(len(scales)):
        assert guesses._abs2(W[i]).tobytes() == P[i].tobytes()
    assert P.tobytes() == (W.real * W.real + W.imag * W.imag).tobytes()


def test_det_exp_is_elementwise_libm():
    """_det_exp of an array equals _det_exp of each element on its own, and is close to np.exp."""
    x = -0.5 * np.linspace(-6, 6, 301) ** 2
    full = guesses._det_exp(x)
    one = np.array([guesses._det_exp(np.array([v]))[0] for v in x])
    assert full.tobytes() == one.tobytes()
    np.testing.assert_allclose(full, np.exp(x), rtol=1e-15, atol=0)


def _wavelet_digest(star: str) -> str:
    """md5 over the GWPS and every candidate's (P0, strength) for one dev_small star (assigned sectors).

    Module-level so a spawned ProcessPoolExecutor worker can run it.
    """
    import sys
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import warnings
    warnings.simplefilter("ignore")
    import pandas as pd
    import scorecard as sc
    man = pd.read_csv(ROOT / "datasets" / "dev_small.csv")
    row = man[man["star_id"].str.endswith(star)].iloc[0]
    from types import SimpleNamespace
    args = SimpleNamespace(synthetic=None, manifest=ROOT / "datasets" / "dev_small.csv", seed=0,
                           data_root=str(DEV_DIR), ignore_manifest_sectors=False, coverage=sc.cov.FULL)
    task = next(t for t in sc.build_tasks(args) if t["star_id"] == row["star_id"])
    _, t, f, *_ = sc._load(task)
    g = guesses.guess_wavelet(t, sc._gap_filled(f), None, None, n_guesses=10)
    h = hashlib.md5(g[0].info["gwps"].tobytes())
    for c in g:
        h.update(np.array([c.P0, c.strength]).tobytes())
    return h.hexdigest()


needs_dev = pytest.mark.skipif(not DEV_DIR.exists(), reason="data/smarts_dev not present (symlink it in worktrees)")


@pytest.mark.slow
@needs_dev
@pytest.mark.parametrize("star", STARS)
def test_guess_wavelet_repeatable_in_process(star):
    """Five in-process calls on the same star give identical GWPS and candidate bytes."""
    rng = np.random.default_rng(1)
    keep = []
    digests = set()
    for _ in range(5):
        keep.append(np.empty(int(rng.integers(1, 50000))))
        digests.add(_wavelet_digest(star))
    assert len(digests) == 1


@pytest.mark.slow
@needs_dev
def test_guess_wavelet_repeatable_across_processes():
    """A spawned worker process gives the same wavelet bytes as this process."""
    here = [_wavelet_digest(s) for s in STARS]
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                             initializer=determinism.init_worker) as ex:
        there = list(ex.map(_wavelet_digest, STARS))
    assert here == there

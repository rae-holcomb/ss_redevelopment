"""
determinism.py -- process setup that makes pipeline runs bit-reproducible.

Why this exists (T14, 2026-10-14)
---------------------------------
Two scorecard runs on identical inputs used to give final periods differing
by up to ~1% (usually ~1e-6) on a few percent of stars. The cause was NOT
threading, scheduling or randomness: Apple Accelerate's ddot (what np.dot and
np.linalg.norm call on 1-D float64 arrays) returns a result whose last bits
depend on the input buffer's memory alignment (32-byte period). numpy buffers
are only 16-byte aligned, so the answer changed with heap state, even between
repeated calls in one process. scipy.optimize.least_squares (trf) uses those
norms for its trust-region radius, so the comb fit took different iteration
paths and stopped at different points. Full evidence:
reports/2026-10-14_t14-bit-reproducible-scorecards/.

The fix has two halves:

1. **Environment**: numpy/scipy linked against OpenBLAS, whose kernels are
   alignment-independent (`environment.yml`, env `ss_openblas`). There is no
   environment variable that makes Accelerate alignment-independent;
   `blas_backend()` / `warn_if_nondeterministic_blas()` detect it.
2. **Thread pinning (this module)**: one BLAS thread per process. OpenBLAS
   splits large reductions across threads, so the summation order (and the
   last bits) depend on the thread count. Pinning also avoids
   oversubscription: with OpenBLAS's default of one thread per logical CPU,
   10 pool workers on an 8-thread machine ran ~50x slower.

Usage
-----
BLAS libraries read their thread-count variables once, when they are loaded,
so `pin_blas_threads()` must run BEFORE numpy/scipy are imported. On macOS,
ProcessPoolExecutor uses the "spawn" start method: each worker re-imports the
main script (and numpy) before any initializer runs, but inherits os.environ
from the parent. So call it at the very top of every entry-point script::

    import determinism            # no numpy import inside this module
    determinism.pin_blas_threads()
    import numpy as np            # now loads with 1 BLAS thread

and pass `initializer=determinism.init_worker` to ProcessPoolExecutor as a
belt-and-braces runtime limit (threadpoolctl) for anything that was already
loaded.
"""

from __future__ import annotations

import os
import warnings

# Every thread-count variable a numpy/scipy BLAS/LAPACK/OpenMP backend reads.
BLAS_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pin_blas_threads(n_threads: int = 1) -> None:
    """Set every BLAS/OpenMP thread-count environment variable to `n_threads`.

    Must be called before numpy/scipy are first imported in this process to
    take effect for this process; child processes started afterwards
    (ProcessPoolExecutor workers) inherit the setting either way. Overrides
    any value already in the user's shell on purpose, so results don't depend
    on how the shell was configured.
    """
    for var in BLAS_THREAD_ENV_VARS:
        os.environ[var] = str(n_threads)


def init_worker() -> None:
    """ProcessPoolExecutor initializer: pin BLAS threads in this worker.

    The environment variables are already inherited from the parent (see
    module docstring); this additionally applies a runtime limit via
    threadpoolctl, which covers a BLAS that was loaded before the variables
    were set. A no-op on that second part if threadpoolctl isn't installed.
    """
    pin_blas_threads(1)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(1)


def blas_backend() -> str:
    """Name of the BLAS library numpy is linked against (lower-case), e.g.
    'openblas', 'accelerate', 'mkl', or 'unknown'.

    Uses threadpoolctl when it can see the library, else numpy's build config.
    Imports numpy, so only call after pin_blas_threads().
    """
    import numpy as np
    try:
        from threadpoolctl import threadpool_info
        for info in threadpool_info():
            if info.get("user_api") == "blas":
                return str(info.get("internal_api", "unknown")).lower()
    except ImportError:
        pass
    try:
        cfg = np.show_config(mode="dicts")
        name = str(cfg["Build Dependencies"]["blas"]["name"]).lower()
    except Exception:  # noqa: BLE001 -- older numpy without mode="dicts"
        return "unknown"
    if "openblas" in name:
        return "openblas"
    if "mkl" in name:
        return "mkl"
    if "accelerate" in name or name == "blas":
        # numpy built against the macOS system BLAS reports a bare "blas"
        return "accelerate"
    return name


def warn_if_nondeterministic_blas() -> bool:
    """Warn if numpy's BLAS is known to make results non-reproducible.

    Returns True if the backend is fine (OpenBLAS), False otherwise. Accelerate
    is known-bad (alignment-dependent ddot, see module docstring); anything else
    is unverified.
    """
    backend = blas_backend()
    if backend == "openblas":
        return True
    warnings.warn(
        f"numpy is linked against {backend!r}, not OpenBLAS: pipeline results "
        "are not guaranteed bit-reproducible run to run (Accelerate is known "
        "to be alignment-dependent). Use the ss_openblas env (environment.yml).",
        RuntimeWarning, stacklevel=2,
    )
    return False


def default_n_workers() -> int:
    """Default process-pool size: one worker per physical core.

    BLAS is pinned to one thread per worker, so the CPU-bound pipeline gains
    little from hyperthreads (measured in T14 on a 4-core/8-thread Intel i5:
    see that report's runtime table). Falls back to os.cpu_count() // 2 if the
    physical-core count can't be read.
    """
    n = None
    try:
        import subprocess
        n = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
    except Exception:  # noqa: BLE001 -- not macOS, or sysctl unavailable
        try:
            import psutil  # optional
            n = psutil.cpu_count(logical=False)
        except Exception:  # noqa: BLE001
            n = None
    if not n:
        n = max(1, (os.cpu_count() or 2) // 2)
    return max(1, int(n))

"""T12 (efficiency pass): each speed-up must leave results bit-identical.

These tests pin the equivalence of every optimized code path against the
straightforward computation it replaced.
"""
import numpy as np
import pytest
from conftest import GUESS_METHODS, prep_flux_for

import guesses
from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf

ACF_INDEPENDENT = [m for m, fn in GUESS_METHODS.items() if getattr(fn, "acf_independent", False)]


def _sig(gs):
    """Comparable signature of a guess list (method, rank, exact P0 and strength)."""
    return [(g.method, g.rank, g.P0, g.strength) for g in gs]


def test_acf_independent_marks_expected_methods():
    """The four light-curve-domain methods are marked; the ACF-based ones are not."""
    assert set(ACF_INDEPENDENT) == {"lombscargle", "lombscargle_short", "acf_fft_highpass", "wavelet"}


@pytest.mark.parametrize("method", ACF_INDEPENDENT)
@pytest.mark.parametrize("lc_name", ["clean_lc", "noise_lc"])
def test_marked_methods_ignore_the_acf(method, lc_name, request):
    """A marked method returns exactly the same candidates whatever ACF it is handed."""
    lc = request.getfixturevalue(lc_name)
    fn = GUESS_METHODS[method]
    flux = prep_flux_for(method, lc.flux)
    real = fn(lc.time, flux, lc.acf_lags, lc.acf, n_guesses=10)
    rng = np.random.default_rng(0)
    junk_lags = np.arange(37) * 0.7
    junk = rng.normal(size=37)
    assert _sig(fn(lc.time, flux, junk_lags, junk, n_guesses=10)) == _sig(real)


@pytest.mark.parametrize("method", list(GUESS_METHODS))
def test_two_view_guesses_unchanged_by_skip(method, clean_lc):
    """two_view_guesses with the long-view skip == the old always-two-calls path."""
    fn = GUESS_METHODS[method]
    t, f = clean_lc.time, prep_flux_for(method, clean_lc.flux)
    long_view = compute_acf(t, f if method != "wavelet" else clean_lc.flux)
    short_view = compute_acf(t, clean_lc.flux, max_lag_frac=STAGE1_MAX_LAG_FRAC)

    def unmarked(*a, **k):
        """Same function without the acf_independent mark (forces the second call)."""
        return fn(*a, **k)

    fast = guesses.two_view_guesses(fn, t, f, short_view, long_view, n_guesses=10)
    slow = guesses.two_view_guesses(unmarked, t, f, short_view, long_view, n_guesses=10)
    assert _sig(fast) == _sig(slow)


@pytest.mark.parametrize("seed", range(5))
def test_comb_scores_over_t0_matches_comb_score(seed, clean_lc, noise_lc):
    """Vectorized phase scan == comb_score in a loop, exactly (incl. the -inf no-teeth case)."""
    import comb_fit as cf
    rng = np.random.default_rng(seed)
    for lc in (clean_lc, noise_lc):
        lags, acf = lc.acf_lags, lc.acf
        for P in rng.uniform(0.05, 0.6 * lags[-1], size=20):
            t0s = np.linspace(rng.uniform(0, 1), rng.uniform(0, 1) + P, 57, endpoint=False)
            t0s = np.append(t0s, lags[-1] + 1.0)  # no tooth in range -> -inf
            for n_max in (0, 3, 5):
                fast = cf._comb_scores_over_t0(lags, acf, P, t0s, n_max)
                slow = [cf.comb_score(lags, acf, P, t, n_max=n_max) for t in t0s]
                assert fast == slow


def _loop_residual_jacobian(st, windows, acf_lags, acf, weights):
    """The pre-T12 per-window residual/jacobian loops, as a reference."""
    m = len(windows)
    lag_subs = [acf_lags[w.mask] for w in windows]
    acf_subs = [acf[w.mask] for w in windows]
    w_subs = [np.ones(len(l)) for l in lag_subs] if weights is None else [weights[w.mask] for w in windows]
    starts = np.cumsum([0] + [len(l) for l in lag_subs])
    ns, idx_P, idx_t0, idx_A0, idx_h0 = st["ns"], st["idx_P"], st["idx_t0"], st["idx_A0"], st["idx_h0"]
    delta_positions = [i for i, n in enumerate(ns) if n > 0]
    delta_pos_of = {i: k for k, i in enumerate(delta_positions)}

    def residual(x):
        """Reference residual."""
        centers = st["centers"](x)
        out = np.empty(st["total_pts"])
        for i in range(m):
            A, h, c = x[idx_A0 + i], x[idx_h0 + i], centers[i]
            out[starts[i]:starts[i + 1]] = w_subs[i] * ((h - A * (lag_subs[i] - c) ** 2) - acf_subs[i])
        return out

    def jacobian(x):
        """Reference jacobian."""
        centers = st["centers"](x)
        J = np.zeros((st["total_pts"], st["n_params"]))
        for i in range(m):
            n, A, c = ns[i], x[idx_A0 + i], centers[i]
            s0, s1 = starts[i], starts[i + 1]
            wi = w_subs[i]
            diff = lag_subs[i] - c
            dresid_dc = wi * 2.0 * A * diff
            J[s0:s1, idx_A0 + i] = -wi * diff**2
            J[s0:s1, idx_h0 + i] = wi
            J[s0:s1, idx_P] = dresid_dc * n
            J[s0:s1, idx_t0] = dresid_dc
            if i in delta_pos_of:
                J[s0:s1, 2 + delta_pos_of[i]] = dresid_dc
        return J
    return residual, jacobian


@pytest.mark.parametrize("use_weights", [False, True])
@pytest.mark.parametrize("n_peaks", [1, 3, 8])
def test_vectorized_comb_model_matches_loop(use_weights, n_peaks, clean_lc):
    """Concatenated residual/jacobian == the per-window loop, bit for bit."""
    import comb_fit as cf
    lags, acf, counts = compute_acf(clean_lc.time, clean_lc.flux, return_counts=True)
    weights = np.sqrt(counts / counts[0]) if use_weights else None
    rng = np.random.default_rng(1)
    for P0 in (0.7, 3.0, 5.5):
        wins = cf._build_windows(lags, P0, 0.9 * P0, n_peaks, max_points_per_window=100)
        st = cf._comb_model_setup(wins, lags, acf, P0, 0.9 * P0, weights=weights)
        ref_r, ref_j = _loop_residual_jacobian(st, wins, lags, acf, weights)
        for _ in range(5):
            x = st["x0"] * (1 + 0.05 * rng.normal(size=st["x0"].size))
            assert st["residual"](x).tobytes() == ref_r(x).tobytes()
            Jn, Jr = st["jacobian"](x), ref_j(x)
            assert Jn.tobytes() == Jr.tobytes()  # also checks signed zeros


def _uncached_full_acf(flux):
    """Pre-T12 compute_acf core (no cache), as a reference."""
    mask = np.isfinite(flux)
    x = np.where(mask, flux - np.mean(flux[mask]), 0.0)
    n = len(x)
    fx = np.fft.fft(x, n=2 * n)
    raw = np.fft.ifft(fx * np.conj(fx)).real[:n]
    fm = np.fft.fft(mask.astype(float), n=2 * n)
    vc = np.clip(np.fft.ifft(fm * np.conj(fm)).real[:n], 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        acf = raw / vc
    acf[vc == 0] = np.nan
    return acf / acf[0], vc


def test_acf_cache_is_exact_and_safe(clean_lc, noise_lc):
    """Cached compute_acf == uncached core for repeated, interleaved and mutated inputs."""
    import acf_utils
    for lc in (clean_lc, noise_lc, clean_lc):
        for frac in (0.7, STAGE1_MAX_LAG_FRAC, 0.7):
            lags, acf, counts = acf_utils.compute_acf(lc.time, lc.flux, max_lag_frac=frac, return_counts=True)
            ref_acf, ref_vc = _uncached_full_acf(np.asarray(lc.flux, float))
            k = len(lags)
            assert acf.tobytes() == ref_acf[:k].tobytes()
            assert counts.tobytes() == np.rint(ref_vc[:k]).tobytes()
            acf[:] = 99.0  # callers own their returned arrays: must not poison the cache
    f2 = clean_lc.flux.copy()
    f2[5] += 1e-12  # one-ulp-scale change -> a different key, recomputed
    _, a2 = acf_utils.compute_acf(clean_lc.time, f2)
    ref2, _ = _uncached_full_acf(f2)
    assert a2.tobytes() == ref2[:len(a2)].tobytes()


def test_exact_cache_lru():
    """_ExactCache evicts the least recently used entry and keys on exact bytes."""
    from acf_utils import _ExactCache
    c = _ExactCache(maxsize=2)
    a, b, d = np.zeros(3), np.ones(3), np.full(3, 2.0)
    ka, kb, kd = (_ExactCache.key(x) for x in (a, b, d))
    c.put(ka, "a")
    c.put(kb, "b")
    assert c.get(ka) == "a"  # a is now most recent
    c.put(kd, "d")           # evicts b
    assert c.get(kb) is None and c.get(ka) == "a" and c.get(kd) == "d"
    assert _ExactCache.key(np.array([0.0])) != _ExactCache.key(np.array([-0.0]))


def _s_ph_loop(time, flux, P, min_bins=5):
    """Pre-T12 ml_features._s_ph, as a reference."""
    dt = np.median(np.diff(time))
    bin_pts = max(1, int(round(P / dt)))
    n_bins = len(flux) // bin_pts
    if bin_pts < 2 or n_bins < min_bins:
        return float("nan")
    stds = []
    for i in range(n_bins):
        seg = flux[i * bin_pts:(i + 1) * bin_pts]
        seg_valid = seg[np.isfinite(seg)]
        if len(seg_valid) >= max(3, bin_pts // 3):
            stds.append(np.std(seg_valid))
    return float(np.mean(stds)) if stds else float("nan")


@pytest.mark.parametrize("seed", range(4))
def test_s_ph_vectorized_matches_loop(seed, clean_lc, noise_lc):
    """Block-std _s_ph == the per-bin loop exactly, with and without NaN gaps."""
    import ml_features as mf
    rng = np.random.default_rng(seed)
    for lc in (clean_lc, noise_lc):
        t, f = lc.time, lc.flux.copy()
        if seed % 2:
            f[rng.integers(0, len(f), 300)] = np.nan  # scattered NaNs inside bins
        for P in np.concatenate([[0.03, 0.06, 0.1], rng.uniform(0.05, 12.0, 25)]):
            a, b = mf._s_ph(t, f, P), _s_ph_loop(t, f, P)
            assert (a == b) or (np.isnan(a) and np.isnan(b)), (P, a, b)


def _nearest_peak_stats_ref(x, y, target):
    """Pre-T12 ml_features._nearest_peak_stats (no caching), as a reference."""
    from scipy.signal import find_peaks
    idx, _ = find_peaks(y)
    if len(idx) == 0:
        return None
    peak_x, peak_y = x[idx], y[idx]
    order = np.argsort(peak_y)[::-1]
    ranks = np.empty(len(idx), dtype=int)
    ranks[order] = np.arange(1, len(idx) + 1)
    nl = int(np.argmin(np.abs(peak_x - target)))
    ni = idx[nl]
    raw = float(peak_y[nl])
    y_max = float(np.max(y))
    half = raw / 2
    i = ni
    while i > 0 and y[i] > half:
        i -= 1
    j = ni
    while j < len(y) - 1 and y[j] > half:
        j += 1
    return dict(rank=int(ranks[nl]), raw_value=raw, frac_power=raw / y_max if y_max > 0 else np.nan,
                rel_distance=float(abs(peak_x[nl] - target) / target), width=float(abs(x[j] - x[i])),
                is_local_max_within_tol=None)


def test_cached_peak_stats_match_reference():
    """Cached peak tables give exactly the uncached _nearest_peak_stats, incl. after array reuse."""
    import ml_features as mf
    rng = np.random.default_rng(3)
    for trial in range(30):
        x = np.sort(rng.uniform(0.1, 50, 400))
        y = np.abs(rng.normal(size=400)) + (trial == 0) * 0.0
        flat = np.ones(50)
        for target in rng.uniform(0.1, 50, 15):
            assert mf._nearest_peak_stats(x, y, target) == _nearest_peak_stats_ref(x, y, target)
        assert mf._nearest_peak_stats(np.arange(50.0), flat, 3.0) is None
        del x, y  # ids get reused by later arrays: the identity check must catch that


def test_fft_family_block_cache_matches_uncached():
    """_fft_family_block with the cached sorted spectrum == recomputing it every call."""
    import ml_features as mf
    from guesses import InitialGuess
    rng = np.random.default_rng(4)
    freq = np.linspace(0.01, 5, 3000)
    power = np.abs(rng.normal(size=3000)) ** 2
    g = InitialGuess(P0=1.0, method="acf_fft", info=dict(fft_freq=freq, fft_power=power))
    for P in rng.uniform(0.2, 90, 20):
        a = mf._fft_family_block(g, "fft", P, 0.1)
        mf._SPECTRUM_CACHE._d.clear()
        b = mf._fft_family_block(g, "fft", P, 0.1)
        assert repr(a) == repr(b)
        periods = 1.0 / freq
        order = np.argsort(periods)
        ref = _nearest_peak_stats_ref(periods[order], power[order], P)
        assert a[0]["fft_nearest_peak_rank"] == ref["rank"]
        assert a[3] == ref["raw_value"] / float(np.median(power[order]))


def test_identity_cache_lru_and_identity():
    """_IdentityCache: hit only for the same objects; LRU eviction."""
    import ml_features as mf
    c = mf._IdentityCache(maxsize=2)
    calls = []
    f = lambda a: calls.append(1) or float(a.sum())  # noqa: E731
    a, b, d = np.ones(3), np.zeros(3), np.full(3, 2.0)
    assert c.get_or_compute((a,), f) == 3.0 and c.get_or_compute((a,), f) == 3.0 and len(calls) == 1
    assert c.get_or_compute((a.copy(),), f) == 3.0 and len(calls) == 2  # equal values, other object
    c.get_or_compute((b,), f)
    c.get_or_compute((a,), f)  # refresh a
    c.get_or_compute((d,), f)  # evicts b (a was used more recently)
    n = len(calls)
    c.get_or_compute((a,), f)
    assert len(calls) == n
    c.get_or_compute((b,), f)
    assert len(calls) == n + 1

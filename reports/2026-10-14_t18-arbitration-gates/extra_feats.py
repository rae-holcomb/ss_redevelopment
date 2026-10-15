"""extra_feats.py -- offline per-candidate statistics computed from cached ACFs
(no refit): the half-lag contrast's per-term t-statistic and an ACF-noise-scaled
contrast. Adds columns to a copy of cands.csv -> cands_x.csv."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))
from alias_metrics import acf_window_max  # noqa: E402


def hlc_terms(lags, acf, P, n_max=6, win_frac=0.1):
    """The individual h(kP) - h((k-1/2)P) terms of acf_half_lag_contrast."""
    w = win_frac * P
    h = {m: acf_window_max(lags, acf, m * P, w) for m in np.arange(0.5, n_max + 0.5, 0.5)}
    return np.array([h[k] - h[k - 0.5] for k in range(1, n_max + 1) if np.isfinite(h[k]) and np.isfinite(h[k - 0.5])])


def run(cache):
    """Write cands_x.csv with hlc_n, hlc_std, hlc_t, acf_noise, hlc_over_noise."""
    cache = Path(cache)
    c = pd.read_csv(cache / "cands.csv")
    out = []
    cur, la = None, None
    for r in c.itertuples():
        if r.star_id != cur:
            z = np.load(cache / "acf" / f"{r.star_id}.npz"); la, ac = z["lags"], z["acf"]; cur = r.star_id
            # ACF noise scale: robust std of the ACF's first difference over the outer half of lags
            # (white-ish small-scale jitter), / sqrt(2)
            d = np.diff(ac[len(ac) // 2:])
            noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
        t = hlc_terms(la, ac, r.P)
        n = len(t)
        sd = float(np.std(t, ddof=1)) if n > 1 else np.nan
        out.append(dict(hlc_n=n, hlc_std=sd, hlc_t=float(t.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
                        acf_noise=noise, hlc_min_term=float(t.min()) if n else np.nan,
                        hlc_frac_pos=float((t > 0).mean()) if n else np.nan))
    c = pd.concat([c, pd.DataFrame(out)], axis=1)
    c.to_csv(cache / "cands_x.csv", index=False)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        run(d)

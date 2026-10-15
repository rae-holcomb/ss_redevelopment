"""
arbitration.py -- offline replay of fit_rotation_period's gating + selection
on a diag_cache.py cache, so candidate arbitration rules can be compared in
seconds instead of re-running the comb fit.

A rule is a dict; `replay(cache_dir, rule)` returns one row per star with the
selected period and its outcome. `rule={}` reproduces the current pipeline
(verified against the cached final outcomes by `check_reproduces`).

Rule keys (all optional)
------------------------
    hlc_min            extra gate: acf_half_lag_contrast >= hlc_min (NaN fails)
    hlc_min_short      like hlc_min but only for candidates with P < short_P
    short_P            threshold for hlc_min_short (days)
    few_h_min          extra gate on few_peaks candidates: mean fitted comb height >= few_h_min
    few_hlc_min        extra gate on few_peaks candidates: hlc >= few_hlc_min
    few_pdm_max        extra gate on few_peaks candidates: PDM theta <= few_pdm_max
    snr_min            replaces the mean-height-SNR gate threshold (default 1.0)
    pos_min            replaces the frac_positive_heights threshold (0.8)
    vertex_min         replaces the frac_vertex_in_window threshold (0.8)
    drop_gate          name of one base gate to ignore: 'g_peaks','g_pos','g_snr','g_vertex'
    select             'redchi' (default) | 'hlc' (max contrast) | 'redchi_over_methods'
    guard              'ratio' (default, half-tooth ratio) | 'none' | 'fold' (fold_half_corr) |
                       'hlc' (compare contrasts)
    guard_thr          threshold for the ratio guard (0.7)
    fold_thr           fold_half_corr threshold for 'fold' guard: long wins if short's
                       fold_half_corr_at_2P < fold_thr ... see _guard
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))

import evaluation as ev  # noqa: E402
from comb_fit import _half_tooth_height_ratio  # noqa: E402

BASE_GATES = ("g_peaks", "g_pos", "g_snr", "g_vertex")


@lru_cache(maxsize=8)
def load(cache_dir: str):
    """Load (stars, cands) DataFrames from a diag_cache output dir."""
    d = Path(cache_dir)
    stars = pd.read_csv(d / "stars.csv")
    # cands_x.csv = cands.csv + extra_feats.py columns, when available
    cands = pd.read_csv(d / ("cands_x.csv" if (d / "cands_x.csv").exists() else "cands.csv"))
    return stars, cands


@lru_cache(maxsize=4096)
def _acf(cache_dir: str, sid: str):
    """Cached (lags, acf) for one star."""
    z = np.load(Path(cache_dir) / "acf" / f"{sid}.npz")
    return z["lags"], z["acf"]


def gate_mask(c: pd.DataFrame, rule: dict) -> np.ndarray:
    """Boolean pass mask for candidate rows `c` under `rule`."""
    g = {
        "g_peaks": c.g_peaks.to_numpy(bool),
        "g_pos": (c.frac_pos >= rule.get("pos_min", 0.8)).to_numpy(),
        "g_snr": (c.mean_hsnr.fillna(-np.inf) >= rule.get("snr_min", 1.0)).to_numpy(),
        "g_vertex": (c.frac_vertex >= rule.get("vertex_min", 0.8)).to_numpy(),
    }
    drop = rule.get("drop_gate")
    m = np.ones(len(c), bool)
    for k, v in g.items():
        if k != drop:
            m &= v
    hlc = c.acf_half_lag_contrast.to_numpy(float)
    if "hlc_min" in rule:
        m &= np.nan_to_num(hlc, nan=-np.inf) >= rule["hlc_min"]
    if "hlc_min_short" in rule:
        short = c.P.to_numpy() < rule.get("short_P", 1.0)
        m &= ~short | (np.nan_to_num(hlc, nan=-np.inf) >= rule["hlc_min_short"])
    few = c.few_peaks.to_numpy(bool)
    if "few_h_min" in rule:
        m &= ~few | (c.mean_h.fillna(-np.inf).to_numpy() >= rule["few_h_min"])
    if "few_acfh_min" in rule:
        m &= ~few | (c.acf_height_at_P.fillna(-np.inf).to_numpy() >= rule["few_acfh_min"])
    if "few_hlc_min" in rule:
        m &= ~few | (np.nan_to_num(hlc, nan=-np.inf) >= rule["few_hlc_min"])
    if "few_pdm_max" in rule:
        m &= ~few | (c.pdm_theta.fillna(np.inf).to_numpy() <= rule["few_pdm_max"])
    if "hlc_allpos" in rule:
        # every h(kP) - h((k-1/2)P) term > 0, OR mean contrast >= hlc_or
        allpos = c.hlc_min_term.fillna(-np.inf).to_numpy() > 0
        strong = np.nan_to_num(hlc, nan=-np.inf) >= rule.get("hlc_or", np.inf)
        m &= allpos | strong
    if "hlc_t_min" in rule:
        m &= c.hlc_t.fillna(-np.inf).to_numpy() >= rule["hlc_t_min"]
    if "min_methods" in rule:
        m &= c.n_methods.to_numpy() >= rule["min_methods"]
    return m


def _order(p: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Sort passing candidates best-first by the rule's selection score."""
    sel = rule.get("select", "redchi")
    rc = p.redchi.replace([np.inf, -np.inf], np.nan).fillna(np.inf)
    if sel == "redchi":
        key = rc
    elif sel == "hlc":
        key = -p.acf_half_lag_contrast.fillna(-np.inf)
    elif sel == "redchi_over_methods":
        key = rc / p.n_methods.clip(lower=1) ** rule.get("method_pow", 1.0)
    else:
        raise ValueError(sel)
    # stable sort on original (redchi) order as a tie-break, like the pipeline
    return p.assign(_k=key.to_numpy()).sort_values(["_k", "idx"], kind="mergesort")


def _guard(passing: pd.DataFrame, rule: dict, cache_dir: str, sid: str):
    """Factor-of-2 guard replay (mirrors comb_fit._resolve_factor2_aliases).

    Returns the selected row."""
    mode = rule.get("guard", "ratio")
    rows = [r for _, r in passing.iterrows()]
    cur = rows[0]
    if mode == "none" or len(rows) < 2:
        return cur
    rel_tol = 0.1
    visited = {cur.idx}
    for _ in range(3):
        partner, cur_short = None, None
        for r in rows:
            if r.idx in visited:
                continue
            q = r.P / cur.P
            if abs(q - 2.0) <= 2.0 * rel_tol:
                partner, cur_short = r, True
                break
            if abs(q - 0.5) <= 0.5 * rel_tol:
                partner, cur_short = r, False
                break
        if partner is None:
            break
        S, L = (cur, partner) if cur_short else (partner, cur)
        if mode == "ratio":
            lags, acf = _acf(cache_dir, sid)
            hr = _half_tooth_height_ratio(lags, acf, S.P, S.t0, L.P, L.t0)
            if not np.isfinite(hr):
                win = cur
            else:
                win = L if hr < rule.get("guard_thr", 0.7) else S
        elif mode == "fold":
            # long wins if the short candidate's 2P fold has clearly
            # different halves (fold_half_corr_at_2P low = double-dip at L)
            v = S.fold_half_corr_at_2P
            if not np.isfinite(v):
                win = cur
            else:
                win = L if v < rule.get("fold_thr", 0.5) else S
        elif mode == "fold_ratio":
            # ratio guard, but ambiguous ratios (guard_thr..guard_hi) go to the fold test
            lags, acf = _acf(cache_dir, sid)
            hr = _half_tooth_height_ratio(lags, acf, S.P, S.t0, L.P, L.t0)
            v = S.fold_half_corr_at_2P
            if not np.isfinite(hr):
                win = cur
            elif hr < rule.get("guard_thr", 0.7):
                win = L
            elif hr < rule.get("guard_hi", 0.9) and np.isfinite(v) and v < rule.get("fold_thr", 0.5):
                win = L
            else:
                win = S
        elif mode == "hlc":
            a, b = S.acf_half_lag_contrast, L.acf_half_lag_contrast
            win = L if (np.nan_to_num(b, nan=-9) > np.nan_to_num(a, nan=-9) * rule.get("hlc_ratio", 1.0)) else S
        else:
            raise ValueError(mode)
        visited.add(partner.idx)
        if win.idx == cur.idx:
            break
        cur = win
    return cur


def replay(cache_dir: str, rule: dict) -> pd.DataFrame:
    """Apply `rule` to every star in the cache; return per-star outcomes."""
    cache_dir = str(cache_dir)
    stars, cands = load(cache_dir)
    groups = dict(tuple(cands.groupby("star_id")))
    out = []
    for s in stars.itertuples():
        c = groups.get(s.star_id)
        P = None
        if c is not None and len(c):
            c = c.sort_values("idx")
            m = gate_mask(c, rule)
            if rule.get("prefer"):
                # soft mode: contrast rule only PREFERS; fall back to the base gates
                base = gate_mask(c, {k: v for k, v in rule.items() if k not in ("hlc_allpos", "hlc_or", "prefer", "fallback_hlc_min")})
                if "fallback_hlc_min" in rule:
                    base &= np.nan_to_num(c.acf_half_lag_contrast.to_numpy(float), nan=-np.inf) > rule["fallback_hlc_min"]
                m = m if m.any() else base
            p = c[m]
            if len(p):
                p = _order(p, rule)
                P = float(_guard(p, rule, cache_dir, s.star_id).P)
        if pd.isna(s.true_period):
            o = "none" if P is None else "false_pos"
        else:
            pr = None if pd.isna(s.period_lo) else (s.period_lo, s.period_hi)
            o = ev.classify_period(P, s.true_period, ev.REL_TOL, pr)
        out.append(dict(star_id=s.star_id, P_sel=P, outcome=o))
    return stars.merge(pd.DataFrame(out), on="star_id")


def summarize(df: pd.DataFrame) -> dict:
    """Headline fractions: success, correct, diffrot, alias, wrong, none (signal),
    and noise false positives (white / red counts)."""
    sig = df[df.kind != "noise"]
    o = sig.outcome
    r = dict(n=len(sig), success=float(o.isin(["correct", "diffrot"]).mean()),
             correct=float((o == "correct").mean()), diffrot=float((o == "diffrot").mean()),
             alias=float(o.isin(["half", "double"]).mean()), wrong=float((o == "wrong").mean()),
             none=float((o == "none").mean()))
    noise = df[df.kind == "noise"]
    if len(noise):
        red = noise.red.astype(str) == "True"
        r["fp_white"] = int((noise[~red].outcome == "false_pos").sum())
        r["fp_red"] = int((noise[red].outcome == "false_pos").sum())
        r["n_noise_each"] = int(red.sum())
    return r


def check_reproduces(cache_dir: str) -> int:
    """Number of stars where replay({}) disagrees with the cached pipeline outcome."""
    df = replay(cache_dir, {})
    fo = df.final_outcome
    bad = (fo != df.outcome)
    return int(bad.sum())

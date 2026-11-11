"""Tiny stdlib-only clients for VizieR TAP and the MAST APIs (P27 scoping).

The ``ss_openblas`` env has no astroquery/pyvo/requests, so these helpers use
``urllib`` only. Every query result is cached as CSV under a cache directory
(default: the worktree's gitignored ``data/p27_cache/``), keyed by a hash of the
query, so re-running a script is free and offline once cached.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
MAST_INVOKE = "https://mast.stsci.edu/api/v0/invoke"
TESSCUT_SECTOR = "https://mast.stsci.edu/tesscut/api/v0.1/sector"
CACHE = os.environ.get("P27_CACHE", os.path.join(os.path.dirname(__file__), "..", "..", "data", "p27_cache"))


def _cache_path(key: str, ext: str = "csv") -> str:
    """Return the cache file path for a query key (sha1 of the key)."""
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, hashlib.sha1(key.encode()).hexdigest()[:16] + "." + ext)


def _post(url: str, data: dict, timeout: int = 300) -> bytes:
    """POST form data and return the raw response body, retrying 3 times."""
    body = urllib.parse.urlencode(data).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=body, headers={"User-Agent": "ss-redevelopment-p27/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - network flakiness; retry then re-raise
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def vizier_tap(adql: str, use_cache: bool = True) -> pd.DataFrame:
    """Run a synchronous ADQL query on VizieR TAP and return a DataFrame (cached)."""
    path = _cache_path("vizier:" + adql)
    if use_cache and os.path.exists(path):
        return pd.read_csv(path)
    raw = _post(VIZIER_TAP, {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql})
    df = pd.read_csv(io.BytesIO(raw))
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return df


ID_COL_RE = re.compile(r"(?i)(gaia|^source$|^tic$|^kic$|epic|dr2|edr3)")


def _id_dtypes(columns) -> dict:
    """Return ``{col: 'Int64'}`` for identifier columns (19-digit Gaia IDs must never pass through float64)."""
    return {c: "Int64" for c in columns if ID_COL_RE.search(str(c))}


VIZIER_ASU_MIRRORS = ("https://vizier.cfa.harvard.edu", "https://vizier.cds.unistra.fr")


def vizier_table(source: str, columns: str = "**", use_cache: bool = True, extra: str = "") -> pd.DataFrame:
    """Download a whole VizieR table via the ASU-TSV interface (cached).

    ``source`` is a table name such as ``J/ApJS/257/46/tablef1``; ``columns`` is
    the ``-out`` list (``**`` = all columns). Tries the CfA mirror first, because
    the CDS TAP/ASU service was unavailable on 2026-11-11 when this was written.
    """
    key = f"asu:{source}:{columns}:{extra}"
    path = _cache_path(key)
    if use_cache and os.path.exists(path):
        cols = pd.read_csv(path, nrows=0).columns
        return pd.read_csv(path, dtype=_id_dtypes(cols))
    q = f"/viz-bin/asu-tsv?-source={urllib.parse.quote(source)}&-out.max=unlimited&-out={urllib.parse.quote(columns)}{extra}"
    last = None
    for host in VIZIER_ASU_MIRRORS * 3:  # mirrors are flaky: three passes
        try:
            with urllib.request.urlopen(host + q, timeout=300) as r:
                txt = r.read().decode("utf-8", "replace")
            if txt.lstrip().startswith("<"):
                last = RuntimeError(f"{host} returned HTML")
                continue
            lines = [ln for ln in txt.splitlines() if ln and not ln.startswith("#")]
            if len(lines) < 4 or "\t" not in lines[0]:
                last = RuntimeError(f"{host}: unexpected ASU payload: {lines[:1]}")
                continue
            # ASU-TSV: header, units, dashes, then data rows (blank-line separated tables are not expected here)
            header = lines[0].split("\t")
            rows = [ln.split("\t") for ln in lines[3:] if not ln.startswith("---")]
            df = pd.DataFrame(rows, columns=header[: len(rows[0])] if rows else header)
            df = df.apply(lambda s: s.str.strip())
            ids = _id_dtypes(df.columns)
            for c in df.columns:
                if c in ids:
                    df[c] = df[c].replace("", pd.NA).astype("Int64")
                    continue
                conv = pd.to_numeric(df[c], errors="coerce")
                if conv.notna().sum() >= 0.9 * max(1, (df[c] != "").sum()):
                    df[c] = conv
            tmp = path + ".tmp"
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)
            return df
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5)
    raise RuntimeError(f"VizieR ASU failed for {source}: {last}")


def vizier_in_query(select_from: str, col: str, ids, chunk: int = 400, extra_where: str = "") -> pd.DataFrame:
    """Run ``SELECT ... WHERE col IN (ids)`` in chunks and concatenate the results."""
    ids = list(ids)
    out = []
    for i in range(0, len(ids), chunk):
        sub = ",".join(str(int(x)) for x in ids[i:i + chunk])
        w = f" AND {extra_where}" if extra_where else ""
        out.append(vizier_tap(f"{select_from} WHERE {col} IN ({sub}){w}"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def mast_invoke(request: dict, use_cache: bool = True) -> dict:
    """Call the MAST ``invoke`` API with a request dict and return parsed JSON (cached)."""
    key = "mast:" + json.dumps(request, sort_keys=True)
    path = _cache_path(key, "json")
    if use_cache and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    raw = _post(MAST_INVOKE, {"request": json.dumps(request)})
    js = json.loads(raw)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(js, f)
    os.replace(tmp, path)
    return js


def tesscut_sectors(ra: float, dec: float, use_cache: bool = True) -> list[int]:
    """Return the list of TESS FFI sectors covering (ra, dec) via the TESScut sector API (cached)."""
    key = f"tesscut:{ra:.6f},{dec:.6f}"
    path = _cache_path(key, "json")
    if use_cache and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    url = f"{TESSCUT_SECTOR}?ra={ra}&dec={dec}&radius=0m"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                js = json.loads(r.read())
            break
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(3)
    sectors = sorted({int(row["sector"]) for row in js.get("results", [])})
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sectors, f)
    os.replace(tmp, path)
    return sectors

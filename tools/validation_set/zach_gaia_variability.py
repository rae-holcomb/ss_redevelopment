"""Cross-match Zach Claytor's TIC list to Gaia DR3 variability (P27; tests his "oscillators hiding in there").

The ESA Gaia archive was rate-limiting (HTTP 429) and CDS VizieR/XMatch were down on 2026-11-13, so this pulls the
Gaia DR3 variability tables from the CfA VizieR mirror (catalogue I/358), restricted to a 12 deg cone around the south
ecliptic pole (which contains the whole SCVZ, ecliptic latitude <= -78 deg), and matches them to the TIC positions
within 2 arcsec:
  * I/358/vclassre: classifier best class + score (all Gaia DR3 sources processed as variable and classified),
  * I/358/vmsosc:   main-sequence oscillator (delta Sct / gamma Dor / SX Phe) specific-object table,
  * I/358/vrm:      rotation-modulation specific-object table (with Prot),
  * I/358/veb:      eclipsing-binary specific-object table.
Also flags stars within 8 deg of the LMC centre (blend risk; cf. T20's LMC-contaminated noise sources).

Writes reports/2026-11-12_validation-set-scoping/data/zach_gaia_variability.csv.gz (one row per TIC) and prints counts.
"""
import os
import sys

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

sys.path.insert(0, os.path.dirname(__file__))
import vo  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "reports", "2026-11-12_validation-set-scoping", "data", "zach_gaia_variability.csv.gz")
SEP = "&-c=90.0%20-66.5607&-c.rd=12"   # south ecliptic pole, 12 deg radius
LMC = SkyCoord(80.894 * u.deg, -69.756 * u.deg)
TABLES = {"vclassre": "Source,RA_ICRS,DE_ICRS,Class,ClassSc", "vmsosc": "Source,RA_ICRS,DE_ICRS,Freq1",
          "vrm": "Source,RA_ICRS,DE_ICRS,Prot", "veb": "Source,RA_ICRS,DE_ICRS,Freq"}


def match(z: SkyCoord, t: pd.DataFrame, radius_arcsec: float = 2.0):
    """Return (index into t or -1, separation arcsec) for each Zach star; positional nearest neighbour."""
    if len(t) == 0:
        return np.full(len(z), -1), np.full(len(z), np.nan)
    c = SkyCoord(t["RA_ICRS"].to_numpy() * u.deg, t["DE_ICRS"].to_numpy() * u.deg)
    idx, d2, _ = z.match_to_catalog_sky(c)
    ok = d2.arcsec < radius_arcsec
    return np.where(ok, idx, -1), d2.arcsec


def main():
    """Download the cone tables (cached), match, summarise, save."""
    z = pd.read_csv(os.path.join(ROOT, "data", "p27_cache", "zach_tic.csv"), dtype={"GAIA": str},
                    usecols=["ID", "GAIA", "ra", "dec", "Tmag", "Teff", "contratio"])
    zc = SkyCoord(z["ra"].to_numpy() * u.deg, z["dec"].to_numpy() * u.deg)
    z["sep_lmc_deg"] = zc.separation(LMC).deg
    for name, cols in TABLES.items():
        t = vo.vizier_table(f"I/358/{name}", cols, extra=SEP)
        print(name, "rows in SCVZ cone:", len(t))
        idx, sep = match(zc, t)
        hit = idx >= 0
        z[f"in_{name}"] = hit.astype(int)
        if name == "vclassre":
            z["gaia_class"] = np.where(hit, t["Class"].to_numpy()[np.clip(idx, 0, None)], None)
            z["gaia_class_score"] = np.where(hit, t["ClassSc"].to_numpy()[np.clip(idx, 0, None)], np.nan)
        if name == "vrm":
            z["gaia_prot"] = np.where(hit, t["Prot"].to_numpy()[np.clip(idx, 0, None)], np.nan)
    z.drop(columns=["GAIA"]).to_csv(OUT, index=False)
    n = len(z)
    print(n, "TICs")
    print("any Gaia DR3 variability-table match:", int((z.filter(like="in_").sum(axis=1) > 0).sum()))
    print("classifier classes:", z["gaia_class"].value_counts().to_dict())
    print("ms_oscillator / rot_mod / EB:", int(z.in_vmsosc.sum()), int(z.in_vrm.sum()), int(z.in_veb.sum()))
    print("within 8 deg of LMC:", int((z.sep_lmc_deg < 8).sum()))


if __name__ == "__main__":
    main()

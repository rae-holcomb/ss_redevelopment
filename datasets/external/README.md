# External input lists

Lists supplied from outside the project. They are inputs for building manifests; they are not frozen manifests.

| File | What | Provenance |
|---|---|---|
| `claytor2025_nonrotator_ticids.txt` | 23,332 unique TIC IDs, one per line (range 25063308–765470988). **Not vetted non-rotators.** They are the Claytor+2024 (ApJ 962, 47, §III.1.2) southern-CVZ noise-template parent sample: every ID has TIC v8.2 Teff 5800–6000 K, `lumclass = DWARF`, ecliptic latitude ≤ −78°, Tmag 6–15 (P27 check, 2026-11-12). Claytor+2024 kept 22,090 of these with ≥ 4 TASOC sectors as "comparatively quiescent" noise templates. In Kepler, 14% of 5800–6000 K dwarfs have a detected period (McQuillan+2014). | From Zach Claytor via Rae, 2026-11-12. Original filename `smarts_star_ticids.txt`; the filename keeps the old "nonrotator" wording. **Use label `expected_undetectable`** (Rae, 2026-11-12: "use them, but remain appropriately wary"): report their *detection rate*, never count detections as false positives. See `reports/2026-11-12_validation-set-scoping/`. |

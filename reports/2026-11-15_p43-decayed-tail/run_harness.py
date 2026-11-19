"""CLI: python run_harness.py <mode> <out.pkl> <variant,variant,...> [n_workers]  (dev_medium, P43 cache)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p43lib as P  # noqa: E402

if __name__ == "__main__":
    mode, out, vs = sys.argv[1:4]
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    recs = P.run_p43_harness(mode, tuple(vs.split(",")), out, n_workers=nw)
    for v in vs.split(","):
        print(P.L.summarize(recs, v))

"""CLI for the T32 harness: python run_harness.py <manifest> <mode> <out.pkl> <variant,variant,...> [n_workers]."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t32lib as L

if __name__ == "__main__":
    man, mode, out, vs = sys.argv[1:5]
    nw = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    recs = L.run_harness(mode, tuple(vs.split(",")), out, n_workers=nw, manifest=man)
    for v in vs.split(","):
        print(L.summarize(recs, v))

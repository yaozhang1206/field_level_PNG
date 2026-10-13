"""Done-checks for the interactive queue (exit 0 = finished).
    python scripts/check_done.py stage3 <results json> <lo-hi>     all realisations lo..hi present
    python scripts/check_done.py summary <results json>             the JSON has a "summary" entry
    python scripts/check_done.py chain <results json>                a sampler run finished (has "summary")"""
import json, sys
kind, path = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception:
    sys.exit(1)
if kind == "stage3":
    lo, hi = map(int, sys.argv[3].split("-"))
    sys.exit(0 if all(str(i) in d.get("runs", {}) for i in range(lo, hi + 1)) else 1)
sys.exit(0 if ("summary" in d and not d.get("partial")) else 1)

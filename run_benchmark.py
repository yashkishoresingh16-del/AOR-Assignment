"""Run the full pipeline on every instance and save the best tours to results.json."""
import json, sys, os
from tsp_core import load_tsp, solve
BEST = {"a280":2579,"d198":15780,"d493":35002,"fl417":11861,"lin318":42029,
        "pcb442":50778,"pr152":73682,"pr226":80369,"pr439":107217,"ts225":126643}
tl = float(sys.argv[1]) if len(sys.argv) > 1 else 20
names = sys.argv[2:] or sorted(BEST)
out = {}
for name in names:
    info = load_tsp(f"data/{name}.tsp")
    r = solve(info["coords"], time_limit=tl, seed=1)
    out[name] = {"n": len(info["coords"]), "nn": r["nn_length"], "ls": r["ls_length"],
                 "final": r["length"], "best_known": BEST[name], "tour": r["tour"]}
    print(name, r["nn_length"], r["ls_length"], r["length"], flush=True)
json.dump(out, open(f"res_{names[0]}.json", "w"))

"""
Benchmark: the shift scheduling instances of Curtois & Qu (2014), schedulingbenchmarks.org, against the best known
rosters (all proven optimal except Instance24).

    python benchmarks/run_benchmark.py              # instances 1-19 (2 to 12 weeks), 120 s each, 2 at a time
    python benchmarks/run_benchmark.py 20 21 --seconds 600

Every roster is re-checked by the independent checker; penalties are recomputed from the roster, not taken from
the solver.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from roster import data  # noqa: E402
from roster.problem import check, penalty  # noqa: E402
from roster.solver import solve  # noqa: E402

MARKS = [10, 30, 60, 120]


def run(args):
    name, seconds = args
    inst = data.benchmark(name)
    t0 = time.time()
    r = solve(inst, seconds=seconds, seed=0)
    problems = check(inst, r.roster)
    pen = penalty(inst, r.roster)
    best = data.BEST[name]
    at = {}
    for m in MARKS:
        pts = [h for h in r.history if h[0] <= m]
        at[str(m)] = pts[-1][1] if pts else None
    return {"name": name, "staff": inst.E, "days": inst.horizon, "shift_types": inst.S, "best": best,
            "lower_bound": data.LOWER[name], "penalty": pen["total"], "valid": not problems,
            "gap": pen["total"] / best - 1, "first": r.first_penalty, "status": r.status, "seconds": time.time() - t0,
            "iterations": r.iterations, "at": at, "breakdown": {k: v for k, v in pen.items() if k != "short"},
            "curve": [[round(t, 1), p] for t, p in r.history]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instances", nargs="*", type=int)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results.json"))
    a = ap.parse_args()
    nums = a.instances or list(range(1, 20))
    names = [f"Instance{k}" for k in nums]
    t0 = time.time()
    rows = []
    with Pool(2) as pool:
        for r in pool.imap(run, [(n, a.seconds) for n in names]):
            rows.append(r)
            print(f"  {r['name']:10s} {r['staff']:3d} staff {r['days']:3d} days  penalty {r['penalty']:7.0f}  best known "
                  f"{r['best']:6d}  gap {100 * r['gap']:6.2f}%  first {r['first']:8.0f}  {r['status']:8s} valid {r['valid']}",
                  flush=True)
    old = json.load(open(a.out)) if os.path.exists(a.out) else {"instances": []}
    keep = [r for r in old.get("instances", []) if r["name"] not in names]
    res = {"instances": sorted(keep + rows, key=lambda r: int(r["name"][8:])), "seconds_per_instance": a.seconds,
           "total_seconds": time.time() - t0, "machine": f"{os.cpu_count()} cores, one search thread per instance"}
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

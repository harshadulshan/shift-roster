"""
Mutation test: do the engine checks notice when the code is wrong?

Each mutant copies the package, plants one bug (a rule the checker forgets, a constraint the solver drops, a sick
call that rewrites the past ...) and runs `verify_engine.py --quick` against the copy. Every mutant must be caught.

    python tests/mutation_test.py              # about 15 minutes
    python tests/mutation_test.py weekend      # only the mutants whose name contains "weekend"
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MUTANTS = [
    ("checker ignores rest between shifts", "problem.py",
     "if row[d] != OFF and row[d + 1] != OFF and inst.shifts[row[d + 1]].id in inst.shifts[row[d]].cannot_follow:",
     "if False:"),
    ("checker checks runs at the period start", "problem.py",
     "if length < emp.min_consec and start > 0 and start + length < H:",
     "if length < emp.min_consec and start + length < H:"),
    ("checker forgets Sundays in weekends", "problem.py",
     "any(d is not None and row[d] != OFF for d in (sat, sun))", "(sat is not None and row[sat] != OFF)"),
    ("checker ignores minimum hours", "problem.py", "if minutes < emp.min_minutes:", "if False:"),
    ("penalty forgets extra people on shift", "problem.py", "over += wo * (n - req)", "over += 0"),
    ("solver drops rest between shifts", "solver.py", "                    m.Add(a + sum(terms) <= 1)",
     "                    pass"),
    ("solver's minimum runs one day short", "solver.py",
     "            for k in range(1, emp.min_consec):\n                for d in range(1, H - k):\n                    if fd",
     "            for k in range(1, emp.min_consec - 1):\n                for d in range(1, H - k):\n                    if fd"),
    ("solver drops the weekend limit", "solver.py", "                m.Add(sum(ks) <= emp.max_weekends)",
     "                pass"),
    ("solver ignores day-off requests", "solver.py", "                terms.append(wt * self.x[e, d, s])",
     "                terms.append(0)"),
    ("sick call rewrites days already worked", "changes.py", "fixed[:, :today - start] = True",
     "fixed[:, :0] = True"),
    ("sick call forgets the last day away", "changes.py",
     "update(range(max(a, today), min(b, H - 1) + 1))", "update(range(max(a, today), min(b, H - 1)))"),
    ("leave not counted towards the contract", "ward.py",
     "credit = 8 * 60 * len(leave_days[k]) if w.leave_credit else 0", "credit = 0"),
    ("by-hand baseline ignores the weekend limit", "report.py",
     "if d in wk_of and wk_of[d] not in worked_wk[e] and len(worked_wk[e]) >= emp.max_weekends:", "if False:"),
]


def main():
    only = sys.argv[1:]
    todo = [m for m in MUTANTS if not only or any(w in m[0] for w in only)]
    caught = 0
    for name, fname, old, new in todo:
        tmp = tempfile.mkdtemp(prefix="mutant_")
        try:
            shutil.copytree(os.path.join(ROOT, "roster"), os.path.join(tmp, "roster"),
                            ignore=shutil.ignore_patterns("__pycache__"))
            os.symlink(os.path.join(ROOT, "data_cache"), os.path.join(tmp, "data_cache"))
            path = os.path.join(tmp, "roster", fname)
            src = open(path).read()
            assert src.count(old) == 1, f"{name}: the text to mutate is not found exactly once"
            open(path, "w").write(src.replace(old, new))
            try:
                res = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tests", "verify_engine.py"), "--quick"],
                                     env=dict(os.environ, RS_ROOT=tmp), capture_output=True, text=True, timeout=1200)
            except subprocess.TimeoutExpired:
                print(f"  MISSED  {name:44s}  (checks did not finish)", flush=True)
                continue
            fails = [l.strip()[6:] for l in res.stdout.splitlines() if l.strip().startswith("FAIL")]
            hit = res.returncode != 0
            caught += hit
            why = fails[0] if fails else (res.stderr.strip().splitlines() or ["crash"])[-1]
            print(f"  {'CAUGHT' if hit else 'MISSED'}  {name:44s}  {why[:90] if hit else ''}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{caught} of {len(todo)} mutants caught")
    sys.exit(0 if caught == len(todo) else 1)


if __name__ == "__main__":
    main()

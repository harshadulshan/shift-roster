"""
Engine checks: the reader, the independent checker rule by rule, the solver against brute force on small rosters
and against the published optima, sick-call repairs, the Excel ward format and the by-hand baseline.

    python tests/verify_engine.py            # about 3 minutes
    python tests/verify_engine.py --quick    # the checks the mutation test needs
"""

from __future__ import annotations

import copy
import io
import itertools
import json
import os
import sys

ROOT = os.environ.get("RS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from roster import changes, data, report, ward  # noqa: E402
from roster.problem import OFF, Employee, Instance, Shift, check, check_employee, penalty  # noqa: E402
from roster.solver import solve  # noqa: E402

QUICK = "--quick" in sys.argv
passed, failed = 0, []


def ok(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}  {detail}")


def tiny(seed, E=3, H=7, S=2):
    """A small random roster problem that brute force can solve exactly."""
    rng = np.random.default_rng(seed)
    shifts = [Shift("D", 480), Shift("N", 600, {"D"})][:S]
    staff = []
    for e in range(E):
        staff.append(Employee(f"E{e}", {"D": 7, "N": int(rng.integers(1, 4))}, int(rng.integers(4, 6)) * 540,
                              int(rng.integers(2, 4)) * 480, int(rng.integers(3, 5)), 2, int(rng.integers(1, 3)), 1))
    days_off = {e: {int(rng.integers(H))} for e in range(E)}
    on = [(int(rng.integers(E)), int(rng.integers(H)), int(rng.integers(S)), int(rng.integers(1, 4))) for _ in range(3)]
    off = [(int(rng.integers(E)), int(rng.integers(H)), int(rng.integers(S)), int(rng.integers(1, 4))) for _ in range(3)]
    cover = {(d, s): (int(rng.integers(0, 3)), 100, 1) for d in range(H) for s in range(S)}
    return Instance(f"tiny{seed}", H, shifts, staff, days_off, on, off, cover)


def brute(inst):
    """Exact optimum: every valid row of every employee (checked by the checker), every combination."""
    E, H, S = inst.E, inst.horizon, inst.S
    rows = []
    for e in range(E):
        valid = [np.array(r) for r in itertools.product(range(-1, S), repeat=H) if not check_employee(inst, e, np.array(r))]
        rows.append(valid)
    if any(len(r) == 0 for r in rows) or np.prod([len(r) for r in rows], dtype=float) > 3e5:
        return None                      # infeasible, or too many combinations to enumerate quickly
    best, arg = np.inf, None
    for combo in itertools.product(*[range(len(r)) for r in rows]):
        R = np.array([rows[e][k] for e, k in enumerate(combo)])
        p = penalty(inst, R)["total"]
        if p < best:
            best, arg = p, R
    return best, arg


def roster_from(text_rows, inst):
    ids = {s.id: k for k, s in enumerate(inst.shifts)}
    return np.array([[ids[c] if c != "." else OFF for c in row] for row in text_rows])


print("1 · Reading the benchmark format")
i1 = data.benchmark("Instance1")
ok("Instance1: 8 staff, 14 days, 1 shift type, 14 cover rows", (i1.E, i1.horizon, i1.S, len(i1.cover)) == (8, 14, 1, 14))
ok("its contracts: max 14 D, 4320/3360 minutes, runs 5/2/2, 1 weekend",
   vars(i1.staff[0])["max_shifts"] == {"D": 14} and (i1.staff[0].max_minutes, i1.staff[0].min_minutes,
                                                     i1.staff[0].max_consec, i1.staff[0].min_consec,
                                                     i1.staff[0].min_consec_off, i1.staff[0].max_weekends)
   == (4320, 3360, 5, 2, 2, 1))
ok("requests and leave read (21 on, 5 off, 8 days off)", (len(i1.on_requests), len(i1.off_requests),
                                                          sum(len(v) for v in i1.days_off.values())) == (21, 5, 8))
i8 = data.benchmark("Instance8")
ok("shift rotation read (Instance8: some shift types may not follow others)",
   any(s.cannot_follow for s in i8.shifts) and all(t in {x.id for x in i8.shifts} for s in i8.shifts for t in s.cannot_follow))

print("2 · Checker, rule by rule")
T = Instance("t", 14, [Shift("D", 480), Shift("N", 480, {"D"})],
             [Employee("A", {"D": 14, "N": 2}, 8 * 480, 3 * 480, 4, 2, 2, 1)], {0: {13}})
base = "DD..DD..DD...."
ok("a valid row passes", not check(T, roster_from([base], T)), check(T, roster_from([base], T)))
cases = {"rest between shifts (D after N)": ("NDD.DD..DD....", " after "),
         "max shifts of a type (3 N, max 2)": ("NN..NN..DD....", "N shifts"),
         "max hours (9 shifts, max 8)": ("DDD.DDD.DDD...", "at most 3840"),
         "min hours (2 shifts, min 3)": ("DD............", "at least 1440"),
         "max days in a row (5, max 4)": ("DDDDD..DD.....", "5 working days in a row"),
         "min run of working days (single day)": ("DD..D..DD.....", "only 1 working days"),
         "min run of days off (single day)": ("DD.DD..DD.....", "only 1 days off"),
         "max weekends (2, max 1)": ("..DD.DD....DD.", "weekends"),
         "agreed day off (day 13)": ("DD..DD.......D", "day off")}
for label, (row, needle) in cases.items():
    pr = check(T, roster_from([row], T))
    ok(f"catches: {label}", any(needle in p for p in pr), pr)
Tb = copy.deepcopy(T)
Tb.days_off, Tb.staff[0].max_weekends = {0: set()}, 2
ok("runs touching the start or end of the period are not checked (nothing known beyond it)",
   not check(Tb, roster_from(["D...DD..DD...D"], Tb)), check(Tb, roster_from(["D...DD..DD...D"], Tb)))
Ts = copy.deepcopy(T)
Ts.days_off = {0: set()}
pr = check(Ts, roster_from(["......DD.....D"], Ts))          # two weekends worked, both only on the Sunday
ok("catches: a weekend worked only on the Sunday counts", any("weekends" in p for p in pr), pr)
T2 = copy.deepcopy(T)
T2.first_weekday = 6                     # day 0 is a Sunday: days 0, 6, 7, 13 are the weekend days
ok("weekends follow the start weekday", [w for w in T2.weekends()] == [(None, 0), (6, 7), (13, None)])
allD = np.zeros((i1.E, i1.horizon), dtype=int)                    # everyone on D every day
pe = penalty(i1, allD)
exp_over = sum(max(0, i1.E - r) * wo for (d, s), (r, wu, wo) in i1.cover.items())
exp_under = sum(max(0, r - i1.E) * wu for (d, s), (r, wu, wo) in i1.cover.items())
exp_off = sum(w for e, d, s, w in i1.off_requests)
ok(f"penalty by hand: everyone on every day gives over {exp_over}, off-requests {exp_off}",
   (pe["cover over"], pe["cover under"], pe["shift off requests"], pe["shift on requests"]) == (exp_over, exp_under, exp_off, 0)
   and exp_over > 0, pe)
pen = penalty(i1, roster_from(["D" * 5 + ".." + "D" * 5 + ".."] * 8, i1))
ok("penalty: cover short/over and requests recomputed", pen["total"] == pen["shift on requests"] + pen["shift off requests"]
   + pen["cover under"] + pen["cover over"] and pen["cover under"] % 100 == 0)

print("3 · Solver = brute force on small rosters")
bad_whole, bad_lns, tried = [], [], 0
for seed in range(10 if QUICK else 18):
    inst = tiny(seed)
    bf = brute(inst)
    if bf is None:
        continue
    tried += 1
    r = solve(inst, seconds=3)
    if check(inst, r.roster) or abs(penalty(inst, r.roster)["total"] - bf[0]) > 1e-9:
        bad_whole.append((seed, r.penalty, bf[0]))
    r2 = solve(inst, seconds=4, seed=seed, whole_limit=0)          # neighbourhood search only
    if check(inst, r2.roster) or abs(penalty(inst, r2.roster)["total"] - bf[0]) > 1e-9:
        bad_lns.append((seed, r2.penalty, bf[0]))
ok(f"whole-problem solve finds the exact optimum ({tried} rosters)", tried >= 5 and not bad_whole, bad_whole[:3])
ok("neighbourhood search alone finds the exact optimum too", not bad_lns, bad_lns[:3])

print("4 · Published optima (Curtois & Qu)")
for nm, secs in ([("Instance1", 10), ("Instance2", 20)] if QUICK else [("Instance1", 10), ("Instance2", 20),
                                                                      ("Instance3", 40), ("Instance4", 60)]):
    inst = data.benchmark(nm)
    r = solve(inst, seconds=secs)
    p = penalty(inst, r.roster)["total"]
    ok(f"{nm}: penalty {p} = proven optimum {data.BEST[nm]}, valid", p == data.BEST[nm] and not check(inst, r.roster))

print("5 · The demo ward and its Excel workbook")
w = ward.demo()
inst = ward.to_instance(w)
ok("18 staff, 3 shift types, 28 days, starts on a Monday", (inst.E, inst.S, inst.horizon, inst.first_weekday) == (18, 3, 28, 0))
ok("Night may not be followed by Day or Evening, Evening not by Day",
   inst.shifts[2].cannot_follow == {"D", "E"} and inst.shifts[1].cannot_follow == {"D"})
e5 = list(w.staff.name).index(w.leave.name[0])
ok("leave becomes days off and counts 8 h a day towards the contract",
   len(inst.days_off[e5]) == 5 and inst.staff[e5].max_minutes == 160 * 60 - 5 * 480)
w2 = ward.from_excel(ward.to_excel(w))
i2 = ward.to_instance(w2)
ok("Excel round trip gives the same problem", i2.cover == inst.cover and i2.on_requests == inst.on_requests and
   i2.off_requests == inst.off_requests and i2.days_off == inst.days_off and
   [vars(a) for a in i2.staff] == [vars(a) for a in inst.staff])
bad = copy.deepcopy(w)
bad.leave.loc[0, "name"] = "Nobody"
try:
    ward.to_instance(bad)
    ok("leave for an unknown person is refused", False)
except ValueError:
    ok("leave for an unknown person is refused", True)
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as x:
    w.staff.to_excel(x, sheet_name="Staff", index=False)
try:
    ward.from_excel(buf.getvalue())
    ok("a workbook without the Shifts sheet is refused", False)
except ValueError:
    ok("a workbook without the Shifts sheet is refused", True)
r = solve(inst, seconds=8 if QUICK else 30, seed=0)
S_ = report.summary(inst, r.roster)
G = report.greedy(inst)
SG = report.summary(inst, G)
ok(f"demo roster keeps every rule ({S_['short']} short, {S_['granted']}/{S_['requests']} requests)", not S_["problems"])
ok(f"by hand breaks rules ({len(SG['problems'])}) and is shorter on cover ({SG['short']} short)",
   len(SG["problems"]) > 0 and SG["short"] > S_["short"])
gw = all(sum(1 for a, b in inst.weekends() if any(x is not None and G[e, x] != OFF for x in (a, b))) <= inst.staff[e].max_weekends
         and not any(G[e, d] != OFF for d in inst.days_off[e]) for e in range(inst.E))
ok("by hand still respects leave and weekend limits (what it claims)", gw)
xl = pd.ExcelFile(io.BytesIO(report.roster_excel(inst, r.roster)))
ok("Excel roster: Roster, Hours and Requests sheets", xl.sheet_names == ["Roster", "Hours", "Requests"])
ps = report.person_stats(inst, r.roster)
ok("hours per person inside every contract", bool(((ps["hours"] >= ps["min hours"] - 1e-9) &
                                                    (ps["hours"] <= ps["max hours"] + 1e-9)).all()))

print("6 · Sick-call repairs")
R = r.roster
e, today = 3, 9
out = changes.reroster(inst, R, today, absences=[(e, 9, 11)], seconds=6, keep_weight=10)
new = out["roster"]
ok("repaired part keeps every rule", not out["problems"], out["problems"][:2])
ok("days before today unchanged", bool((new[:, :today] == R[:, :today]).all()))
ok("the absent person is off on the absent days", bool((new[e, 9:12] == OFF).all()))
nothing = R.copy()
nothing[e, 9:12] = OFF
short = lambda X: int(report.cover_table(out["sub"], X[:, out["start"]:]).query(f"day >= {today - out['start']}").short.sum())
ok(f"no more shifts short than doing nothing ({short(new)} vs {short(nothing)})", short(new) <= short(nothing))
same = changes.reroster(inst, R, today, absences=[], extra_cover=None, seconds=3, keep_weight=1000)
ok("nothing changed -> no shift changes", int(same["changed"].sum()) == 0)
ot = copy.deepcopy(inst)                      # allow 8 h overtime, so someone can take an extra shift
for emp in ot.staff:
    emp.max_minutes += 480
more = changes.reroster(ot, R, 14, extra_cover={(16, 0): 2}, seconds=8, keep_weight=2)
ok(f"extra cover asked for is filled when someone can take it ({int(np.sum(R[:, 16] == 0))} -> "
   f"{int(np.sum(more['roster'][:, 16] == 0))} on Day shift)",
   int(np.sum(more["roster"][:, 16] == 0)) >= int(np.sum(R[:, 16] == 0)) + 1 and not more["problems"])

if not QUICK:
    print("7 · Benchmark results")
    path = os.path.join(ROOT, "benchmarks", "results.json")
    if os.path.exists(path):
        rows = json.load(open(path))["instances"]
        ok(f"all {len(rows)} benchmark rosters valid", len(rows) >= 19 and all(r["valid"] for r in rows))
        ok("penalties never below the proven optimum (would mean a checker bug)",
           all(r["penalty"] >= data.LOWER[r["name"]] for r in rows))
        ok("best known matched on at least 6 instances", sum(r["gap"] < 1e-9 for r in rows) >= 6)
    else:
        ok("benchmarks/results.json exists", False)

print(f"\n{passed} passed, {len(failed)} failed")
if failed:
    print("FAILED: " + "; ".join(failed))
sys.exit(1 if failed else 0)

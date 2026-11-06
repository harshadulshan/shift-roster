"""
Someone calls in sick (or leave is approved late, or more cover is needed) after the roster is published.

The roster is repaired from today on, changing as few published shifts as possible: every change costs a little,
a shift left short costs a lot. Days before today stay as worked. The repaired part is a smaller rostering problem
that starts a few days before today, so that runs of working days already started are respected; the hours,
shifts and weekends already worked are taken off each person's limits. Sick days count as worked for the contract
hours (no one has to make them up), and the minimum-run rules are not checked right next to the absence (as at
the ends of the period).
"""

from __future__ import annotations

import copy
import datetime as dt

import numpy as np

from .problem import OFF, Instance, check
from .solver import lns


def window(inst: Instance, R, today: int, absences=(), extra_cover=None):
    """The problem from (a few days before) today on. Returns (sub-instance, first day, fixed mask, start roster)."""
    H = inst.horizon
    start = max(0, today - min(7, max(e.max_consec for e in inst.staff) + 1))
    if start > 0 and (inst.first_weekday + start) % 7 == 6:       # keep each weekend on one side
        start -= 1
    sub = copy.deepcopy(inst)
    sub.horizon = H - start
    sub.first_weekday = (inst.first_weekday + start) % 7
    if inst.start_date:
        sub.start_date = (dt.date.fromisoformat(inst.start_date) + dt.timedelta(days=start)).isoformat()
    wk_before = [(a, b) for a, b in inst.weekends() if max(x for x in (a, b) if x is not None) < start]
    absent = {}
    for e, a, b in absences:
        absent.setdefault(int(e), set()).update(range(max(a, today), min(b, H - 1) + 1))
    for e, emp in enumerate(sub.staff):
        before = R[e, :start]
        mins = sum(inst.shifts[v].minutes for v in before if v != OFF)
        lost = sum(inst.shifts[R[e, d]].minutes for d in absent.get(e, ()) if R[e, d] != OFF)
        emp.max_minutes = max(0, emp.max_minutes - mins - lost)
        emp.min_minutes = max(0, emp.min_minutes - mins - lost)
        for s, sh in enumerate(inst.shifts):
            emp.max_shifts[sh.id] = max(0, emp.max_shifts.get(sh.id, 10 ** 6) - int(np.sum(before == s)))
        worked = sum(1 for a, b in wk_before if any(x is not None and R[e, x] != OFF for x in (a, b)))
        emp.max_weekends = max(0, emp.max_weekends - worked)
    sub.days_off = {e: {d - start for d in ds if d >= start} | {d - start for d in absent.get(e, ())}
                    for e, ds in inst.days_off.items()}
    sub.breaks = {e: {d - start for d in ds if d >= start} for e, ds in inst.breaks.items()}
    for e, ds in absent.items():                      # minimum runs are not checked next to the absence
        sub.breaks.setdefault(e, set()).update(d - start for d in ds)
    for e in absent:
        sub.days_off.setdefault(e, set()).update(d - start for d in absent[e])
    sub.on_requests = [(e, d - start, s, w) for e, d, s, w in inst.on_requests if d >= today]
    sub.off_requests = [(e, d - start, s, w) for e, d, s, w in inst.off_requests if d >= today]
    sub.cover = {(d - start, s): v for (d, s), v in inst.cover.items() if d >= start}
    for (d, s), n in (extra_cover or {}).items():
        if d >= today:
            req, wu, wo = sub.cover.get((d - start, s), (0, 100, 1))
            sub.cover[(d - start, s)] = (max(0, req + n), wu, wo)
    fixed = np.zeros((inst.E, sub.horizon), dtype=bool)
    fixed[:, :today - start] = True
    base = R[:, start:].copy()
    return sub, start, fixed, base


def reroster(inst: Instance, R, today: int, absences=(), extra_cover=None, seconds: float = 20.0,
             keep_weight: int = 10, seed: int = 0, progress=None) -> dict:
    """absences: (employee, first day, last day). extra_cover: {(day, shift): +n}. Returns the new full roster,
    the changed cells and the check of the repaired part."""
    sub, start, fixed, base = window(inst, R, today, absences, extra_cover)
    off = np.zeros_like(fixed)
    for e, ds in sub.days_off.items():
        for d in ds:
            if 0 <= d < sub.horizon and not fixed[e, d]:
                off[e, d] = True
    res = lns(sub, seconds=seconds, seed=seed, start=base, keep=base, keep_weight=keep_weight, fixed_mask=fixed,
              extra_off=off, progress=progress)
    new = R.copy()
    new[:, start:] = res.roster
    changed = np.zeros_like(R, dtype=bool)
    changed[:, today:] = new[:, today:] != R[:, today:]
    return {"roster": new, "changed": changed, "sub": sub, "start": start, "problems": check(sub, res.roster),
            "result": res, "moves": _moves(inst, R, new, changed)}


def _moves(inst, R, new, changed):
    out = []
    for e, d in zip(*np.nonzero(changed)):
        a, b = int(R[e, d]), int(new[e, d])
        out.append({"e": int(e), "day": int(d), "was": inst.shifts[a].id if a != OFF else "off",
                    "now": inst.shifts[b].id if b != OFF else "off"})
    return out

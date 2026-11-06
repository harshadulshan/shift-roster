"""Tables, the by-hand baseline, and the Excel roster."""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd

from .problem import OFF, Instance, check, penalty

COLORS = ["#3b82f6", "#f0883e", "#a371f7", "#2ea043", "#e3b341", "#39c5cf", "#db61a2", "#8b949e", "#f85149",
          "#79c0ff", "#ffa657", "#d2a8ff", "#7ee787", "#56d364", "#1f6feb", "#bc4c00", "#8957e5", "#1b7c83"]


def dates(inst: Instance):
    start = dt.date.fromisoformat(inst.start_date) if inst.start_date else None
    return [start + dt.timedelta(days=d) if start else None for d in range(inst.horizon)]


def day_label(inst: Instance, d: int, short: bool = True) -> str:
    ds = dates(inst)[d]
    wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][(inst.first_weekday + d) % 7]
    if ds is None:
        return f"{wd} {d + 1}"
    return f"{wd} {ds.day}" if short else f"{wd} {ds:%d %b}"


def tick(inst: Instance, d: int) -> str:
    """Compact axis label: weekday initial and day of month (or day number)."""
    ds = dates(inst)[d]
    wd = "MTWTFSS"[(inst.first_weekday + d) % 7]
    return f"{wd}{ds.day if ds else d + 1}"


def is_weekend(inst: Instance, d: int) -> bool:
    return (inst.first_weekday + d) % 7 >= 5


def greedy(inst: Instance, seed: int = 0) -> np.ndarray:
    """What a busy manager does by hand: day by day, fill each shift with whoever is free, may legally follow
    yesterday's shift, has not hit their run, hours or weekend limit, and has worked least so far. It does not look
    ahead, so minimum hours and minimum runs of working days and days off are often broken."""
    rng = np.random.default_rng(seed)
    E, H, S = inst.E, inst.horizon, inst.S
    R = np.full((E, H), OFF, dtype=int)
    mins = np.zeros(E)
    count = np.zeros((E, S), dtype=int)
    order = sorted(range(S), key=lambda s: -len([1 for t in inst.shifts if inst.shifts[s].id in t.cannot_follow]))
    wk_of = {}
    for k, (sat, sun) in enumerate(inst.weekends()):
        for x in (sat, sun):
            if x is not None:
                wk_of[x] = k
    worked_wk = [set() for _ in range(E)]
    for d in range(H):
        for s in order:
            req = inst.cover.get((d, s), (0, 0, 0))[0]
            cand = []
            for e, emp in enumerate(inst.staff):
                if R[e, d] != OFF or d in inst.days_off.get(e, ()):
                    continue
                if d > 0 and R[e, d - 1] != OFF and inst.shifts[s].id in inst.shifts[R[e, d - 1]].cannot_follow:
                    continue
                run = 0
                while d - run - 1 >= 0 and R[e, d - run - 1] != OFF:
                    run += 1
                if run >= emp.max_consec or count[e, s] >= emp.max_shifts.get(inst.shifts[s].id, 10 ** 6):
                    continue
                if mins[e] + inst.shifts[s].minutes > emp.max_minutes:
                    continue
                if d in wk_of and wk_of[d] not in worked_wk[e] and len(worked_wk[e]) >= emp.max_weekends:
                    continue
                cand.append((mins[e] / max(emp.max_minutes, 1), rng.random(), e))
            for _, _, e in sorted(cand)[:req]:
                R[e, d] = s
                if d in wk_of:
                    worked_wk[e].add(wk_of[d])
                mins[e] += inst.shifts[s].minutes
                count[e, s] += 1
    return R


def person_stats(inst: Instance, R) -> pd.DataFrame:
    rows = []
    wk = inst.weekends()
    for e, emp in enumerate(inst.staff):
        row = R[e]
        minutes = sum(inst.shifts[v].minutes for v in row if v != OFF)
        on = [w for (ee, d, s, w) in inst.on_requests if ee == e]
        on_ok = [w for (ee, d, s, w) in inst.on_requests if ee == e and row[d] == s]
        off_req = {(d) for (ee, d, s, w) in inst.off_requests if ee == e}
        off_bad = {d for (ee, d, s, w) in inst.off_requests if ee == e and row[d] == s}
        asked = len(on) + len(off_req)
        granted = len(on_ok) + len(off_req) - len(off_bad)
        r = {"name": emp.name or emp.id, "role": emp.role, "shifts": int(np.sum(row != OFF)),
             "hours": minutes / 60, "min hours": emp.min_minutes / 60, "max hours": emp.max_minutes / 60}
        for s, sh in enumerate(inst.shifts):
            r[sh.id] = int(np.sum(row == s))
        r["weekends"] = sum(1 for sat, sun in wk if any(d is not None and row[d] != OFF for d in (sat, sun)))
        r["max weekends"] = emp.max_weekends
        r["leave days"] = len(inst.days_off.get(e, ()))
        r["requests"] = asked
        r["granted"] = granted
        rows.append(r)
    return pd.DataFrame(rows)


def cover_table(inst: Instance, R) -> pd.DataFrame:
    rows = []
    for d in range(inst.horizon):
        for s, sh in enumerate(inst.shifts):
            req = inst.cover.get((d, s), (0, 0, 0))[0]
            n = int(np.sum(R[:, d] == s))
            rows.append({"day": d, "date": day_label(inst, d), "shift": sh.id, "needed": req, "rostered": n,
                         "short": max(0, req - n), "extra": max(0, n - req)})
    return pd.DataFrame(rows)


def requests_table(inst: Instance, R) -> pd.DataFrame:
    rows = []
    off = {}
    for e, d, s, w in inst.off_requests:
        off.setdefault((e, d), []).append((s, w))
    for (e, d), lst in off.items():
        shifts = {s for s, _ in lst}
        what = "day off" if len(shifts) == inst.S else "not " + "/".join(inst.shifts[s].id for s in sorted(shifts))
        rows.append({"name": inst.staff[e].name or inst.staff[e].id, "day": day_label(inst, d, False), "asked": what,
                     "priority": max(w for _, w in lst), "granted": R[e, d] not in shifts,
                     "rostered": inst.shifts[R[e, d]].id if R[e, d] != OFF else "off"})
    for e, d, s, w in inst.on_requests:
        rows.append({"name": inst.staff[e].name or inst.staff[e].id, "day": day_label(inst, d, False),
                     "asked": f"work {inst.shifts[s].id}", "priority": w, "granted": R[e, d] == s,
                     "rostered": inst.shifts[R[e, d]].id if R[e, d] != OFF else "off"})
    return pd.DataFrame(rows, columns=["name", "day", "asked", "priority", "granted", "rostered"])


def summary(inst: Instance, R) -> dict:
    pb = penalty(inst, R)
    ct = cover_table(inst, R)
    rq = requests_table(inst, R)
    return {"problems": check(inst, R), "penalty": pb["total"], "short": int(ct["short"].sum()),
            "extra": int(ct["extra"].sum()), "requests": len(rq), "granted": int(rq["granted"].sum()) if len(rq) else 0,
            "breakdown": pb}


def roster_table(inst: Instance, R) -> pd.DataFrame:
    cols = [day_label(inst, d) for d in range(inst.horizon)]
    data = [[(inst.shifts[v].id if v != OFF else ("L" if d in inst.days_off.get(e, ()) else "")) for d, v in enumerate(R[e])]
            for e in range(inst.E)]
    return pd.DataFrame(data, columns=cols, index=[s.name or s.id for s in inst.staff])


def roster_excel(inst: Instance, R, title: str = "Roster", changed=None) -> bytes:
    """The roster as people read it on the notice board: one row per person, one column per day, coloured shifts,
    weekends shaded, leave marked L, cover and shortfalls underneath, then the hours per person."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    ws = wb.active
    ws.title = "Roster"
    thin = Side(style="thin", color="D0D7DE")
    ws.cell(1, 1, title).font = Font(bold=True, size=13)
    ws.cell(2, 1, inst.description).font = Font(color="57606A")
    ws.cell(4, 1, "Name").font = Font(bold=True)
    ws.cell(4, 2, "Role").font = Font(bold=True)
    for d in range(inst.horizon):
        c = ws.cell(4, 3 + d, day_label(inst, d))
        c.font = Font(bold=True, size=9)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        if is_weekend(inst, d):
            c.fill = PatternFill("solid", fgColor="EEF2F7")
    fills = {s: PatternFill("solid", fgColor=COLORS[s % len(COLORS)].lstrip("#")) for s in range(inst.S)}
    for e in range(inst.E):
        r = 5 + e
        ws.cell(r, 1, inst.staff[e].name or inst.staff[e].id)
        ws.cell(r, 2, inst.staff[e].role)
        for d in range(inst.horizon):
            v = int(R[e, d])
            c = ws.cell(r, 3 + d)
            c.alignment = Alignment(horizontal="center")
            c.border = Border(top=thin, bottom=thin, left=thin, right=thin)
            if v != OFF:
                c.value = inst.shifts[v].id
                c.fill = fills[v]
                c.font = Font(bold=True, color="FFFFFF")
            elif d in inst.days_off.get(e, ()):
                c.value = "L"
                c.font = Font(color="8B949E")
            elif is_weekend(inst, d):
                c.fill = PatternFill("solid", fgColor="F6F8FA")
            if changed is not None and changed[e, d]:
                c.border = Border(top=Side(style="medium", color="CF222E"), bottom=Side(style="medium", color="CF222E"),
                                  left=Side(style="medium", color="CF222E"), right=Side(style="medium", color="CF222E"))
    r0 = 6 + inst.E
    ws.cell(r0, 1, "Short (needed - rostered)").font = Font(bold=True, color="CF222E")
    for s, sh in enumerate(inst.shifts):
        ws.cell(r0 + 1 + s, 1, f"{sh.id} {sh.name}".strip())
        for d in range(inst.horizon):
            req = inst.cover.get((d, s), (0, 0, 0))[0]
            n = int(np.sum(R[:, d] == s))
            c = ws.cell(r0 + 1 + s, 3 + d, f"{n}/{req}")
            c.alignment = Alignment(horizontal="center")
            c.font = Font(size=9, color="CF222E" if n < req else "57606A", bold=n < req)
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 22
    for d in range(inst.horizon):
        ws.column_dimensions[get_column_letter(3 + d)].width = 5.2
    ws.freeze_panes = "C5"
    leg = r0 + 2 + inst.S
    ws.cell(leg, 1, "Shifts").font = Font(bold=True)
    for s, sh in enumerate(inst.shifts):
        c = ws.cell(leg + 1 + s, 1, f"{sh.id}  {sh.name} {sh.start}".strip())
        ws.cell(leg + 1 + s, 2).fill = fills[s]
    ws2 = wb.create_sheet("Hours")
    st = person_stats(inst, R)
    ws2.append(list(st.columns))
    for row in st.itertuples(index=False):
        ws2.append(list(row))
    ws3 = wb.create_sheet("Requests")
    rq = requests_table(inst, R)
    ws3.append(list(rq.columns))
    for row in rq.itertuples(index=False):
        ws3.append(["yes" if v is True else "no" if v is False else v for v in row])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def person_csv(inst: Instance, R) -> bytes:
    rows = []
    ds = dates(inst)
    for e in range(inst.E):
        for d in range(inst.horizon):
            v = int(R[e, d])
            if v == OFF:
                continue
            sh = inst.shifts[v]
            rows.append({"name": inst.staff[e].name or inst.staff[e].id, "date": ds[d].isoformat() if ds[d] else d,
                         "shift": sh.id, "shift name": sh.name, "start": sh.start, "hours": sh.minutes / 60})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

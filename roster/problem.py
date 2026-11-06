"""
The rostering problem and the independent checker.

A roster gives every employee, for every day, one shift type or a day off. Hard rules (must hold): one shift a day,
shift types that cannot follow each other (e.g. no early shift after a night), at most so many shifts of each type,
minimum and maximum total hours, maximum and minimum runs of working days, minimum runs of days off, at most so many
weekends, and agreed days off. Soft goals (penalties): shift requests granted or refused, and the number of people
on each shift above or below what the ward needs.

Same rules and penalties as the Curtois & Qu (2014) shift scheduling benchmark (schedulingbenchmarks.org): runs at the
very start and end of the period are not checked against the minimum run lengths (nothing is known about the days
before and after), a weekend is worked if Saturday or Sunday is, and day 0 is a Monday.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

OFF = -1


@dataclass
class Shift:
    id: str
    minutes: int
    cannot_follow: set = field(default_factory=set)     # shift ids that may not be worked the day after this one
    start: str = ""                                      # display only, e.g. "07:00"
    name: str = ""


@dataclass
class Employee:
    id: str
    max_shifts: dict                                     # shift id -> max count
    max_minutes: int
    min_minutes: int
    max_consec: int                                      # most working days in a row
    min_consec: int                                      # fewest working days in a row (between days off)
    min_consec_off: int                                  # fewest days off in a row (between working days)
    max_weekends: int
    name: str = ""
    role: str = ""


@dataclass
class Instance:
    name: str
    horizon: int
    shifts: list
    staff: list
    days_off: dict = field(default_factory=dict)        # employee index -> set of days
    on_requests: list = field(default_factory=list)     # (e, d, s, weight): penalty if e does NOT work s on d
    off_requests: list = field(default_factory=list)    # (e, d, s, weight): penalty if e works s on d
    cover: dict = field(default_factory=dict)           # (d, s) -> (required, weight under, weight over)
    start_date: str = ""                                 # ISO date of day 0, display only
    first_weekday: int = 0                               # weekday of day 0 (0 = Monday)
    breaks: dict = field(default_factory=dict)          # employee -> days of absence (sick): minimum runs not
                                                         # checked next to them, as at the ends of the period
    description: str = ""

    @property
    def E(self):
        return len(self.staff)

    @property
    def S(self):
        return len(self.shifts)

    def shift_index(self, sid):
        return next(k for k, s in enumerate(self.shifts) if s.id == sid)

    def weekends(self):
        """(Saturday, Sunday) day pairs inside the period (None when that day falls outside it)."""
        out = []
        for d in range(-1, self.horizon):
            if (self.first_weekday + d) % 7 == 5:
                sat = d if d >= 0 else None
                sun = d + 1 if d + 1 < self.horizon else None
                out.append((sat, sun))
        return out


# ------------------------------------------------------------------------------------------------ reading
def read_nrp(text: str, name: str = "") -> Instance:
    """The schedulingbenchmarks.org text format (SECTION_HORIZON, SECTION_SHIFTS, SECTION_STAFF, ...)."""
    sections, cur = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("SECTION_"):
            cur = line
            sections[cur] = []
        elif cur:
            sections[cur].append(line)
    horizon = int(sections["SECTION_HORIZON"][0])
    shifts = []
    for line in sections["SECTION_SHIFTS"]:
        parts = line.split(",")
        follow = {x for x in (parts[2] if len(parts) > 2 else "").split("|") if x}
        shifts.append(Shift(parts[0], int(parts[1]), follow))
    sid = {s.id: k for k, s in enumerate(shifts)}
    staff = []
    for line in sections["SECTION_STAFF"]:
        p = line.split(",")
        mx = {s.id: 10 ** 6 for s in shifts}
        for item in p[1].split("|"):
            if "=" in item:
                k, v = item.split("=")
                mx[k] = int(v)
        staff.append(Employee(p[0], mx, int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6]), int(p[7])))
    eid = {e.id: k for k, e in enumerate(staff)}
    days_off = {k: set() for k in range(len(staff))}
    for line in sections.get("SECTION_DAYS_OFF", []):
        p = line.split(",")
        days_off[eid[p[0]]].update(int(x) for x in p[1:] if x != "")
    on = [(eid[p[0]], int(p[1]), sid[p[2]], int(p[3])) for p in (l.split(",") for l in sections.get("SECTION_SHIFT_ON_REQUESTS", []))]
    off = [(eid[p[0]], int(p[1]), sid[p[2]], int(p[3])) for p in (l.split(",") for l in sections.get("SECTION_SHIFT_OFF_REQUESTS", []))]
    cover = {}
    for line in sections["SECTION_COVER"]:
        p = line.split(",")
        cover[(int(p[0]), sid[p[1]])] = (int(p[2]), int(p[3]), int(p[4]))
    return Instance(name, horizon, shifts, staff, days_off, on, off, cover)


# ------------------------------------------------------------------------------------------------ checking
def runs(row, working: bool):
    """(start, length) of every maximal run of working days (or of days off) in one employee's row."""
    out, start = [], None
    for d, v in enumerate(list(row) + [None]):
        is_w = v is not None and v != OFF
        hit = v is not None and is_w == working
        if hit and start is None:
            start = d
        elif not hit and start is not None:
            out.append((start, d - start))
            start = None
    return out


def check(inst: Instance, roster) -> list:
    """Every hard rule the roster breaks, in words. Empty = valid. Written separately from the solver."""
    R = np.asarray(roster)
    if R.shape != (inst.E, inst.horizon):
        return [f"roster has shape {R.shape}, expected {(inst.E, inst.horizon)}"]
    problems = []
    for e in range(inst.E):
        problems += check_employee(inst, e, R[e])
    return problems


def check_employee(inst: Instance, e: int, row) -> list:
    """The hard rules one employee's row breaks."""
    H, problems = inst.horizon, []
    row = np.asarray(row)
    emp = inst.staff[e]
    if True:
        nm = emp.name or emp.id
        bad = [d for d in range(H) if not (row[d] == OFF or 0 <= row[d] < inst.S)]
        if bad:
            return [f"{nm}: unknown shift on day {bad[0]}"]
        for d in sorted(inst.days_off.get(e, ())):
            if row[d] != OFF:
                problems.append(f"{nm}: works on agreed day off {d}")
        for d in range(H - 1):
            if row[d] != OFF and row[d + 1] != OFF and inst.shifts[row[d + 1]].id in inst.shifts[row[d]].cannot_follow:
                problems.append(f"{nm}: {inst.shifts[row[d + 1]].id} on day {d + 1} after {inst.shifts[row[d]].id} on day {d}")
        for s, sh in enumerate(inst.shifts):
            n = int(np.sum(row == s))
            if n > emp.max_shifts.get(sh.id, 10 ** 6):
                problems.append(f"{nm}: {n} {sh.id} shifts, at most {emp.max_shifts[sh.id]}")
        minutes = sum(inst.shifts[v].minutes for v in row if v != OFF)
        if minutes > emp.max_minutes:
            problems.append(f"{nm}: {minutes} minutes, at most {emp.max_minutes}")
        if minutes < emp.min_minutes:
            problems.append(f"{nm}: {minutes} minutes, at least {emp.min_minutes}")
        br = inst.breaks.get(e, set())
        for start, length in runs(row, True):
            if length > emp.max_consec:
                problems.append(f"{nm}: {length} working days in a row from day {start}, at most {emp.max_consec}")
            if length < emp.min_consec and start > 0 and start + length < H:
                if not ({start - 1, start + length} & br):
                    problems.append(f"{nm}: only {length} working days in a row from day {start}, at least {emp.min_consec}")
        for start, length in runs(row, False):
            if length < emp.min_consec_off and start > 0 and start + length < H:
                if not (set(range(start, start + length)) & br):
                    problems.append(f"{nm}: only {length} days off in a row from day {start}, at least {emp.min_consec_off}")
        wk = sum(1 for sat, sun in inst.weekends() if any(d is not None and row[d] != OFF for d in (sat, sun)))
        if wk > emp.max_weekends:
            problems.append(f"{nm}: works {wk} weekends, at most {emp.max_weekends}")
    return problems


def penalty(inst: Instance, roster) -> dict:
    """Soft penalties: requests to work a shift not granted, requests not to work a shift not granted, cover."""
    R = np.asarray(roster)
    on = sum(w for e, d, s, w in inst.on_requests if R[e, d] != s)
    off = sum(w for e, d, s, w in inst.off_requests if R[e, d] == s)
    under = over = 0
    short = []
    for (d, s), (req, wu, wo) in inst.cover.items():
        n = int(np.sum(R[:, d] == s))
        if n < req:
            under += wu * (req - n)
            short.append((d, s, req - n))
        elif n > req:
            over += wo * (n - req)
    return {"total": on + off + under + over, "shift on requests": on, "shift off requests": off,
            "cover under": under, "cover over": over, "short": short}

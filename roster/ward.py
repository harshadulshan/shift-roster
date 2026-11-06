"""
A ward (or any team) as tables a manager can fill in: shifts, staff and their contracts, cover needed, leave and
requests. Read from and written to one Excel workbook, and turned into the rostering problem.

Sheets:  Settings (name, start date, weeks, weights) · Shifts (code, name, start, end, cannot_follow)
         Staff (name, role, min_hours, max_hours, max_<shift> ..., max_in_a_row, min_in_a_row, min_off_in_a_row,
         max_weekends) · Cover (day = Mon..Sun or a date, one column per shift) · Leave (name, from, to)
         Requests (name, date, shift or "any", want = work / off, priority 1-3)
"""

from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .problem import Employee, Instance, Shift

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass
class Ward:
    name: str
    start: dt.date
    weeks: int
    shifts: pd.DataFrame
    staff: pd.DataFrame
    cover: pd.DataFrame
    leave: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["name", "from", "to", "reason"]))
    requests: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["name", "date", "shift", "want", "priority"]))
    under_weight: int = 100
    over_weight: int = 1
    request_weight: int = 1          # penalty per priority point of a request not granted
    leave_credit: bool = True        # leave days count towards the contract hours (8 h each)

    @property
    def days(self):
        return [self.start + dt.timedelta(days=k) for k in range(7 * self.weeks)]


def minutes_between(a: str, b: str) -> int:
    h1, m1 = map(int, str(a)[:5].split(":"))
    h2, m2 = map(int, str(b)[:5].split(":"))
    m = (h2 * 60 + m2) - (h1 * 60 + m1)
    return m if m > 0 else m + 24 * 60


def _date(v) -> dt.date:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    return pd.to_datetime(str(v)).date()


def to_instance(w: Ward) -> Instance:
    days = w.days
    H = len(days)
    index = {d: k for k, d in enumerate(days)}
    shifts = []
    for _, r in w.shifts.iterrows():
        follow = {x.strip() for x in str(r.get("cannot_follow", "") or "").replace(",", "|").split("|") if x.strip()}
        shifts.append(Shift(str(r["code"]), minutes_between(r["start"], r["end"]), follow, str(r["start"])[:5],
                            str(r.get("name", r["code"]))))
    codes = [s.id for s in shifts]
    names = list(w.staff["name"].astype(str))
    leave_days = {k: set() for k in range(len(names))}
    for _, r in w.leave.iterrows():
        if str(r["name"]) not in names:
            raise ValueError(f"Leave: '{r['name']}' is not in the staff list")
        e = names.index(str(r["name"]))
        a, b = _date(r["from"]), _date(r["to"])
        for d in days:
            if a <= d <= b:
                leave_days[e].add(index[d])
    staff = []
    for k, (_, r) in enumerate(w.staff.iterrows()):
        mx = {}
        for c in codes:
            v = r.get(f"max_{c}")
            mx[c] = 10 ** 6 if v is None or pd.isna(v) or str(v).strip() == "" else int(float(v))
        credit = 8 * 60 * len(leave_days[k]) if w.leave_credit else 0
        staff.append(Employee(f"E{k + 1}", mx, max(0, int(round(float(r["max_hours"]) * 60)) - credit),
                              max(0, int(round(float(r["min_hours"]) * 60)) - credit),
                              int(r["max_in_a_row"]), int(r["min_in_a_row"]), int(r["min_off_in_a_row"]),
                              int(r["max_weekends"]), str(r["name"]), str(r.get("role", ""))))
    on, off = [], []
    for _, r in w.requests.iterrows():
        if str(r["name"]) not in names:
            raise ValueError(f"Requests: '{r['name']}' is not in the staff list")
        e = names.index(str(r["name"]))
        d = index.get(_date(r["date"]))
        if d is None:
            continue
        wt = int(w.request_weight * int(r.get("priority", 1) or 1))
        sh = str(r.get("shift", "any") or "any").strip()
        want = str(r.get("want", "off")).strip().lower()
        targets = codes if sh.lower() in ("any", "", "nan") else [sh]
        for c in targets:
            if c not in codes:
                raise ValueError(f"Requests: unknown shift '{c}'")
            s = codes.index(c)
            if want == "work":
                on.append((e, d, s, wt))
            else:
                off.append((e, d, s, wt))
    cover = {}
    pattern, dated = {}, {}
    for _, r in w.cover.iterrows():
        key = str(r["day"]).strip()
        vals = {c: int(r[c]) for c in codes if c in r and not pd.isna(r[c])}
        if key[:3].title() in WEEKDAYS:
            pattern[WEEKDAYS.index(key[:3].title())] = vals
        else:
            dated[_date(key)] = vals
    for d, day in enumerate(days):
        vals = dated.get(day, pattern.get(day.weekday(), {}))
        for c, req in vals.items():
            cover[(d, codes.index(c))] = (int(req), w.under_weight, w.over_weight)
    inst = Instance(w.name, H, shifts, staff, leave_days, on, off, cover, w.start.isoformat(), w.start.weekday())
    inst.description = (f"{len(staff)} staff, {len(shifts)} shift types, {w.weeks} weeks from "
                        f"{w.start:%a %d %b %Y}.")
    return inst


# ------------------------------------------------------------------------------------------------ Excel
def to_excel(w: Ward) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as x:
        pd.DataFrame({"setting": ["name", "start (a date)", "weeks", "weight: one person short", "weight: one person extra",
                                  "weight: per request priority point", "leave counts towards hours (yes/no)"],
                      "value": [w.name, w.start.isoformat(), w.weeks, w.under_weight, w.over_weight, w.request_weight,
                                "yes" if w.leave_credit else "no"]}).to_excel(x, sheet_name="Settings", index=False)
        w.shifts.to_excel(x, sheet_name="Shifts", index=False)
        w.staff.to_excel(x, sheet_name="Staff", index=False)
        w.cover.to_excel(x, sheet_name="Cover", index=False)
        w.leave.to_excel(x, sheet_name="Leave", index=False)
        w.requests.to_excel(x, sheet_name="Requests", index=False)
    return buf.getvalue()


def from_excel(data: bytes) -> Ward:
    xl = pd.ExcelFile(io.BytesIO(data))
    need = ["Settings", "Shifts", "Staff", "Cover"]
    missing = [s for s in need if s not in xl.sheet_names]
    if missing:
        raise ValueError(f"The workbook needs the sheets {', '.join(need)} (missing: {', '.join(missing)}). "
                         "Download the template to see the layout.")
    st = pd.read_excel(xl, "Settings")
    sv = dict(zip(st.iloc[:, 0].astype(str).str.strip(), st.iloc[:, 1]))

    def get(prefix, default):
        for k, v in sv.items():
            if k.lower().startswith(prefix):
                return v
        return default

    def table(name, cols):
        if name not in xl.sheet_names:
            return pd.DataFrame(columns=cols)
        df = pd.read_excel(xl, name)
        df.columns = [str(c).strip() for c in df.columns]
        return df.dropna(how="all")

    shifts = table("Shifts", [])
    for c in ("start", "end"):
        shifts[c] = shifts[c].map(lambda v: v.strftime("%H:%M") if hasattr(v, "strftime") else str(v)[:5])
    shifts["cannot_follow"] = shifts.get("cannot_follow", "").fillna("").astype(str)
    staff = table("Staff", [])
    staff["name"] = staff["name"].astype(str).str.strip()
    if staff["name"].duplicated().any():
        raise ValueError(f"Staff: the name '{staff['name'][staff['name'].duplicated()].iloc[0]}' appears twice")
    w = Ward(str(get("name", "My ward")), _date(get("start", dt.date.today())), int(get("weeks", 4)), shifts, staff,
             table("Cover", []), table("Leave", ["name", "from", "to", "reason"]),
             table("Requests", ["name", "date", "shift", "want", "priority"]),
             int(get("weight: one person short", 100)), int(get("weight: one person extra", 1)),
             int(get("weight: per request", 1)), str(get("leave counts", "yes")).strip().lower().startswith("y"))
    to_instance(w)                    # raises a readable error if something does not fit
    return w


# ------------------------------------------------------------------------------------------------ demo
FIRST = ["Nadeesha", "Chamari", "Dilani", "Sanduni", "Ishara", "Hiruni", "Kavindi", "Tharushi", "Nimesha", "Piumi",
         "Sachini", "Madushani", "Ruwini", "Gayani", "Anjali", "Kasun", "Dinesh", "Lahiru", "Chathura", "Sampath",
         "Tharindu", "Malki", "Shehani", "Nilmini"]
LAST = ["Perera", "Fernando", "Silva", "Jayasinghe", "Bandara", "Herath", "Wickramasinghe", "Rathnayake", "Gunasekara",
        "Dissanayake", "Senanayake", "Kumari", "Wijesuriya", "Abeykoon", "Ekanayake", "Liyanage", "Samarasinghe",
        "Karunaratne", "Weerasinghe", "Pathirana"]


def demo(start: dt.date | None = None, weeks: int = 4, seed: int = 7) -> Ward:
    """A fictional general medical ward: 18 nursing staff, three 8-hour shifts, four weeks. Generated names."""
    rng = np.random.default_rng(seed)
    if start is None:
        start = dt.date(2026, 11, 2)
    shifts = pd.DataFrame({"code": ["D", "E", "N"], "name": ["Day", "Evening", "Night"],
                           "start": ["07:00", "15:00", "23:00"], "end": ["15:00", "23:00", "07:00"],
                           "cannot_follow": ["", "D", "D|E"]})
    first = list(rng.permutation(FIRST))
    last = list(rng.permutation(LAST))
    rows = []
    for k in range(18):
        nm = f"{first[k]} {last[k % len(LAST)]}"
        if k < 2:
            role, mn, mx, n = "Nursing Sister", 144 * weeks / 4, 160 * weeks / 4, 0
        elif k < 16:
            role, mn, mx, n = "Nursing Officer", 144 * weeks / 4, 160 * weeks / 4, 7 * weeks / 4
        else:
            role, mn, mx, n = "Nursing Officer (part-time)", 80 * weeks / 4, 96 * weeks / 4, 4 * weeks / 4
        rows.append({"name": nm, "role": role, "min_hours": mn, "max_hours": mx, "max_D": None, "max_E": None,
                     "max_N": int(n), "max_in_a_row": 5, "min_in_a_row": 2, "min_off_in_a_row": 2,
                     "max_weekends": 3 if k in (4, 9, 13) else 2})
    staff = pd.DataFrame(rows)
    cover = pd.DataFrame({"day": WEEKDAYS, "D": [5, 5, 5, 5, 5, 4, 4], "E": [4, 4, 4, 4, 4, 3, 3], "N": [3] * 7})
    days = [start + dt.timedelta(days=k) for k in range(7 * weeks)]
    leave = pd.DataFrame([
        {"name": staff.name[5], "from": days[8], "to": days[12], "reason": "annual leave"},
        {"name": staff.name[11], "from": days[15], "to": days[21], "reason": "annual leave"},
        {"name": staff.name[16], "from": days[2], "to": days[3], "reason": "exam"},
    ])
    req = []
    for k in range(26):
        e = int(rng.integers(18))
        d = days[int(rng.integers(len(days)))]
        if rng.random() < 0.6:
            req.append({"name": staff.name[e], "date": d, "shift": "any", "want": "off", "priority": int(rng.integers(1, 4))})
        else:
            sh = str(rng.choice(["D", "E"] if e < 2 else ["D", "E", "N"]))
            req.append({"name": staff.name[e], "date": d, "shift": sh, "want": "work", "priority": int(rng.integers(1, 3))})
    requests = pd.DataFrame(req).drop_duplicates(["name", "date"]).reset_index(drop=True)
    return Ward("Ward 12 · General Medicine (fictional)", start, weeks, shifts, staff, cover, leave, requests)

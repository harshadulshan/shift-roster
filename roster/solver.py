"""
Building a roster with CP-SAT (Google OR-Tools) and large neighbourhood search.

Every hard rule belongs to one employee; employees are linked only through the soft cover targets. So a roster can be
improved one small group at a time: free a few employees (or a few employees over a window of days), keep everyone
else as they are, and let CP-SAT find the best way to re-roster the group exactly. Repeating this with different
groups (large neighbourhood search) handles rosters with 150 staff over a whole year, where one model of everything
would not even fit in memory. Small problems are first solved as a whole, so CP-SAT can prove optimality.

The model: one yes/no variable per free (employee, day, shift); every hard rule as a constraint; the soft penalties
as the objective. Cells that are not free are constants taken from the current roster.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model

from .problem import OFF, Instance, check, check_employee, penalty

VMIN = 40.0                      # smallest neighbourhood, in free (employee, day) cells


@dataclass
class Result:
    roster: np.ndarray                 # employees x days, shift index or OFF (-1)
    penalty: float
    bound: float                       # lower bound on the best possible penalty (0 if unknown)
    status: str                        # "optimal", "feasible", "infeasible"
    seconds: float
    history: list = field(default_factory=list)      # (seconds, penalty)
    iterations: int = 0
    first_penalty: float = float("nan")

    @property
    def ok(self):
        return self.status in ("optimal", "feasible")


# ------------------------------------------------------------------------------------------------ model
class Sub:
    """CP-SAT model over the free cells (mask) with every other cell fixed to the base roster."""

    def __init__(self, inst: Instance, base: np.ndarray, mask: np.ndarray, keep: np.ndarray | None = None,
                 keep_weight: int = 0, extra_off: np.ndarray | None = None, fair: dict | None = None,
                 objective: bool = True):
        self.inst, self.base, self.mask = inst, base, mask
        E, H, S = inst.E, inst.horizon, inst.S
        m = self.m = cp_model.CpModel()
        self.x, self.w = {}, {}
        emps = [e for e in range(E) if mask[e].any()]
        self.emps = emps
        days_off = {e: set(inst.days_off.get(e, ())) for e in range(E)}
        if extra_off is not None:
            for e, d in zip(*np.nonzero(extra_off)):
                days_off[int(e)].add(int(d))

        def X(e, d, s):
            if mask[e, d]:
                return self.x[e, d, s]
            return 1 if base[e, d] == s else 0

        def W(e, d):
            if mask[e, d]:
                return self.w[e, d]
            return 1 if base[e, d] != OFF else 0

        self.X, self.W = X, W
        for e in emps:
            for d in np.nonzero(mask[e])[0]:
                d = int(d)
                if d in days_off[e]:
                    self.w[e, d] = 0
                    for s in range(S):
                        self.x[e, d, s] = 0
                    continue
                xs = []
                for s in range(S):
                    v = m.NewBoolVar("")
                    self.x[e, d, s] = v
                    xs.append(v)
                wv = m.NewBoolVar("")
                m.Add(sum(xs) == wv)
                self.w[e, d] = wv
        forb = {s: [t for t, nx in enumerate(inst.shifts) if nx.id in sh.cannot_follow] for s, sh in enumerate(inst.shifts)}

        def clause(lits):
            """Add OR(lits): literals are (var or 0/1 constant, positive?)."""
            out = []
            for v, pos in lits:
                if isinstance(v, int):
                    if (v == 1) == pos:
                        return            # already true
                    continue              # constant false literal
                out.append(v if pos else v.Not())
            if not out:
                m.AddBoolOr([])           # infeasible
            else:
                m.AddBoolOr(out)

        for e in emps:
            emp = inst.staff[e]
            fd = mask[e]
            # shift rotation: x[d, s] + sum(x[d+1, t] for t in forb[s]) <= 1
            for d in range(H - 1):
                if not (fd[d] or fd[d + 1]):
                    continue
                for s in range(S):
                    if not forb[s]:
                        continue
                    a = X(e, d, s)
                    if isinstance(a, int) and a == 0:
                        continue
                    terms = [X(e, d + 1, t) for t in forb[s]]
                    if all(isinstance(t, int) for t in terms) and isinstance(a, int):
                        continue
                    m.Add(a + sum(terms) <= 1)
            # max shifts per type, total minutes, weekends: counted over the whole period
            for s, sh in enumerate(inst.shifts):
                cap = emp.max_shifts.get(sh.id, 10 ** 6)
                if cap < H:
                    m.Add(sum(X(e, d, s) for d in range(H)) <= cap)
            minutes = sum(inst.shifts[s].minutes * X(e, d, s) for d in range(H) for s in range(S))
            m.Add(minutes <= emp.max_minutes)
            m.Add(minutes >= emp.min_minutes)
            c = emp.max_consec
            if c < H:
                for d in range(H - c):
                    if fd[d:d + c + 1].any():
                        m.Add(sum(W(e, j) for j in range(d, d + c + 1)) <= c)
            # a run of k < min working days between two days off is forbidden (period ends are free)
            br = inst.breaks.get(e, set())
            for k in range(1, emp.min_consec):
                for d in range(1, H - k):
                    if fd[d - 1:d + k + 1].any() and d - 1 not in br and d + k not in br:
                        clause([(W(e, d - 1), True)] + [(W(e, j), False) for j in range(d, d + k)] + [(W(e, d + k), True)])
            for k in range(1, emp.min_consec_off):
                for d in range(1, H - k):
                    if fd[d - 1:d + k + 1].any() and not any(j in br for j in range(d, d + k)):
                        clause([(W(e, d - 1), False)] + [(W(e, j), True) for j in range(d, d + k)] + [(W(e, d + k), False)])
            wks = inst.weekends()
            if emp.max_weekends < len(wks):
                ks = []
                for sat, sun in wks:
                    a, b = (W(e, sat) if sat is not None else 0), (W(e, sun) if sun is not None else 0)
                    if isinstance(a, int) and isinstance(b, int):
                        ks.append(1 if (a or b) else 0)
                        continue
                    kv = m.NewBoolVar("")
                    for v in (a, b):
                        if isinstance(v, int):
                            if v:
                                m.Add(kv == 1)
                        else:
                            m.AddImplication(v, kv)
                    ks.append(kv)
                m.Add(sum(ks) <= emp.max_weekends)
        if not objective:
            return
        # objective (constant parts are left out: they do not change within this neighbourhood)
        terms = []
        free = set(emps)
        for e, d, s, wt in inst.on_requests:
            if e in free and mask[e, d]:
                terms.append(wt * (1 - self.x[e, d, s]) if not isinstance(self.x[e, d, s], int) else wt * (1 - self.x[e, d, s]))
        for e, d, s, wt in inst.off_requests:
            if e in free and mask[e, d]:
                terms.append(wt * self.x[e, d, s])
        free_days = np.nonzero(mask.any(axis=0))[0]
        fd_set = set(int(d) for d in free_days)
        for (d, s), (req, wu, wo) in inst.cover.items():
            if d not in fd_set:
                continue
            n_fixed = int(sum(1 for e in range(E) if not mask[e, d] and base[e, d] == s))
            var = [self.x[e, d, s] for e in emps if mask[e, d] and not isinstance(self.x[e, d, s], int)]
            if not var:
                continue
            u = m.NewIntVar(0, max(req, 0), "")
            o = m.NewIntVar(0, E, "")
            m.Add(n_fixed + sum(var) - o + u == req)
            terms += [wu * u, wo * o]
        if keep is not None and keep_weight:
            for e in emps:
                for d in np.nonzero(mask[e])[0]:
                    d = int(d)
                    v = int(keep[e, d])
                    if v == OFF:
                        terms.append(keep_weight * self.w[e, d])
                    else:
                        terms.append(keep_weight * (1 - self.x[e, d, v]))
        if fair:
            terms += self._fairness(fair)
        m.Minimize(sum(terms) if terms else 0)

    def _fairness(self, fair):
        """Optional (app): spread chosen shift types and weekends evenly. fair = {"shifts": [s...], "weight": w,
        "targets": {e: target}} -> penalty weight per shift above the employee's fair share."""
        inst, m, out = self.inst, self.m, []
        for e in self.emps:
            tgt = fair["targets"].get(e)
            if tgt is None:
                continue
            n = sum(self.X(e, d, s) for d in range(inst.horizon) for s in fair["shifts"])
            ex = m.NewIntVar(0, inst.horizon, "")
            m.Add(ex >= n - int(np.ceil(tgt)))
            out.append(fair["weight"] * ex)
        return out

    def hint(self, roster):
        for (e, d, s), v in self.x.items():
            if not isinstance(v, int):
                self.m.AddHint(v, int(roster[e, d] == s))

    def solve(self, seconds, workers=1, seed=0, callback=None, first=False):
        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = max(0.05, float(seconds))
        sv.parameters.num_search_workers = int(workers)
        sv.parameters.random_seed = int(seed)
        sv.parameters.linearization_level = 2          # LP relaxation with cuts: much stronger bounds here
        if first:
            sv.parameters.stop_after_first_solution = True
        st = sv.Solve(self.m, callback) if callback else sv.Solve(self.m)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, st, sv
        R = self.base.copy()
        for e in self.emps:
            for d in np.nonzero(self.mask[e])[0]:
                d = int(d)
                R[e, d] = OFF
                for s in range(self.inst.S):
                    v = self.x[e, d, s]
                    if (v if isinstance(v, int) else sv.BooleanValue(v)):
                        R[e, d] = s
        return R, st, sv


# ------------------------------------------------------------------------------------------------ search
def cells(inst):
    return inst.E * inst.horizon * inst.S


def single_type_row(inst: Instance, e: int, t: int, base_row, free, off_days, seconds: float = 5.0):
    """A valid row for one employee using only shift type t on the free days (other days kept): a quick first
    roster that only has to choose working days. None if that does not work."""
    H, emp, sh = inst.horizon, inst.staff[e], inst.shifts[t]
    m = cp_model.CpModel()
    w = {}
    for d in range(H):
        if free[d]:
            w[d] = 0 if d in off_days else m.NewBoolVar("")
        else:
            w[d] = 1 if base_row[d] != OFF else 0

    def blocked(u, v):           # shift type u on a day, v the next day
        return inst.shifts[v].id in inst.shifts[u].cannot_follow

    typ = [t if free[d] else base_row[d] for d in range(H)]
    for d in range(H - 1):
        a, b = w[d], w[d + 1]
        if isinstance(a, int) and isinstance(b, int):
            continue
        if typ[d] != OFF and typ[d + 1] != OFF and blocked(typ[d], typ[d + 1]):
            m.Add(a + b <= 1)
    fixed_min = sum(inst.shifts[base_row[d]].minutes for d in range(H) if not free[d] and base_row[d] != OFF)
    n_t_fixed = sum(1 for d in range(H) if not free[d] and base_row[d] == t)
    n = sum(w[d] for d in range(H) if free[d])
    m.Add(fixed_min + sh.minutes * n <= emp.max_minutes)
    # the minimum is reached by maximising the working days and stopping once it is met: much faster than asking
    # for it directly when the contract is tight (e.g. 232 of at most 236 possible working days)
    need = max(0, -(-(emp.min_minutes - fixed_min) // sh.minutes))
    m.Add(n_t_fixed + n <= emp.max_shifts.get(sh.id, 10 ** 6))
    W = [w[d] for d in range(H)]
    c = emp.max_consec
    for d in range(H - c):
        m.Add(sum(W[d:d + c + 1]) <= c)

    def clause(lits):
        out = []
        for v, pos in lits:
            if isinstance(v, int):
                if (v == 1) == pos:
                    return
                continue
            out.append(v if pos else v.Not())
        m.AddBoolOr(out)
    br = inst.breaks.get(e, set())
    for k in range(1, emp.min_consec):
        for d in range(1, H - k):
            if d - 1 not in br and d + k not in br:
                clause([(W[d - 1], True)] + [(W[j], False) for j in range(d, d + k)] + [(W[d + k], True)])
    for k in range(1, emp.min_consec_off):
        for d in range(1, H - k):
            if not any(j in br for j in range(d, d + k)):
                clause([(W[d - 1], False)] + [(W[j], True) for j in range(d, d + k)] + [(W[d + k], False)])
    ks = []
    for sat, sun in inst.weekends():
        kv = m.NewBoolVar("")
        for v in (W[sat] if sat is not None else 0, W[sun] if sun is not None else 0):
            if isinstance(v, int):
                if v:
                    m.Add(kv == 1)
            else:
                m.AddImplication(v, kv)
        ks.append(kv)
    if ks:
        m.Add(sum(ks) <= emp.max_weekends)
    m.Maximize(n)

    class Enough(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()

        def on_solution_callback(self):
            if self.ObjectiveValue() >= need:
                self.StopSearch()

    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = seconds
    sv.parameters.num_search_workers = 1
    st = sv.Solve(m, Enough())
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE) or sv.ObjectiveValue() < need - 1e-9:
        return None
    row = np.array(base_row).copy()
    for d in range(H):
        if free[d]:
            v = w[d]
            row[d] = t if (v if isinstance(v, int) else sv.BooleanValue(v)) else OFF
    return row


def fill_row(inst: Instance, e: int, R, mask, extra_off, seconds: float):
    """A valid row for one employee with any shift types: maximise the minutes worked and stop as soon as the
    contract's minimum is met (the hard part of tight contracts)."""
    sub = Sub(inst, R, mask, extra_off=extra_off, objective=False)
    emp = inst.staff[e]
    minutes = sum(inst.shifts[s].minutes * sub.X(e, d, s) for d in range(inst.horizon) for s in range(inst.S))
    sub.m.Maximize(minutes)

    class Enough(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()

        def on_solution_callback(self):
            if self.ObjectiveValue() >= emp.min_minutes:
                self.StopSearch()

    R2, st, sv = sub.solve(seconds, workers=1, callback=Enough())
    return R2


def construct(inst: Instance, deadline: float, rng, extra_off=None, fixed_mask=None, base=None, only=None) -> np.ndarray:
    """A first valid roster. Every hard rule belongs to one employee, so employees can be rostered one group at a
    time; for long periods each employee first gets a simple pattern of one shift type (the search improves it)."""
    E, H, S = inst.E, inst.horizon, inst.S
    R = np.full((E, H), OFF, dtype=int) if base is None else base.copy()
    emps = list(range(E)) if only is None else list(only)
    fm = np.zeros((E, H), dtype=bool) if fixed_mask is None else fixed_mask
    if extra_off is not None:
        R[extra_off & ~fm] = OFF
    demand = np.zeros(S)
    for (d, s), (req, wu, wo) in inst.cover.items():
        demand[s] += req
    order_t = list(np.argsort(-demand))

    def one(e):
        free = ~fm[e]
        offs = set(inst.days_off.get(e, ()))
        if extra_off is not None:
            offs |= set(int(d) for d in np.nonzero(extra_off[e])[0])
        for t in order_t:
            if inst.staff[e].max_shifts.get(inst.shifts[t].id, 1) <= 0:
                continue
            row = single_type_row(inst, e, int(t), R[e], free, offs, seconds=10.0)
            if row is not None:
                return row
        mask = np.zeros((E, H), dtype=bool)
        mask[e] = free
        R2 = fill_row(inst, e, R, mask, extra_off, 120.0)
        if R2 is None:
            raise ValueError(f"no valid roster for {inst.staff[e].name or inst.staff[e].id}: "
                             "their rules contradict each other")
        return R2[e]

    if H * S <= 1500:
        per = max(1, min(len(emps), 5))
        for k in range(0, len(emps), per):
            grp = emps[k:k + per]
            mask = np.zeros((E, H), dtype=bool)
            mask[grp] = True
            mask &= ~fm
            if not mask.any():
                continue
            left = max(0.3, (deadline - time.time()) / max(1, (len(emps) - k) / per))
            R2, st, _ = Sub(inst, R, mask, extra_off=extra_off).solve(min(left, 5.0))
            if R2 is not None:
                R = R2
            else:
                for e in grp:
                    R[e] = one(e)
    else:
        for e in emps:
            R[e] = one(e)
    return R


def bad_employees(inst, R, extra_off=None):
    out = [e for e in range(inst.E) if check_employee(inst, e, R[e])]
    if extra_off is not None:
        out += [e for e in range(inst.E) if np.any((R[e] != OFF) & extra_off[e]) and e not in out]
    return out


def lns(inst: Instance, seconds: float = 60.0, seed: int = 0, start: np.ndarray | None = None, progress=None,
        keep: np.ndarray | None = None, keep_weight: int = 0, fixed_mask: np.ndarray | None = None,
        extra_off: np.ndarray | None = None, fair: dict | None = None, whole_limit: int = 600) -> Result:
    """Best roster within the time budget. fixed_mask: cells that may not change (days already worked).
    extra_off: cells forced off (sickness, new leave). keep/keep_weight: prefer few changes to an existing roster."""
    t0 = time.time()
    deadline = t0 + seconds
    rng = np.random.default_rng(seed)
    E, H, S = inst.E, inst.horizon, inst.S
    fm = np.zeros((E, H), dtype=bool) if fixed_mask is None else fixed_mask
    history, bound, status = [], 0.0, "feasible"

    def score(R):
        p = penalty(inst, R)["total"]
        if keep is not None and keep_weight:
            p += keep_weight * int(np.sum((R != keep) & ~fm))
        if fair:
            for e, tgt in fair["targets"].items():
                n = int(np.isin(R[e], fair["shifts"]).sum())
                p += fair["weight"] * max(0, n - int(np.ceil(tgt)))
        return p

    R = None
    if start is not None:
        R = start.copy()
        if extra_off is not None:          # cells forced off must be re-planned: start from them off
            R[extra_off & ~fm] = OFF
    # 1. a valid first roster, a few employees at a time (only the employees whose rows break a rule)
    if R is None:
        R = construct(inst, t0 + seconds * 0.2, rng, extra_off, fm if fixed_mask is not None else None)
    else:
        bad = bad_employees(inst, R, extra_off)
        if bad:
            R = construct(inst, t0 + seconds * 0.2, rng, extra_off, fm if fixed_mask is not None else None, R, bad)
    cur = score(R)
    first = cur
    history.append((time.time() - t0, cur))
    if progress:
        progress(history[-1])
    # 2. the whole problem at once when it is small: CP-SAT may prove optimality
    if inst.E * inst.horizon <= whole_limit and time.time() < deadline - 0.1:
        sub = Sub(inst, R, ~fm, keep, keep_weight, extra_off, fair)
        sub.hint(R)
        share = 0.5
        R2, st, sv = sub.solve(min(deadline - time.time(), seconds * share), seed=seed)
        if R2 is not None:
            new = score(R2)
            if new <= cur:
                R, cur = R2, new
                history.append((time.time() - t0, cur))
                if progress:
                    progress(history[-1])
            if fixed_mask is None and keep is None and not fair:
                bound = max(bound, float(sv.BestObjectiveBound()))
            if st == cp_model.OPTIMAL:
                return Result(R, cur, cur, "optimal", time.time() - t0, history, 0, first)
    it = 0
    V = float(min(E * H, 160))                  # free (employee, day) cells per neighbourhood, adapted below
    vmax = float(min(E * H, max(400, 60000 // max(S, 1))))
    sub_time = 2.0
    n_opt = 0
    stall = 0
    can = np.array([[emp.max_shifts.get(sh.id, 1) > 0 for sh in inst.shifts] for emp in inst.staff])
    pb = penalty(inst, R)
    cover_share = (pb["cover under"] + pb["cover over"]) / max(pb["total"], 1)
    while time.time() < deadline - 0.05:
        it += 1
        mask = np.zeros((E, H), dtype=bool)
        kind = rng.random()
        if cover_share > 0.5:                 # cover dominates the penalty: focus on it more often
            kind = 0.35 + 0.65 * kind if rng.random() < 0.3 else kind
        if kind < 0.35 or not inst.cover:
            # a few employees over a stretch of days
            k = int(min(E, rng.integers(2, 7)))
            L = int(min(H, max(7, V // k)))
            a = int(rng.integers(0, H - L + 1))
            mask[rng.choice(E, size=k, replace=False), a:a + L] = True
        elif kind < 0.65:
            # many employees over a few days
            L = int(min(H, rng.integers(2, 15)))
            k = int(min(E, max(2, V // L)))
            a = int(rng.integers(0, H - L + 1))
            mask[rng.choice(E, size=k, replace=False), a:a + L] = True
        else:
            # around one costly cover cell (short- or over-staffed shift)
            d0, s0 = _worst_cover(inst, R, rng, slice(0, H))
            L = int(min(H, rng.integers(3, 12)))
            a = int(min(max(0, d0 - L // 2), H - L))
            k = int(min(E, max(2, V // L)))
            on = [e for e in range(E) if R[e, d0] == s0]
            # who could take this shift: allowed to work it, not on leave that day
            able = [e for e in range(E) if R[e, d0] != s0 and can[e, s0] and d0 not in inst.days_off.get(e, ())]
            rest = [e for e in range(E) if R[e, d0] != s0 and e not in set(able)]
            pool = (list(rng.permutation(on))[:max(1, k // 3)] + list(rng.permutation(able)) +
                    list(rng.permutation(rest)))
            mask[np.array(pool[:k], dtype=int), a:a + L] = True
        mask &= ~fm
        if not mask.any():
            continue
        left = deadline - time.time()
        sub = Sub(inst, R, mask, keep, keep_weight, extra_off, fair)
        sub.hint(R)
        R2, st, sv = sub.solve(min(sub_time, left), seed=int(rng.integers(1 << 30)))
        cur_before = cur
        if R2 is not None:
            new = score(R2)
            if new < cur - 1e-9:
                R, cur, stall = R2, new, 0
                history.append((time.time() - t0, cur))
                if progress:
                    progress(history[-1])
            else:
                stall += 1
                if new <= cur + 1e-9:
                    R = R2              # sideways moves keep the search moving
            # adapt the neighbourhood: solved fast to optimality -> bigger, timed out -> smaller
            n_opt += st == cp_model.OPTIMAL
            improved = new < cur_before - 1e-9
            if st == cp_model.OPTIMAL and sv.WallTime() < 0.3 * sub_time:
                if V >= vmax:                 # neighbourhoods are the whole problem: give them more time
                    sub_time = min(10.0, sub_time * 1.2)
                V = min(vmax, V * 1.15)
            elif st != cp_model.OPTIMAL and not improved:
                V = max(VMIN, V / 1.15)
        else:
            V = max(VMIN, V / 1.15)
    lns.stats = {'V': V, 'opt': n_opt, 'it': it}
    return Result(R, cur, bound, status, time.time() - t0, history, it, first)


def _worst_cover(inst, R, rng, days):
    cand, wts = [], []
    for (d, s), (req, wu, wo) in inst.cover.items():
        if not (days.start <= d < days.stop):
            continue
        n = int(np.sum(R[:, d] == s))
        c = wu * max(0, req - n) + wo * max(0, n - req)
        if c > 0:
            cand.append((d, s))
            wts.append(c)
    if not cand:
        d = int(rng.integers(days.start, days.stop))
        return d, int(rng.integers(inst.S))
    p = np.array(wts, float)
    return cand[int(rng.choice(len(cand), p=p / p.sum()))]


def check_cells(inst, R, extra_off):
    """Hard-rule problems of a roster, including cells that must now be off."""
    probs = check(inst, R)
    if extra_off is not None and np.any((R != OFF) & extra_off):
        probs.append("works on a new day off")
    return probs


def solve(inst: Instance, seconds: float = 60.0, seed: int = 0, progress=None, **kw) -> Result:
    return lns(inst, seconds=seconds, seed=seed, progress=progress, **kw)


def verify(inst: Instance, res: Result) -> list:
    return check(inst, res.roster) if res.ok else [f"no roster ({res.status})"]

"""
Shift Roster
------------
Build a fair staff roster in a minute: every rule kept (rest between shifts, hours, runs of days, weekends, leave),
as many shift requests granted as possible, and the cover the ward needs. Repair it in seconds when someone calls in
sick. Tested on the 24 public nurse rostering benchmark instances against the best known rosters.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import copy
import html
import json
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from roster import changes, data, report, ward
from roster.problem import OFF
from roster.solver import solve

st.set_page_config(page_title="Shift Roster", page_icon="📅", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "benchmarks", "results.json")
C_BLUE, C_GREY, C_OK, C_BAD, C_WARN, C_VIOLET = "#3b82f6", "#8b949e", "#2ea043", "#f85149", "#f0883e", "#a371f7"
SHIFT_COLORS = report.COLORS

ss = st.session_state
for key, default in {"ward": None, "inst": None, "key": None, "version": 0, "R": None, "res": None, "plan_sig": None,
                     "hand": None, "fix": None, "whatif": None, "changed": None, "seconds": 60, "fair": True,
                     "keep_level": "Medium"}.items():
    ss.setdefault(key, default)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def layout(fig, height=320, **kw):
    top = 70 if kw.get("title") else 30
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)")
    return fig


def load(w, inst, key):
    ss.ward, ss.inst, ss.key = w, inst, key
    ss.R, ss.res, ss.plan_sig, ss.hand, ss.fix, ss.whatif, ss.changed = None, None, None, None, None, None, None
    ss.fair = w is not None
    ss.version += 1


def sig():
    return (ss.key, ss.version)


def fair_spec(inst):
    """Share night shifts in proportion to contract hours among those who may work nights."""
    nights = [s for s, sh in enumerate(inst.shifts) if sh.id.upper().startswith("N") or "night" in sh.name.lower()]
    if not nights:
        return None
    need = sum(v[0] for (d, s), v in inst.cover.items() if s in nights)
    able = [e for e, emp in enumerate(inst.staff) if any(emp.max_shifts.get(inst.shifts[s].id, 1) > 0 for s in nights)]
    tot = sum(inst.staff[e].min_minutes for e in able) or 1
    targets = {e: need * inst.staff[e].min_minutes / tot for e in able}
    return {"shifts": nights, "weight": 3, "targets": targets}


def plan_ok():
    return ss.R is not None and ss.plan_sig == sig()


def name(inst, e):
    return inst.staff[e].name or inst.staff[e].id


def roster_figure(inst, R, changed=None, rows=None, height=None, days=None):
    """The roster as a coloured grid: one row per person, one column per day."""
    rows = list(range(inst.E)) if rows is None else rows
    compact = days is not None
    days = list(range(inst.horizon)) if days is None else list(days)
    z, text, hover = [], [], []
    for e in rows:
        zr, tr, hr = [], [], []
        for d in days:
            v = int(R[e, d])
            leave = d in inst.days_off.get(e, ())
            zr.append(v if v != OFF else (-2 if leave else -1))
            lab = inst.shifts[v].id if v != OFF else ("L" if leave else "")
            if changed is not None and changed[e, d]:
                lab = f"<b>{lab or '·'}</b>"
            tr.append(lab)
            hr.append(f"{name(inst, e)} · {report.day_label(inst, d, False)}<br>"
                      f"{(inst.shifts[v].name or inst.shifts[v].id) + ' ' + inst.shifts[v].start if v != OFF else ('leave' if leave else 'off')}"
                      + ("<br><b>changed</b>" if changed is not None and changed[e, d] else ""))
        z.append(zr)
        text.append(tr)
        hover.append(hr)
    S = inst.S
    vals = [-2, -1] + list(range(S))
    cols = ["#2b2f36", "#161b22"] + [SHIFT_COLORS[s % len(SHIFT_COLORS)] for s in range(S)]
    n = len(vals)
    scale = []
    for k, c in enumerate(cols):
        scale += [[k / n, c], [(k + 1) / n, c]]
    fig = go.Figure(go.Heatmap(z=np.array(z, float), text=text, texttemplate="%{text}", hovertext=hover,
                               hoverinfo="text", colorscale=scale, zmin=-2.5, zmax=S - 0.5, showscale=False,
                               xgap=2, ygap=2, textfont=dict(size=11, color="#ffffff")))
    fig.update_xaxes(tickvals=list(range(len(days))),
                     ticktext=[(report.tick if compact else report.day_label)(inst, d) for d in days],
                     side="top", tickangle=0 if len(days) <= 28 else -90, tickfont=dict(size=10))
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=[name(inst, e) for e in rows], autorange="reversed",
                     tickfont=dict(size=11))
    for k, d in enumerate(days):
        if report.is_weekend(inst, d):
            fig.add_vrect(x0=k - 0.5, x1=k + 0.5, fillcolor="rgba(163,113,247,.10)", line_width=0, layer="below")
    if changed is not None:
        for i, e in enumerate(rows):
            for k, d in enumerate(days):
                if changed[e, d]:
                    fig.add_shape(type="rect", x0=k - 0.48, x1=k + 0.48, y0=i - 0.48, y1=i + 0.48,
                                  line=dict(color="#ffd33d", width=2.5))
    fig.update_layout(height=height or 60 + 26 * len(rows), margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def cover_figure(inst, R, height=230):
    ct = report.cover_table(inst, R)
    fig = go.Figure()
    for s, sh in enumerate(inst.shifts):
        c = ct[ct["shift"] == sh.id]
        fig.add_trace(go.Bar(x=c["day"], y=c["rostered"], name=sh.id, marker_color=SHIFT_COLORS[s % len(SHIFT_COLORS)],
                             hovertemplate=f"{sh.id}: %{{y}} rostered<extra></extra>"))
    sh_t = ct.groupby("day")[["short"]].sum().reset_index()
    need = ct.groupby("day")[["needed"]].sum().reset_index()
    fig.add_trace(go.Scatter(x=need["day"], y=need["needed"], mode="lines", line=dict(color="#e6edf3", dash="dot",
                                                                                         shape="hvh"),
                             name="needed", hovertemplate="needed %{y}<extra></extra>"))
    bad = sh_t[sh_t.short > 0]
    fig.add_trace(go.Scatter(x=bad["day"], y=need.set_index("day").loc[bad["day"], "needed"] + 0.6, mode="text",
                             text=[f"−{v}" for v in bad["short"]], textfont=dict(color=C_BAD, size=13),
                             name="short", hoverinfo="skip"))
    layout(fig, height, barmode="stack", title="People on shift each day, and what is needed")
    fig.update_xaxes(tickvals=list(range(inst.horizon)), ticktext=[report.tick(inst, d) for d in range(inst.horizon)],
                     tickangle=0 if inst.horizon <= 28 else -90, tickfont=dict(size=10))
    return fig


# ---------------------------------------------------------------------------
# sidebar + header
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📅 Shift Roster")
    st.caption("Fair staff rosters that keep every rule, and quick repairs when someone calls in sick.")
    st.divider()
    side = st.container()
    st.divider()
    st.caption("CP-SAT (Google OR-Tools) with large neighbourhood search. See **ℹ️ Methodology**.")

st.title("📅 Shift Roster")
st.caption("A roster for the whole team in a minute: **every rule kept** (rest between shifts, hours, runs of days, "
           "weekends, leave), **as many requests granted as possible**, and the **cover the ward needs**. Repaired in "
           "seconds when someone calls in sick.")
tabs = st.tabs(["📥 Team", "⚙️ Roster", "📅 Calendar", "⚖️ Fairness", "🤒 Sick call", "🔍 What would help",
                "🧪 Benchmark", "ℹ️ Methodology"])

# ---------------------------------------------------------------------------
# 1. Team
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Demo: a hospital ward**")
        st.caption("18 nurses, Day / Evening / Night shifts, 4 weeks, leave and 25 requests. Fictional names.")
        if st.button("Load the demo", type="primary", width="stretch"):
            w = ward.demo()
            load(w, ward.to_instance(w), "demo")
            st.rerun()
    with c2:
        st.markdown("**Benchmark: nurse rostering instances**")
        st.caption("The 24 Curtois & Qu instances (8 to 150 staff, 2 to 52 weeks) with the best known rosters.")
        bk = st.selectbox("Instance", list(data.BEST), index=6, label_visibility="collapsed")
        if st.button("Load", width="stretch", key="load_bench"):
            try:
                load(None, data.benchmark(bk), bk)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load {bk}: {exc}")
    with c3:
        st.markdown("**Your team**")
        st.caption("One Excel workbook: shifts, staff and contracts, cover needed, leave, requests. Start from the "
                   "template (the demo ward).")
        up = st.file_uploader("Workbook", type=["xlsx"], label_visibility="collapsed")
        if up is not None and st.button(f"Use {up.name}", type="primary", width="stretch"):
            try:
                w = ward.from_excel(up.getvalue())
                load(w, ward.to_instance(w), f"upload:{up.name}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.download_button("Template (the demo ward, Excel)", ward.to_excel(ward.demo()), "roster_template.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    inst = ss.inst
    if inst is None:
        st.info("Load the demo, a benchmark instance or your own team to begin.")
    else:
        st.divider()
        st.subheader(ss.ward.name if ss.ward is not None else inst.name)
        st.caption(inst.description)
        need = sum(v[0] for v in inst.cover.values())
        cap = sum(e.max_minutes for e in inst.staff) / max(np.mean([s.minutes for s in inst.shifts]), 1)
        m = st.columns(5)
        m[0].metric("Staff", inst.E)
        m[1].metric("Shifts to fill", f"{need:,}", help="Sum of the cover needed over the period")
        m[2].metric("Shifts the team can work, at most", f"{cap:,.0f}", help="From everyone's maximum hours")
        m[3].metric("Leave days", sum(len(v) for v in inst.days_off.values()))
        m[4].metric("Requests", len({(e, d) for e, d, s, w in inst.on_requests + inst.off_requests}))
        g1, g2 = st.columns([3, 2])
        with g1:
            z = np.array([[inst.cover.get((d, s), (0, 0, 0))[0] for d in range(inst.horizon)] for s in range(inst.S)])
            fig = go.Figure(go.Heatmap(z=z, x=list(range(inst.horizon)), y=[sh.id for sh in inst.shifts],
                                       colorscale=[[0, "#161b22"], [1, C_BLUE]], text=z, texttemplate="%{text}",
                                       hovertemplate="%{y}, day %{x}: %{z} needed<extra></extra>", showscale=False,
                                       xgap=2, ygap=2))
            layout(fig, 90 + 26 * inst.S, title="Cover needed per shift and day")
            fig.update_xaxes(tickvals=list(range(inst.horizon)),
                             ticktext=[report.tick(inst, d) for d in range(inst.horizon)],
                             tickangle=0 if inst.horizon <= 28 else -90, tickfont=dict(size=10))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")
        with g2:
            sh = pd.DataFrame([{"shift": s.id, "name": s.name, "starts": s.start, "hours": s.minutes / 60,
                                "may not be followed next day by": ", ".join(sorted(s.cannot_follow)) or "—"}
                               for s in inst.shifts])
            st.dataframe(sh, hide_index=True, width="stretch")
        if ss.ward is not None:
            with st.expander("Staff and contracts (editable: change a limit, then build the roster again)"):
                edited = st.data_editor(ss.ward.staff, hide_index=True, width="stretch", num_rows="fixed",
                                        key=f"staff_{ss.version}", disabled=["name"])
                if not edited.equals(ss.ward.staff):
                    try:
                        w2 = copy.deepcopy(ss.ward)
                        w2.staff = edited.reset_index(drop=True)
                        inst2 = ward.to_instance(w2)
                        ss.ward, ss.inst = w2, inst2
                        ss.version += 1
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            l1, l2 = st.columns(2)
            with l1.expander(f"Leave ({len(ss.ward.leave)})"):
                st.dataframe(ss.ward.leave, hide_index=True, width="stretch")
            with l2.expander(f"Requests ({len(ss.ward.requests)})"):
                st.dataframe(ss.ward.requests, hide_index=True, width="stretch")
        else:
            st.dataframe(pd.DataFrame([{"staff": e.id, **{f"max {k}": (v if v < 10 ** 5 else "") for k, v in e.max_shifts.items()},
                                        "min hours": e.min_minutes / 60, "max hours": e.max_minutes / 60,
                                        "max in a row": e.max_consec, "min in a row": e.min_consec,
                                        "min off in a row": e.min_consec_off, "max weekends": e.max_weekends}
                                       for e in inst.staff]), hide_index=True, width="stretch", height=260)

# ---------------------------------------------------------------------------
# 2. Roster
# ---------------------------------------------------------------------------
with tabs[1]:
    inst = ss.inst
    if inst is None:
        st.info("Load a team first (📥 Team).")
    else:
        b1, b2, b3 = st.columns([1, 2, 2])
        go_btn = b1.button("Build roster", type="primary", width="stretch")
        b2.slider("Search time (seconds)", 5, 300, key="seconds",
                  help="Longer = better rosters; small teams are often proven optimal in seconds.")
        b3.toggle("Share night shifts fairly", key="fair", disabled=fair_spec(inst) is None,
                  help="Nights in proportion to contract hours (as long as cover and requests allow).")
        if go_btn:
            bar = st.progress(0.0, "Building…")
            live = st.empty()
            pts, last = [], [0.0, 0]
            t0 = time.time()

            def prog(h):
                pts.append(h)
                now = time.time()
                bar.progress(min(1.0, (now - t0) / ss.seconds), f"{now - t0:.0f} s · best so far: penalty {h[1]:,.0f}")
                if now - last[0] > 1.0 and len(pts) > 1:
                    a = np.array(pts)
                    fig = go.Figure(go.Scatter(x=a[:, 0], y=a[:, 1], mode="lines", line=dict(color=C_BLUE, width=2.5,
                                                                                         shape="hv")))
                    layout(fig, 200, title="Best roster so far (penalty)", xaxis_range=[0, ss.seconds])
                    last[1] += 1
                    live.plotly_chart(fig, width="stretch", key=f"live_{last[1]}")
                    last[0] = now

            fair = fair_spec(inst) if ss.fair else None
            res = solve(inst, seconds=float(ss.seconds), seed=0, progress=prog, fair=fair)
            bar.empty()
            live.empty()
            ss.res, ss.R, ss.plan_sig, ss.fix, ss.whatif, ss.changed = res, res.roster, sig(), None, None, None
            G = report.greedy(inst)
            ss.hand = {"R": G, **report.summary(inst, G)}
        if ss.R is None:
            st.info("Press **Build roster**.")
        elif not plan_ok():
            st.warning("The team changed since this roster: build it again.")
        else:
            R, res = ss.R, ss.res
            S_ = report.summary(inst, R)
            if S_["problems"]:
                st.error("The checker found broken rules: " + "; ".join(S_["problems"][:3]))
            else:
                st.success(f"✅ **Independent check passed:** every rule kept for all {inst.E} staff: rest between "
                           "shifts, hours, runs of working days and days off, weekends, leave.")
            H_ = ss.hand
            m = st.columns(5)
            m[0].metric("Rules broken", len(S_["problems"]), delta=f"{len(S_['problems']) - len(H_['problems'])} vs by hand",
                        delta_color="inverse")
            m[1].metric("Shifts short", S_["short"], delta=f"{S_['short'] - H_['short']:+d} vs by hand",
                        delta_color="inverse", help="Shifts with fewer people than needed (person-shifts)")
            m[2].metric("Requests granted", f"{S_['granted']} of {S_['requests']}",
                        delta=f"{S_['granted'] - H_['granted']:+d} vs by hand")
            m[3].metric("Extra people on shift", S_["extra"], help="Above the cover needed (minimum hours force some)")
            if ss.key in data.BEST:
                best = data.BEST[ss.key]
                m[4].metric("Penalty vs best known", f"{S_['penalty']:,}",
                            delta=f"{S_['penalty'] / best - 1:+.1%} vs {best:,}", delta_color="inverse")
            else:
                m[4].metric("Status", "optimal ✔" if res.status == "optimal" else "best found",
                            help=f"{res.iterations:,} neighbourhoods searched in {res.seconds:.0f} s")
            st.caption(f"*By hand* = filling each day in turn with whoever is free, may follow yesterday's shift and has "
                       f"worked least, without looking ahead (what a busy manager does). It broke {len(H_['problems'])} "
                       f"rules, left {H_['short']} shifts short and granted {H_['granted']} of {H_['requests']} requests.")
            g1, g2 = st.columns([3, 2])
            with g1:
                st.plotly_chart(cover_figure(inst, R), width="stretch", key="cover_roster")
            with g2:
                cats = ["Shifts short", "Requests not granted", "Extra people"]
                fig = go.Figure()
                for lab, col, src in (("by hand", C_GREY, H_), ("this roster", C_BLUE, S_)):
                    fig.add_trace(go.Bar(x=cats, y=[src["short"], src["requests"] - src["granted"], src["extra"]],
                                         name=lab, marker=dict(color=col, cornerradius=4)))
                layout(fig, 230, title="This roster vs by hand", barmode="group")
                st.plotly_chart(fig, width="stretch")
            if H_["problems"]:
                kinds = {}
                for p in H_["problems"]:
                    k = p.split(": ", 1)[1]
                    k = ("single working day between days off" if "working days in a row" in k else
                         "single day off between shifts" if "days off in a row" in k else
                         "fewer hours than the contract" if "at least" in k and "minutes" in k else
                         "too many weekends" if "weekends" in k else "rest between shifts" if " after " in k else
                         "too many days in a row" if "in a row" in k else k)
                    kinds[k] = kinds.get(k, 0) + 1
                st.caption("Rules broken by hand: " + ", ".join(f"{v} × {k}" for k, v in
                                                                sorted(kinds.items(), key=lambda x: -x[1])))

# ---------------------------------------------------------------------------
# 3. Calendar
# ---------------------------------------------------------------------------
with tabs[2]:
    inst = ss.inst
    if not plan_ok():
        st.info("Build the roster first (⚙️ Roster).")
    else:
        R = ss.R
        legend = " · ".join(f"<span style='color:{SHIFT_COLORS[s % len(SHIFT_COLORS)]}'>■</span> {html.escape(sh.id)} "
                            f"{html.escape(sh.name)} {sh.start}" for s, sh in enumerate(inst.shifts[:12]))
        st.markdown(legend + " · L leave · <span style='color:#a371f7'>▒</span> weekend"
                    + (" · <span style='color:#ffd33d'>▢</span> changed after the sick call" if ss.changed is not None else ""),
                    unsafe_allow_html=True)
        rows = None
        if inst.E > 40:
            pick = st.slider("Staff shown", 1, inst.E, (1, 40))
            rows = list(range(pick[0] - 1, pick[1]))
        st.plotly_chart(roster_figure(inst, R, ss.changed, rows), width="stretch", key="roster_calendar")
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Roster (Excel, coloured)", report.roster_excel(inst, R, ss.ward.name if ss.ward else inst.name,
                                                                          ss.changed),
                           "roster.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width="stretch")
        d2.download_button("Shifts per person (CSV)", report.person_csv(inst, R), "shifts.csv", "text/csv",
                           width="stretch")
        st.plotly_chart(cover_figure(inst, R), width="stretch", key="cover_calendar")

# ---------------------------------------------------------------------------
# 4. Fairness
# ---------------------------------------------------------------------------
with tabs[3]:
    inst = ss.inst
    if not plan_ok():
        st.info("Build the roster first (⚙️ Roster).")
    else:
        R = ss.R
        ps = report.person_stats(inst, R)
        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure()
            fig.add_trace(go.Bar(y=ps["name"], x=ps["max hours"] - ps["min hours"], base=ps["min hours"],
                                 orientation="h", marker_color="rgba(139,148,158,.25)", name="contract range",
                                 hovertemplate="%{y}: %{base:.0f}-%{x:.0f} h<extra></extra>"))
            fig.add_trace(go.Scatter(y=ps["name"], x=ps["hours"], mode="markers", name="rostered",
                                     marker=dict(color=C_BLUE, size=10, symbol="diamond"),
                                     hovertemplate="%{y}: %{x:.0f} h<extra></extra>"))
            layout(fig, 80 + 22 * inst.E, title="Hours: each person inside their contract")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")
        with g2:
            fig = go.Figure()
            for s, sh in enumerate(inst.shifts[:12]):
                fig.add_trace(go.Bar(y=ps["name"], x=ps[sh.id], name=sh.id, orientation="h",
                                     marker_color=SHIFT_COLORS[s % len(SHIFT_COLORS)]))
            layout(fig, 80 + 22 * inst.E, title="Shifts of each type", barmode="stack")
            fig.update_yaxes(autorange="reversed", showticklabels=False)
            st.plotly_chart(fig, width="stretch")
        g3, g4 = st.columns(2)
        with g3:
            fig = go.Figure(go.Histogram(x=ps["weekends"], marker_color=C_VIOLET, xbins=dict(size=1)))
            layout(fig, 240, title="Weekends worked per person", bargap=0.1)
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, width="stretch")
        with g4:
            rq = report.requests_table(inst, R)
            if len(rq):
                fig = go.Figure()
                for g, col, lab in ((True, C_OK, "granted"), (False, C_BAD, "not granted")):
                    sub = rq[rq.granted == g]
                    fig.add_trace(go.Histogram(x=sub["priority"], name=lab, marker_color=col, xbins=dict(size=1)))
                layout(fig, 240, title="Requests by priority", barmode="stack", bargap=0.1)
                fig.update_xaxes(dtick=1, title="priority")
                st.plotly_chart(fig, width="stretch")
        st.dataframe(ps.round(1), hide_index=True, width="stretch")
        rq = report.requests_table(inst, R)
        if len(rq):
            with st.expander(f"Requests ({int(rq.granted.sum())} of {len(rq)} granted)"):
                st.dataframe(rq.sort_values(["granted", "priority"], ascending=[True, False]), hide_index=True,
                             width="stretch")

# ---------------------------------------------------------------------------
# 5. Sick call
# ---------------------------------------------------------------------------
with tabs[4]:
    inst = ss.inst
    if not plan_ok():
        st.info("Build the roster first (⚙️ Roster).")
    else:
        R = ss.R
        st.markdown("**Someone called in sick, or more cover is needed?** The roster is repaired from today on, "
                    "changing as few published shifts as possible. Days before today stay as worked.")
        c1, c2, c3 = st.columns(3)
        labels = [report.day_label(inst, d, False) for d in range(inst.horizon)]
        with c1:
            today = st.selectbox("Today", list(range(inst.horizon)), index=min(9, inst.horizon - 1),
                                 format_func=lambda d: labels[d])
            who = st.selectbox("Who is away", [None] + list(range(inst.E)),
                               format_func=lambda e: "nobody" if e is None else name(inst, e))
            span = st.slider("Days away (from today)", 1, 14, 3, disabled=who is None)
        with c2:
            add_day = st.selectbox("Extra cover needed on", [None] + list(range(today, inst.horizon)),
                                   format_func=lambda d: "no extra cover" if d is None else labels[d])
            add_shift = st.selectbox("Shift", list(range(inst.S)), format_func=lambda s: inst.shifts[s].id,
                                     disabled=add_day is None)
            add_n = st.number_input("Extra people", 1, 10, 1, disabled=add_day is None)
        with c3:
            st.radio("Avoid changing published shifts", ["Low", "Medium", "High"], key="keep_level", horizontal=True,
                     help="High: fewer changes, may leave a shift short. Low: fills more gaps, more changes.")
            secs = st.slider("Search time (seconds)", 5, 120, 15, key="fix_secs")
            go_fix = st.button("Repair the roster", type="primary", width="stretch",
                               disabled=who is None and add_day is None)
        if go_fix:
            absn = [(who, today, min(inst.horizon - 1, today + span - 1))] if who is not None else []
            extra = {(add_day, add_shift): int(add_n)} if add_day is not None else None
            kw = {"Low": 2, "Medium": 10, "High": 40}[ss.keep_level]
            with st.spinner("Repairing…"):
                out = changes.reroster(inst, R, today, absn, extra, seconds=float(secs), keep_weight=kw)
            out["absences"], out["extra"], out["today"] = absn, extra, today
            ss.fix = out
        F = ss.fix
        if F:
            new, chg = F["roster"], F["changed"]
            if F["problems"]:
                st.error("The checker found broken rules: " + "; ".join(F["problems"][:3]))
            else:
                st.success("✅ Repaired and checked: from today on, every rule holds for everyone (minimum runs are "
                           "not checked right next to the absence).")
            away = {(e, d) for e, a, b in F["absences"] for d in range(a, b + 1)}
            moves = [mv for mv in F["moves"] if (mv["e"], mv["day"]) not in away]
            sub, st0 = F["sub"], F["start"]
            nothing = R.copy()
            for e, d in away:
                nothing[e, d] = OFF
            t_ = F["today"] - st0
            short_nothing = int(report.cover_table(sub, nothing[:, st0:]).query("day >= @t_").short.sum())
            short_now = int(report.cover_table(sub, new[:, st0:]).query("day >= @t_").short.sum())
            lost = int(sum(1 for e, d in away if R[e, d] != OFF))
            m = st.columns(4)
            m[0].metric("Shifts lost to the absence", lost)
            m[1].metric("Shifts short from today", short_now, delta=f"{short_now - short_nothing:+d} vs doing nothing",
                        delta_color="inverse")
            m[2].metric("Other people's shifts changed", len(moves))
            m[3].metric("People affected", len({mv["e"] for mv in moves}))
            g1, g2 = st.columns([3, 2])
            with g1:
                rows = sorted({mv["e"] for mv in F["moves"]})
                if rows:
                    win = range(max(0, F["today"] - 2), min(inst.horizon, F["today"] + 16))
                    st.plotly_chart(roster_figure(inst, new, chg, rows, days=win), width="stretch", key="roster_fix")
            with g2:
                if moves:
                    st.dataframe(pd.DataFrame([{"who": name(inst, mv["e"]), "day": labels[mv["day"]], "was": mv["was"],
                                                "now": mv["now"]} for mv in moves]), hide_index=True, width="stretch")
                else:
                    st.caption("No one else's shifts change.")
                if st.button("Use the repaired roster", type="primary", disabled=bool(F["problems"])):
                    # the absence becomes leave: its shifts count towards the contract; minimum runs are not
                    # checked right next to it
                    x = copy.deepcopy(ss.inst)
                    for e, a, b in F["absences"]:
                        lost_min = sum(x.shifts[R[e, d]].minutes for d in range(a, b + 1) if R[e, d] != OFF)
                        x.days_off.setdefault(e, set()).update(range(a, b + 1))
                        x.staff[e].min_minutes = max(0, x.staff[e].min_minutes - lost_min)
                        x.staff[e].max_minutes = max(0, x.staff[e].max_minutes - lost_min)
                        x.breaks.setdefault(e, set()).update(range(a, b + 1))
                    for (d, s_), n in (F["extra"] or {}).items():
                        req, wu, wo = x.cover.get((d, s_), (0, 100, 1))
                        x.cover[(d, s_)] = (req + n, wu, wo)
                    ss.inst, ss.R, ss.changed, ss.fix = x, new, chg, None
                    st.rerun()

# ---------------------------------------------------------------------------
# 6. What would help
# ---------------------------------------------------------------------------
LEVERS = {
    "weekend": "One more weekend allowed for everyone",
    "overtime": "Up to 8 hours of overtime for everyone",
    "runs": "Single working days and single days off allowed",
    "hire": "One more staff member (the most common contract, no requests)",
}


def apply_lever(inst, lever):
    x = copy.deepcopy(inst)
    if lever == "weekend":
        for e in x.staff:
            e.max_weekends += 1
    elif lever == "overtime":
        for e in x.staff:
            e.max_minutes += 480
    elif lever == "runs":
        for e in x.staff:
            e.min_consec, e.min_consec_off = 1, 1
    elif lever == "hire":
        roles = [e.role or str(e.max_minutes) for e in x.staff]
        common = max(set(roles), key=roles.count)
        ref = x.staff[roles.index(common)]
        new = copy.deepcopy(ref)
        new.id, new.name, new.role = f"E{x.E + 1}", "New staff member", ref.role
        x.staff.append(new)
        x.days_off[x.E - 1] = set()
    return x


with tabs[5]:
    inst = ss.inst
    if not plan_ok():
        st.info("Build the roster first (⚙️ Roster).")
    else:
        st.markdown("**Shifts still short? See what would fix them.** Each option is rostered again from scratch and "
                    "compared with the current roster.")
        picks = st.multiselect("Options to try", list(LEVERS), default=["weekend", "overtime", "hire"],
                               format_func=LEVERS.get)
        secs = st.slider("Search time per option (seconds)", 5, 120, 20, key="wi_secs")
        if st.button("Compare", type="primary", disabled=not picks):
            rows = []
            bar = st.progress(0.0, "Trying options…")
            for k, lv in enumerate(picks):
                bar.progress(k / len(picks), f"Trying: {LEVERS[lv]}")
                x = apply_lever(inst, lv)
                r = solve(x, seconds=float(secs), seed=0)
                s_ = report.summary(x, r.roster)
                rows.append({"option": LEVERS[lv], "short": s_["short"], "requests granted": s_["granted"],
                             "requests": s_["requests"], "valid": not s_["problems"]})
            bar.empty()
            ss.whatif = rows
        if ss.whatif:
            cur = report.summary(inst, ss.R)
            rows = [{"option": "Current rules", "short": cur["short"], "requests granted": cur["granted"],
                     "requests": cur["requests"], "valid": not cur["problems"]}] + ss.whatif
            df = pd.DataFrame(rows)
            fig = go.Figure(go.Bar(y=df["option"], x=df["short"], orientation="h",
                                   marker=dict(color=[C_GREY] + [C_BLUE] * (len(df) - 1), cornerradius=4),
                                   text=df["short"], textposition="outside",
                                   hovertemplate="%{y}: %{x} shifts short<extra></extra>"))
            layout(fig, 90 + 46 * len(df), title="Shifts short with each option")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# 7. Benchmark
# ---------------------------------------------------------------------------
with tabs[6]:
    if not os.path.exists(RESULTS):
        st.info("Run `python benchmarks/run_benchmark.py` to create the benchmark results.")
    else:
        res = json.load(open(RESULTS))
        rows = res["instances"]
        gaps = np.array([max(0.0, r["gap"]) for r in rows])
        st.markdown(f"#### Curtois & Qu nurse rostering benchmark: {len(rows)} instances, "
                    f"{res['seconds_per_instance']:.0f} seconds each")
        st.caption("Instances from schedulingbenchmarks.org (8 to 120 staff, 2 to 12 weeks here). Best known rosters "
                   "are proven optimal. Every roster below passed this app's independent checker; penalties are "
                   "recomputed from the roster.")
        m = st.columns(5)
        m[0].metric("Valid rosters", f"{sum(r['valid'] for r in rows)} of {len(rows)}")
        m[1].metric("Best known matched", f"{int(np.sum(gaps < 1e-9))} of {len(rows)}")
        m[2].metric("Proven optimal here", f"{sum(r['status'] == 'optimal' for r in rows)}")
        m[3].metric("Median gap", f"{np.median(gaps):.1%}")
        m[4].metric("First roster, median gap", f"{np.median([r['first'] / r['best'] - 1 for r in rows]):.0%}")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[r["name"].replace("Instance", "") for r in rows], y=[100 * g for g in gaps],
                             marker=dict(color=[C_OK if g < 1e-9 else C_BLUE for g in gaps], cornerradius=3),
                             customdata=[[r["staff"], r["days"], r["penalty"], r["best"]] for r in rows],
                             hovertemplate="Instance %{x}: %{customdata[0]} staff, %{customdata[1]} days<br>"
                                           "penalty %{customdata[2]:,} vs best %{customdata[3]:,} (%{y:.1f}%)<extra></extra>"))
        layout(fig, 320, title="Gap to the best known roster (green: matched)", yaxis_title="% above best known",
               xaxis_title="instance")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame([{"instance": r["name"], "staff": r["staff"], "days": r["days"],
                                    "shift types": r["shift_types"], "best known": r["best"], "this app": r["penalty"],
                                    "gap": f"{max(0, r['gap']):.2%}", "first roster": r["first"],
                                    "status": r["status"], "valid": "✅" if r["valid"] else "❌"} for r in rows]),
                     hide_index=True, width="stretch")
        st.caption(f"{res.get('machine', '')}. Re-run with `python benchmarks/run_benchmark.py`. Instances 20-24 (half "
                   "a year to a year, 50-150 staff) need far longer than this time budget.")

# ---------------------------------------------------------------------------
# 8. Methodology
# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("""
#### The problem
Hospitals, care homes, pharmacies, factories and call centres build rosters by hand, often in a spreadsheet, every
month. Every person has a contract (hours, nights, weekends), nobody may start an early shift straight after a night,
people ask for days off, and the ward needs enough people on every shift. By hand this takes days, breaks rules
(single days off, too many nights in a row) and feels unfair. It is the nurse rostering problem, one of the classic
problems of operations research.

#### Hard rules (always kept)
One shift a day · shift types that may not follow each other · at most so many shifts of each type · minimum and
maximum hours · at most so many working days in a row · no isolated working days or days off (minimum runs) · at most
so many weekends · leave.

#### Soft goals (penalties)
A person short on a shift (default 100) · a person extra (1) · a request not granted (its priority) · optionally,
nights above a person's fair share.

#### How the roster is built
1. **First roster:** each person on their own (every hard rule belongs to one person): a valid pattern of working days
   that meets their hours.
2. **Whole problem at once** for small teams: CP-SAT (Google OR-Tools) with its LP relaxation, which often proves
   the roster optimal.
3. **Large neighbourhood search:** free a few people over a few weeks, or many people over a few days, or the people
   around a short shift; keep everyone else; let CP-SAT re-roster that part exactly. Repeat for the time budget,
   adapting the size of the part.

#### Sick calls
The roster is repaired from today: hours, shifts and weekends already worked are taken off each limit, the days of
absence count towards the contract, and every changed shift costs a little, so few published shifts change.

#### Independent checker
Separate code re-checks every rule for every person and recomputes every penalty from the roster.

#### Limits
* One period at a time; nothing is carried over from the previous roster except in sick-call repairs.
* Skill mix (e.g. one senior per shift) is not modelled yet; it can be approximated with shift types per grade.
* Half-year and year rosters for 50+ staff need much longer than a minute.

#### Data and credits
Benchmark: T. Curtois and R. Qu (2014), *Computational results on new staff scheduling benchmark instances*,
schedulingbenchmarks.org. The demo ward is fictional, with generated names.
""")

# ---------------------------------------------------------------------------
# sidebar status
# ---------------------------------------------------------------------------
with side:
    if ss.inst is None:
        st.info("Start in **📥 Team**.")
    else:
        nm = ss.ward.name if ss.ward is not None else ss.inst.name
        st.markdown(f"**Team:** {html.escape(nm)}  \n{ss.inst.E} staff · {ss.inst.horizon // 7} weeks")
        if plan_ok():
            s_ = report.summary(ss.inst, ss.R)
            st.markdown(f"**Roster:** {len(s_['problems'])} rules broken · {s_['short']} short · "
                        f"{s_['granted']}/{s_['requests']} requests")
        else:
            st.markdown("**Roster:** not built yet")

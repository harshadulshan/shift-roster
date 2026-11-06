"""
App checks with Streamlit's AppTest: load the demo and a benchmark, build rosters, look at every tab, repair a sick
call and adopt it, compare what-if options.

    python tests/verify_app.py          # about 3 minutes
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

passed, failed = 0, []


def ok(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}  {detail}")


def button(at, label):
    return next(b for b in at.button if b.label == label)


def metric(at, label):
    return next((m for m in at.metric if m.label == label), None)


def texts(elements):
    return " | ".join(str(e.value) for e in elements)


at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=400)
at.run()
ok("starts without errors", not at.exception, at.exception)
ok("asks to load a team", "Load the demo" in texts(at.info))

print("Demo ward")
button(at, "Load the demo").click().run()
ok("demo loads: 18 staff, 320 shifts to fill", not at.exception and metric(at, "Staff").value == "18"
   and metric(at, "Shifts to fill").value == "320", at.exception)
at.slider(key="seconds").set_value(15).run()
button(at, "Build roster").click().run()
ok("builds without errors", not at.exception, at.exception)
ok("independent check passed", "Independent check passed" in texts(at.success), texts(at.error))
rb = metric(at, "Rules broken")
ok("0 rules broken, fewer than by hand", rb.value == "0" and rb.delta.startswith("-"), (rb.value, rb.delta))
sh = metric(at, "Shifts short")
ok("fewer shifts short than by hand", sh.delta.startswith("-"), sh.delta)
ok("calendar, cover, fairness charts drawn", len(at.get("plotly_chart")) >= 8, len(at.get("plotly_chart")))
ok("no stray values printed on the page", not any(m.value.strip() in ("None", "(None, None, None)") for m in at.markdown)
   and not at.json)
ok("sidebar shows the roster", any("0 rules broken" in m.value for m in at.sidebar.markdown))

print("Sick call")
who = next(s for s in at.selectbox if s.label == "Who is away")
who.select_index(4).run()
at.slider(key="fix_secs").set_value(6).run()
button(at, "Repair the roster").click().run()
ok("repair passes the checker", not at.exception and "Repaired and checked" in texts(at.success), texts(at.error))
lost = metric(at, "Shifts lost to the absence")
ok("shows the shifts lost and the changes", lost is not None and metric(at, "Other people's shifts changed") is not None)
button(at, "Use the repaired roster").click().run()
ok("adopted: still 0 rules broken, changes marked", not at.exception and metric(at, "Rules broken").value == "0"
   and any("changed after the sick call" in m.value for m in at.markdown), at.exception)

print("What would help")
at.slider(key="wi_secs").set_value(5).run()
ms = next(m for m in at.multiselect if m.label == "Options to try")
ms.set_value(["weekend", "hire"]).run()
button(at, "Compare").click().run()
ok("options compared", not at.exception and any(len(d.value) == 3 for d in at.dataframe if "option" in d.value.columns),
   at.exception)

print("Benchmark instance")
sel = next(s for s in at.selectbox if s.label == "Instance")
sel.set_value("Instance2").run()
button(at, "Load").click().run()
ok("Instance2 loads", not at.exception and any("Instance2" in s.value for s in at.subheader))
at.slider(key="seconds").set_value(20).run()
button(at, "Build roster").click().run()
pv = metric(at, "Penalty vs best known")
ok("Instance2 reaches the proven optimum 828", pv is not None and pv.value == "828", pv.value if pv else None)

print("Benchmark tab")
if os.path.exists(os.path.join(ROOT, "benchmarks", "results.json")):
    n = len(json.load(open(os.path.join(ROOT, "benchmarks", "results.json")))["instances"])
    ok(f"{n} of {n} valid", metric(at, "Valid rosters").value == f"{n} of {n}")

print(f"\n{passed} passed, {len(failed)} failed")
if failed:
    print("FAILED: " + "; ".join(failed))
sys.exit(1 if failed else 0)

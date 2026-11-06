<div align="center">

<img src="docs/hero.svg" alt="Shift Roster: a fictional ward roster made by hand breaks 140 rules and leaves 28 shifts short; Shift Roster's roster breaks none" width="100%">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![OR-Tools](https://img.shields.io/badge/OR--Tools-CP--SAT-4285F4)
![Benchmark](https://img.shields.io/badge/nurse%20rostering-19%2F19%20valid%2C%206%20proven%20optimal-8957e5)
![Checks](https://img.shields.io/badge/mutation%20test-13%2F13%20bugs%20caught-2EA44F)
![License](https://img.shields.io/badge/license-MIT-blue)

<img src="docs/scoreboard.svg" alt="Demo ward, 4 weeks: 0 rules broken (by hand 140), 11 shifts short (by hand 28), 20 of 25 requests granted (by hand 11); benchmark: best known matched on 6 of 19 instances, median gap 2.0%" width="100%">

</div>

## 🎯 The problem

Every month, ward sisters, care-home managers, factory supervisors and call-centre leads spend **days** building the
next roster by hand in a spreadsheet. Everyone has a contract (hours, nights, weekends), nobody may start a day shift
straight after a night, people ask for days off, and every shift needs enough people. Done by hand, rosters **break rest
rules** (single days off between shifts), **feel unfair** (always the same people on nights and weekends) and **fall
apart** when someone calls in sick.

**Shift Roster** builds the whole roster in a minute, keeps **every rule**, grants **as many requests as possible**,
shares nights fairly, shows exactly **where the team is short and what would fix it**, and repairs the roster in
seconds after a sick call, changing as few published shifts as possible.

## ✨ What it does

| Tab | What you get |
|---|---|
| 📥 **Team** | the demo ward, any of the 24 benchmark instances, or your own Excel workbook (shifts, staff and contracts, cover, leave, requests) |
| ⚙️ **Roster** | a roster in a minute with live progress, checked by independent code, compared with filling it by hand |
| 📅 **Calendar** | the roster as a coloured grid (weekends, leave, changes marked), the cover per day, a coloured Excel roster to print |
| ⚖️ **Fairness** | hours inside every contract, nights and weekends per person, requests granted by priority |
| 🤒 **Sick call** | who is away and for how long, or extra cover needed: repaired from today with the fewest changes |
| 🔍 **What would help** | still short? compare one more weekend each, overtime, relaxed runs or one more hire |

## 📸 The app

<img src="docs/screenshots/00_overview.png" alt="A roster for the demo ward: independent check passed; 0 rules broken (140 fewer than by hand), fewer shifts short, more requests granted" width="100%">

| **The team: cover needed and shift rules** | **The roster vs filling it by hand** |
|:---:|:---:|
| ![Cover needed per shift and day, and which shift may not follow which](docs/screenshots/01_team.png) | ![Metrics, people on shift per day vs needed, comparison with the by-hand roster](docs/screenshots/02_roster.png) |
| **The calendar** | **Fairness** |
| ![Roster grid: 18 nurses, 4 weeks, Day/Evening/Night, leave marked](docs/screenshots/03_calendar.png) | ![Hours inside each contract, shifts of each type, weekends per person, requests by priority](docs/screenshots/04_fairness.png) |
| **A sick call, repaired** | **What would fix the shortfall** |
| ![Sanduni is away for 3 days: the roster is repaired from today with a few changes, every rule kept](docs/screenshots/05_sickcall.png) | ![Shifts short with one more weekend each, overtime, or one more hire](docs/screenshots/06_whatif.png) |

## 🤒 Sick calls in seconds

<img src="docs/sickcall.svg" alt="Sanduni calls in sick for 3 days; 3 shifts become uncovered; the roster is repaired from today with a few changes for other people, every rule kept" width="100%">

Days already worked never change. Hours, shifts and weekends already worked are taken off each limit, the days away
count towards the contract (nobody makes them up), and every changed shift costs a little, so the repair touches as
few people as possible. The level is yours: **avoid changes** (may leave a shift short) or **fill every gap**.

## 🧪 Benchmark

<img src="docs/benchmark.svg" alt="19 Curtois and Qu instances, 300 s each, one thread: 19 valid, best known matched on 6 (all proven optimal here), median gap 2.0%" width="100%">

The [Curtois & Qu (2014)](https://www.schedulingbenchmarks.org/nrp/) instances are the standard test set for nurse
rostering, built from real hospital rules; every best known roster is proven optimal. Instances 1 to 19 (8 to 120 staff,
2 to 12 weeks) were rostered for 5 minutes each on one thread; every roster passed the independent checker.

| Measure | Result |
|---|---|
| Rosters valid | **19 of 19** |
| Best known matched (proven optimal by this app too) | **6** (instances 1 to 6) |
| Within 1% / 3% / 10% of the best known | 9 / 13 / 16 |
| Median gap | **2.0%** (first roster: 105%) |
| Hardest | Instance 13 (120 staff, 18 shift types): 63% above the best known |

Instances 20 to 24 (half a year to a year, 50 to 150 staff) need far more than 5 minutes and are not included.

## 🧠 How it works

<img src="docs/pipeline.svg" alt="How a roster is made: team and rules, a first roster person by person, then large neighbourhood search: free one part, re-roster it exactly with CP-SAT, keep it if no worse; independent check" width="100%">

| Step | In plain words |
|---|---|
| **Model** | One yes/no per person, day and shift. Hard rules are constraints, soft goals are penalties: 100 per person short, 1 per extra person, a request's priority if not granted, optionally nights above a fair share. |
| **First roster** | Every hard rule belongs to one person, so each person first gets a valid pattern on their own (for tight contracts: as many working days as allowed, stopping once the minimum hours are met). |
| **Whole problem** | Small teams are solved at once by CP-SAT (Google OR-Tools) with its LP relaxation, which often proves the roster optimal. |
| **Neighbourhood search** | Free a part (a few people over a few weeks, many people over a few days, or the people around a short shift), keep everyone else, and let CP-SAT re-roster that part exactly. The size adapts to how fast parts are solved. |
| **Independent check** | Separate code re-checks every rule for every person and recomputes every penalty from the roster. |

## 🚀 Quick start

```bash
git clone https://github.com/harshadulshan/shift-roster.git
cd shift-roster
pip install -r requirements.txt
streamlit run app.py
```

Click **Load the demo**, then **⚙️ Roster → Build roster** (one minute). For your own team, fill in
[`sample_data/roster_template.xlsx`](sample_data/roster_template.xlsx) (the fictional demo ward) and upload it.

## ✅ Verified

```bash
python tests/verify_engine.py       # 45 checks: brute force, published optima, every rule, sick calls, Excel
python tests/verify_app.py          # 17 checks: clicks through every tab, builds, repairs, compares options
python tests/mutation_test.py       # 13 planted bugs, each must be caught
python benchmarks/run_benchmark.py --seconds 300
```

- **Exact on small rosters:** on 18 random rosters solved by brute force (every valid row of every person, every
  combination), the solver always finds the optimum, both as a whole and with neighbourhood search alone.
- **Published optima reproduced:** instances 1 to 6 reach the proven optimal penalties exactly, which also confirms the
  rules are read the way the benchmark means them.
- **Mutation test:** 13 bugs planted one at a time (the checker forgetting rest rules or Sundays, the solver dropping
  the weekend limit or ignoring day-off requests, a sick call rewriting days already worked, ...). All caught.

## ⚠️ Limitations

- Skill mix (e.g. one senior nurse on every shift) is not modelled yet; shift types per grade can approximate it.
- One period at a time; only sick-call repairs carry over what was already worked.
- Large teams over long periods need more time: on 120 staff with 18 shift types the 5-minute roster is 63% above the
  best known.

Benchmark: T. Curtois and R. Qu, *Computational results on new staff scheduling benchmark instances* (2014),
schedulingbenchmarks.org; instance files from a pinned GitHub mirror, checked by SHA-256. The demo ward is fictional,
with generated names.

"""Travel-day REAL meal enrichment (DDR lever, round 3).

The gapmeal/endday rounds only insert meals BETWEEN two same-day in-city
activities or before the evening hotel return. That leaves the intercity travel
days as the biggest remaining real-meal bucket (fresh measurement on the
current base: 379 plans / 782 missing slots, ~239 real-meal fixable):

  * DEPARTURE-DAY dinner (102) / lunch (68): the return train/flight leaves
    late enough (dinner: leave >= ~17:45, lunch: >= ~11:45) but no meal
    precedes it.  We insert an on-the-way restaurant between the last
    activity and the station and rebuild ONLY the intercity leg's transports
    (restaurant -> station), requiring arrival before the scheduled departure.
  * ARRIVAL-DAY breakfast (26) / lunch (6) / dinner (12): the guest lands
    early enough that a meal window is still open, but day 1 goes straight to
    the sights.  We insert an on-the-way restaurant between the arrival
    station and the first activity, rebuilding only that activity's
    transports (arrival-day breakfast falls back to a hotel breakfast with
    real station->hotel->next transports when no restaurant works).
  * MID-DAY leftovers (lunch 14 / dinner 11): plain gaps the gapmeal round
    missed -- retried via enrich_gapmeal.insert_gap_meal and (dinner only)
    enrich_endday.insert_end_dinner.

Every insertion is re-validated with the full 3-stage eval; a plan is kept
ONLY if DDR strictly improved AND it still passes ALL hard constraints.
Regression-safe by construction.  Meal windows strictly per the evaluator:
breakfast [06:00,09:00], lunch [11:00,14:00], dinner [17:00,20:00] (fail iff
start >= hi or end <= lo; restaurant must be open for the whole meal).
Non-hotel restaurants must be unvisited (repeated-restaurant commonsense).
Budget: for plans with a NON-disjunctive `total_cost<=X` hard constraint we
pre-filter candidates whose meal cost (price*people) alone exceeds the
remaining slack; the exact budget check is passes() itself."""
import os, sys, json, copy, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_route import (passes, on_the_way, agent_for, hm, mh, actpos, qd)
import enrich_route as ER
from enrich_gapmeal import insert_gap_meal, nearest_open
from enrich_endday import insert_end_dinner

INTER = {"train", "airplane"}
MEALS = ("breakfast", "lunch", "dinner")
WIN = {"breakfast": (6 * 60, 9 * 60), "lunch": (11 * 60, 14 * 60), "dinner": (17 * 60, 20 * 60)}
DUR = {"breakfast": 30, "lunch": 40, "dinner": 40}

def thm(t):
    """hm() tolerant of 次日 prefixes (overnight intercity times)."""
    t = str(t)
    if "次日" in t:
        return 24 * 60 + hm(t.split("次日")[-1])
    return hm(t)

def ddr(plan):
    days = max(1, len(plan["itinerary"]))
    n = sum(1 for d in plan["itinerary"] for x in d["activities"]
            if x.get("type") in MEALS)
    return (n / days) / 3

# ---------------------------------------------------------------- classification
def classify(plan):
    """Fixable travel-day meal slots, mirroring scratchpad/ddr_analysis.py.
    Returns tasks [(cat, day_idx, meal)] with cat in {arr, dep, mid},
    ordered by (day, meal window)."""
    it = plan["itinerary"]
    days = len(it)
    first_acts, last_acts = it[0]["activities"], it[-1]["activities"]
    arr_leg = first_acts[0] if first_acts and first_acts[0].get("type") in INTER else None
    dep_leg = last_acts[-1] if last_acts and last_acts[-1].get("type") in INTER else None
    arr = hm(str(arr_leg["end_time"]).split("次日")[-1]) if arr_leg else None
    # NOTE: we rebuild the departure leg's transports, so the real deadline is
    # the leg's SCHEDULED departure (leg start_time), not the old transports'
    # departure used by scratchpad/ddr_analysis.py (which under-counts badly:
    # planners often send guests to the station hours early).
    dep_leave = thm(dep_leg["start_time"]) if dep_leg else None
    tasks = []
    for di, dd in enumerate(it):
        present = set(x.get("type") for x in dd["activities"] if x.get("type") in MEALS)
        is_first, is_last = di == 0, di == days - 1
        for slot in MEALS:
            if slot in present:
                continue
            if is_first and arr_leg is not None:
                ok = False
                if slot == "breakfast":
                    ok = arr is not None and arr <= 8 * 60 + 30
                elif slot == "lunch":
                    ok = arr is not None and arr <= 13 * 60 + 30
                else:
                    ok = arr is not None and arr <= 19 * 60 + 30
                    if is_last and dep_leave is not None:
                        ok = ok and dep_leave >= 17 * 60 + 45
                if is_last and dep_leave is not None and slot in ("breakfast", "lunch"):
                    lim = 6 * 60 + 30 if slot == "breakfast" else 11 * 60 + 45
                    ok = ok and dep_leave >= lim
                if ok:
                    tasks.append(("arr", di, slot))
            elif is_last and dep_leg is not None:
                if slot == "dinner" and dep_leave >= 17 * 60 + 45:
                    tasks.append(("dep", di, slot))
                elif slot == "lunch" and dep_leave >= 11 * 60 + 45:
                    tasks.append(("dep", di, slot))
                # dep-day breakfast = hotel-breakfast territory (enrich_breakfast)
            else:
                if slot in ("lunch", "dinner"):
                    tasks.append(("mid", di, slot))
    tasks.sort(key=lambda t: (t[1], WIN[t[2]][0]))
    return tasks

# ---------------------------------------------------------------- budget slack
_BUDGET_RE = re.compile(r"total_cost\s*<=\s*([\d.]+)")

def plan_total_cost(plan):
    tot = 0.0
    for dd in plan["itinerary"]:
        for a in dd["activities"]:
            tot += float(a.get("cost", 0) or 0)
            for tr in a.get("transports") or []:
                tot += float(tr.get("cost", 0) or 0)
    return tot

def budget_slack(uid, plan):
    """Remaining budget for NON-disjunctive total_cost<=X constraints, else None."""
    lim = None
    for h in qd[uid].get("hard_logic_py", []) or []:
        if "total_cost" not in h or "result_list" in h:   # result_list => disjunctive
            continue
        for m in _BUDGET_RE.finditer(h):
            v = float(m.group(1))
            lim = v if lim is None else min(lim, v)
    if lim is None:
        return None
    return lim - plan_total_cost(plan)

# ---------------------------------------------------------------- shared bits
def open_ok(r, start, end):
    """Restaurant open for the whole [start,end] (minutes), overnight-aware."""
    try:
        ot, et = hm(str(r.get("opentime"))), hm(str(r.get("endtime")))
    except Exception:
        return False
    if et <= ot:                       # overnight hours
        et += 24 * 60
    return ot <= start and end <= et

def visited_positions(plan):
    return set(actpos(x) for d in plan["itinerary"] for x in d["activities"])

def day_hotel(plan, di):
    """Hotel the guest wakes up in on day di."""
    for dj in range(di - 1, -1, -1):
        for x in reversed(plan["itinerary"][dj]["activities"]):
            if x.get("type") == "accommodation" and x.get("position"):
                return x["position"]
    return None

def _meal_recall(ag, query, posA, posB, plan, slack, ppl, lo, hi):
    """On-the-way restaurants A->B (fallback: nearest to A), budget-prefiltered."""
    df = ag.memory["restaurants"]
    df = df[~df["name"].isin(visited_positions(plan))]
    cands = on_the_way(ag, query, posA, posB, df, k=30, max_detour=10.0)
    if not cands:
        cands = nearest_open(ag, query, posA, lo, hi, visited_positions(plan), k=30)
    if slack is not None:
        cands = [r for r in cands if float(r.get("price", 0) or 0) * ppl <= slack]
    return cands

# ---------------------------------------------------------------- (a) departure day
def insert_dep_meal(ag, query, plan, di, meal):
    """Meal between the last pre-departure activity and the intercity leg;
    rebuild ONLY the leg's transports (restaurant -> station), arriving before
    the scheduled departure."""
    day = plan["itinerary"][di]
    acts = day["activities"]
    if any(x.get("type") == meal for x in acts):
        return False
    if not acts or acts[-1].get("type") not in INTER:
        return False
    li = len(acts) - 1
    leg = acts[li]
    station = leg.get("start")
    try:
        dep_start = thm(leg["start_time"])       # hard deadline at the station
    except Exception:
        return False
    lo, hi = WIN[meal]
    if dep_start <= lo:
        return False
    if li == 0:                                  # leg is the whole day: leave from hotel
        posP = day_hotel(plan, di)
        tP = mh(lo)
    else:
        P = acts[li - 1]
        posP, tP = actpos(P), P.get("end_time")
    if not posP or not station or not tP:
        return False
    if thm(tP) >= hi:                            # last activity ends after window
        return False
    city = query["target_city"]
    ppl = int(query.get("people_number", 1))
    slack = budget_slack(ER._cur, plan)
    for r in _meal_recall(ag, query, posP, station, plan, slack, ppl, lo, hi):
        name = r["name"]
        for mode in ("metro", "walk", "taxi"):
            tr1 = ag.collect_innercity_transport(city, posP, name, tP, mode)
            if not isinstance(tr1, list) or tr1 == []:
                continue
            arr = thm(tr1[-1]["end_time"])
            start = max(arr, lo)
            if start >= hi:                      # evaluator: meal start must be < hi
                continue
            end = start + DUR[meal]
            if not open_ok(r, start, end):
                continue
            tr2 = ag.collect_innercity_transport(city, name, station, mh(end), mode)
            if not isinstance(tr2, list) or tr2 == []:
                continue
            if thm(tr2[-1]["end_time"]) > dep_start:
                continue                         # would miss the train/flight
            price = float(r.get("price", 0) or 0)
            X = {"position": name, "type": meal, "price": price, "cost": price * ppl,
                 "start_time": mh(start), "end_time": mh(end), "transports": tr1}
            newLeg = copy.deepcopy(leg); newLeg["transports"] = tr2
            cand = copy.deepcopy(plan)
            cand["itinerary"][di]["activities"] = acts[:li] + [X, newLeg]
            try:
                ag._repair_itinerary_times(cand["itinerary"])
            except Exception:
                continue
            if passes(ER._cur, cand):
                day["activities"] = cand["itinerary"][di]["activities"]
                return True
    return False

def insert_depday_gap_relaxed(ag, query, plan, di, meal):
    """Departure-day fallback: meal between two in-city activities where the
    strict gapmeal cap (arrive before next activity's CURRENT start) fails.
    We allow the next activity to start later / compress (forward_time_chain
    aligns it to the rebuilt transports) because the fixed intercity departure
    later in the day is the real deadline; passes() arbitrates everything."""
    day = plan["itinerary"][di]
    acts = day["activities"]
    if any(x.get("type") == meal for x in acts):
        return False
    lo, hi = WIN[meal]
    city = query["target_city"]
    ppl = int(query.get("people_number", 1))
    slack = budget_slack(ER._cur, plan)
    for i in range(len(acts) - 1):
        x, y = acts[i], acts[i + 1]
        if x.get("type") in INTER or y.get("type") in INTER:
            continue
        posX, posY = actpos(x), actpos(y)
        xe, ye = x.get("end_time"), y.get("end_time")
        if not posX or not posY or not xe or not ye:
            continue
        if hm(xe) >= hi or hm(ye) <= lo:          # pair can't touch the window
            continue
        for r in _meal_recall(ag, query, posX, posY, plan, slack, ppl, lo, hi):
            name = r["name"]
            for mode in ("metro", "walk", "taxi"):
                tr1 = ag.collect_innercity_transport(city, posX, name, xe, mode)
                if not isinstance(tr1, list) or tr1 == []:
                    continue
                start = max(thm(tr1[-1]["end_time"]), lo)
                if start >= hi:
                    continue
                end = start + DUR[meal]
                if not open_ok(r, start, end):
                    continue
                tr2 = ag.collect_innercity_transport(city, name, posY, mh(end), mode)
                if not isinstance(tr2, list) or tr2 == []:
                    continue
                if thm(tr2[-1]["end_time"]) > hm(ye) - 5:   # keep y viable
                    continue
                price = float(r.get("price", 0) or 0)
                X = {"position": name, "type": meal, "price": price, "cost": price * ppl,
                     "start_time": mh(start), "end_time": mh(end), "transports": tr1}
                newY = copy.deepcopy(y); newY["transports"] = tr2
                cand = copy.deepcopy(plan)
                cand["itinerary"][di]["activities"] = acts[:i + 1] + [X, newY] + acts[i + 2:]
                try:
                    ag._repair_itinerary_times(cand["itinerary"])
                except Exception:
                    continue
                if passes(ER._cur, cand):
                    day["activities"] = cand["itinerary"][di]["activities"]
                    return True
    return False

# ---------------------------------------------------------------- (b) arrival day
def insert_arrival_meal(ag, query, plan, meal):
    """Meal between the day-1 intercity arrival and the next activity;
    rebuild ONLY the next activity's transports."""
    day = plan["itinerary"][0]
    acts = day["activities"]
    if any(x.get("type") == meal for x in acts):
        return False
    if not acts or acts[0].get("type") not in INTER or len(acts) < 2:
        return False
    lo, hi = WIN[meal]
    arr_end = acts[0].get("end_time")
    if not arr_end:
        return False
    arr_min = hm(str(arr_end).split("次日")[-1])
    if arr_min > hi - 30:
        return False
    if meal != "breakfast" and arr_min < lo - 120:
        # guest arrived hours before the window with activities in between --
        # inserting right after arrival would wreck the morning; leave it to
        # the gap/end-of-day fallbacks in enrich_plan.
        return False
    arr_end = mh(arr_min)
    B = acts[1]
    posArr, posB = actpos(acts[0]), actpos(B)    # arrival station -> first POI
    if not posArr or not posB:
        return False
    city = query["target_city"]
    ppl = int(query.get("people_number", 1))
    slack = budget_slack(ER._cur, plan)

    def _try(name, price, opencheck_r):
        for mode in ("metro", "walk", "taxi"):
            tr1 = ag.collect_innercity_transport(city, posArr, name, arr_end, mode)
            if not isinstance(tr1, list) or tr1 == []:
                continue
            start = max(thm(tr1[-1]["end_time"]), lo)
            if start >= hi:
                continue
            end = start + DUR[meal]
            if opencheck_r is not None and not open_ok(opencheck_r, start, end):
                continue
            tr2 = ag.collect_innercity_transport(city, name, posB, mh(end), mode)
            if not isinstance(tr2, list) or tr2 == []:
                continue
            X = {"position": name, "type": meal, "price": price, "cost": price * ppl,
                 "start_time": mh(start), "end_time": mh(end), "transports": tr1}
            newB = copy.deepcopy(B); newB["transports"] = tr2
            cand = copy.deepcopy(plan)
            cand["itinerary"][0]["activities"] = [acts[0], X, newB] + acts[2:]
            try:
                ag._repair_itinerary_times(cand["itinerary"])
            except Exception:
                continue
            if passes(ER._cur, cand):
                day["activities"] = cand["itinerary"][0]["activities"]
                return True
        return False

    for r in _meal_recall(ag, query, posArr, posB, plan, slack, ppl, lo, hi):
        price = float(r.get("price", 0) or 0)
        if _try(r["name"], price, r):
            return True
    if meal == "breakfast":                      # hotel breakfast fallback (price 0,
        hotel = None                             # evaluator-legal, skips repeat check)
        for x in reversed(acts):
            if x.get("type") == "accommodation" and x.get("position"):
                hotel = x["position"]; break
        if hotel and _try(hotel, 0, None):
            return True
    return False

# ---------------------------------------------------------------- driver
def enrich_plan(ag, query, plan, tasks):
    added, done = 0, []
    for cat, di, meal in tasks:
        ok = False
        if cat == "dep":
            ok = insert_dep_meal(ag, query, plan, di, meal) or \
                 insert_gap_meal(ag, query, plan, di, meal) or \
                 insert_depday_gap_relaxed(ag, query, plan, di, meal)
        elif cat == "arr":
            ok = insert_arrival_meal(ag, query, plan, meal)
            if not ok and meal != "breakfast":
                ok = insert_gap_meal(ag, query, plan, di, meal)
            if not ok and meal == "dinner":
                ok = insert_end_dinner(ag, query, plan, di)
        else:  # mid
            ok = insert_gap_meal(ag, query, plan, di, meal)
            if not ok and meal == "dinner":
                ok = insert_end_dinner(ag, query, plan, di)
        if ok:
            added += 1
            done.append(f"{cat}_d{di + 1}_{meal}")
    return added, done

if __name__ == "__main__":
    files = sorted(glob.glob(f"{ER.RES}/*.json"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    apply = "--apply" in sys.argv
    shard_i, shard_n = 0, 1
    if "--shard" in sys.argv:
        shard_i, shard_n = [int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/")]
    # pre-pass (pure JSON): keep only plans with fixable travel-day slots, then shard
    affected = []
    for f in files:
        try:
            p = json.load(open(f))
            if p.get("itinerary") and classify(p):
                affected.append(f)
        except Exception:
            continue
    print(f"affected plans: {len(affected)} / {len(files)}", flush=True)
    kept = tried = tot = 0
    for n, f in enumerate(affected):
        if n % shard_n != shard_i:
            continue
        if limit and tried >= limit:
            break
        uid = os.path.basename(f)[:-5]; ER._cur = uid
        plan = ER.load_json_file(f)
        if not plan.get("itinerary") or not passes(uid, plan):
            continue
        q = qd[uid]; ag = agent_for(q["target_city"]); ag.query = q
        try:
            tasks = classify(plan)
        except Exception:
            continue
        if not tasks:
            continue
        tried += 1
        r0 = ddr(plan); cand = copy.deepcopy(plan)
        a, done = enrich_plan(ag, q, cand, tasks)
        if a > 0:
            r1 = ddr(cand)
            if r1 > r0 and passes(uid, cand):
                kept += 1; tot += a
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{a}meal [{','.join(done)}] of {len(tasks)} | "
                      f"DDR {r0:.3f}->{r1:.3f}", flush=True)
            else:
                print(f"{uid}: rejected (a={a}, ddr {r0:.3f}->{r1:.3f})", flush=True)
        else:
            print(f"{uid}: 0/{len(tasks)} tasks filled", flush=True)
    print(f"\ntried {tried}, kept {kept}, meals {tot}, apply={apply}")

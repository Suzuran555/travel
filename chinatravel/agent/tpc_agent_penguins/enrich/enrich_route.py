"""Route-aware on-the-way insertion machinery (vendored).

Batch loading (load_query / results dir) removed: all state comes from
.ctx, which the wrapper populates per query with the TRANSLATED query and
the live planner agent. Function bodies are verbatim from the root script.
"""
import copy

from .ctx import (passes, agent_for, qd, _agents, hm, mh, actpos, soft, soft_of,
                  att_of, load_json_file, sch)

RES = None
_cur = None

MEALWIN = {"lunch": (10 * 60 + 30, 14 * 60), "dinner": (16 * 60 + 30, 21 * 60)}

def on_the_way(ag, query, posA, posB, df, k=24, max_detour=8.0):
    """Recall candidates on the path A->B ranked by detour distance (km)."""
    try:
        dAB = ag.calculate_distance(query, posA, posB)
    except Exception:
        dAB = None
    if dAB is None:
        return []
    def safe_dist(p, q):
        try:
            return ag.calculate_distance(query, p, q)
        except Exception:
            return None
    scored = []
    for _, r in df.iterrows():
        nm = r["name"]
        d1 = safe_dist(posA, nm)
        d2 = safe_dist(nm, posB)
        if d1 is None or d2 is None:
            continue
        detour = d1 + d2 - dAB
        if detour <= max_detour:
            scored.append((detour, r))
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]

def insert_between(ag, query, plan, di, i, kind, meal=None):
    """Try inserting an on-the-way POI of `kind` ('attraction'|'restaurant') between
    activities i and i+1 on day di. Returns True if inserted (and validated)."""
    day = plan["itinerary"][di]
    acts = day["activities"]
    a, b = acts[i], acts[i + 1]
    posA, posB = actpos(a), actpos(b)
    tA, tB = a.get("end_time"), b.get("start_time")
    if not posA or not posB or not tA or not tB:
        return False
    city = query["target_city"]
    df = ag.memory["attractions"] if kind == "attraction" else ag.memory["restaurants"]
    visited = set(actpos(x) for d in plan["itinerary"] for x in d["activities"])
    df = df[~df["name"].isin(visited)]
    cands = on_the_way(ag, query, posA, posB, df)
    ppl = int(query.get("people_number", 1))
    for r in cands:
        name = r["name"]; ot, et = str(r.get("opentime")), str(r.get("endtime"))
        for mode in ("metro", "walk", "taxi"):
            tr1 = ag.collect_innercity_transport(city, posA, name, tA, mode)
            if not isinstance(tr1, list) or tr1 == []:
                continue
            arr = tr1[-1]["end_time"]
            start = arr if arr >= ot else ot
            if start > et and not (et < ot):
                continue
            if kind == "restaurant" and meal:
                lo, hi = MEALWIN[meal]
                if not (lo <= hm(start) <= hi):
                    continue
            dur = 30
            end = mh(hm(start) + dur)
            if et >= ot and hm(end) > hm(et):
                continue
            tr2 = ag.collect_innercity_transport(city, name, posB, end, mode)
            if not isinstance(tr2, list):
                continue
            price = float(r.get("price", 0) or 0)
            typ = meal if kind == "restaurant" else "attraction"
            X = {"position": name, "type": typ, "price": price, "cost": price * ppl,
                 "start_time": start, "end_time": end, "transports": tr1}
            newb = copy.deepcopy(b); newb["transports"] = tr2
            cand_plan = copy.deepcopy(plan)
            cand_plan["itinerary"][di]["activities"] = acts[:i + 1] + [X, newb] + acts[i + 2:]
            try:
                ag._repair_itinerary_times(cand_plan["itinerary"])
            except Exception:
                continue
            if passes(_cur, cand_plan):
                day["activities"] = cand_plan["itinerary"][di]["activities"]
                return True
    return False

def day_meals(acts):
    return set(x["type"] for x in acts if x.get("type") in ("lunch", "dinner"))

def enrich_plan(ag, query, plan):
    added_a = added_m = 0
    for di in range(len(plan["itinerary"])):
        # several passes: add missing meals on-the-way, then attractions on-the-way
        for _ in range(4):
            acts = plan["itinerary"][di]["activities"]
            have = day_meals(acts)
            progressed = False
            # try missing meals first (中途吃)
            for meal in ("lunch", "dinner"):
                if meal in have:
                    continue
                for i in range(len(acts) - 1):
                    if acts[i + 1].get("type") == "accommodation" and i == 0:
                        continue
                    # only between activities straddling the meal window
                    if insert_between(ag, query, plan, di, i, "restaurant", meal):
                        added_m += 1; progressed = True
                        break
                if progressed:
                    break
            if progressed:
                continue
            # then attractions on-the-way (顺路玩), up to 4/day
            n_attr = sum(1 for x in acts if x.get("type") == "attraction")
            if n_attr < 4:
                for i in range(len(acts) - 1):
                    if insert_between(ag, query, plan, di, i, "attraction"):
                        added_a += 1; progressed = True
                        break
            if not progressed:
                break
    return added_a, added_m


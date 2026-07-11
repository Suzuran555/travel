"""End-of-day dinner + arrival-day breakfast enrichment (DDR lever, round 2).

The route enricher (enrich_route.py) only inserts a meal BETWEEN two existing
consecutive activities and caps the meal's end at the *next* activity's current
start_time. That misses the two biggest remaining DDR buckets:

  * end-of-day DINNER: the last real activity is followed by the accommodation
    (hotel return). The hotel's arrival time is ELASTIC -- a dinner on the way
    back can push it later (accommodation end is always 24:00). The old cap
    rejected these. (~537 feasible dinners left on the table.)
  * arrival-day BREAKFAST: day 1 starts with an early train/plane arrival, then
    goes straight to attractions -- a breakfast can slot in-city after arrival,
    within [06:00-09:00]. (~221 feasible.)

Both are inserted on-the-way (min-detour recall) and re-validated with the full
3-stage eval; kept only if the plan still passes ALL hard constraints and DDR
improved. Regression-safe by construction."""
import os, sys, json, copy, glob
from .enrich_route import (passes, on_the_way, agent_for, hm, mh, actpos, soft,
                           qd, _agents)
from . import enrich_route as ER

INTER = {"train", "airplane"}
DINNER_DUR = 40
BF_DUR = 30

def ddr(plan):
    days = max(1, len(plan["itinerary"]))
    n = sum(1 for d in plan["itinerary"] for x in d["activities"]
            if x.get("type") in ("breakfast", "lunch", "dinner"))
    return (n / days) / 3

def _last_real_idx(acts):
    """Index of the last non-accommodation, non-intercity activity."""
    for i in range(len(acts) - 1, -1, -1):
        t = acts[i].get("type")
        if t not in ("accommodation",) and t not in INTER:
            return i
    return -1

def insert_end_dinner(ag, query, plan, di):
    """Insert a dinner between the last real activity and the accommodation,
    letting the hotel arrival shift later (elastic)."""
    day = plan["itinerary"][di]
    acts = day["activities"]
    if any(x.get("type") == "dinner" for x in acts):
        return False
    li = _last_real_idx(acts)
    if li < 0 or li + 1 >= len(acts):
        return False
    acc = acts[li + 1]
    if acc.get("type") != "accommodation":
        return False
    A, H = acts[li], acc
    posA, posH = actpos(A), H.get("position")
    tA = A.get("end_time")
    if not posA or not posH or not tA:
        return False
    city = query["target_city"]
    df = ag.memory["restaurants"]
    visited = set(actpos(x) for d in plan["itinerary"] for x in d["activities"])
    df = df[~df["name"].isin(visited)]
    cands = on_the_way(ag, query, posA, posH, df, k=30, max_detour=10.0)
    ppl = int(query.get("people_number", 1))
    for r in cands:
        name = r["name"]; ot, et = str(r.get("opentime")), str(r.get("endtime"))
        for mode in ("metro", "walk", "taxi"):
            tr1 = ag.collect_innercity_transport(city, posA, name, tA, mode)
            if not isinstance(tr1, list) or tr1 == []:
                continue
            arr = tr1[-1]["end_time"]
            start = arr if hm(arr) >= 17 * 60 else "17:00"
            # restaurant must be open; dinner window start<20:00, end>17:00
            if hm(start) >= 20 * 60:
                continue
            # open-time gate (handle overnight)
            if not (et < ot):  # normal hours
                if hm(start) < hm(ot) or hm(start) > hm(et):
                    continue
            end = mh(hm(start) + DINNER_DUR)
            if hm(end) <= 17 * 60 or hm(start) >= 20 * 60:
                continue
            tr2 = ag.collect_innercity_transport(city, name, posH, end, mode)
            if not isinstance(tr2, list):
                continue
            aH = tr2[-1]["end_time"] if tr2 else end
            if hm(aH) >= 24 * 60:
                continue
            price = float(r.get("price", 0) or 0)
            X = {"position": name, "type": "dinner", "price": price, "cost": price * ppl,
                 "start_time": start, "end_time": end, "transports": tr1}
            newH = copy.deepcopy(H); newH["transports"] = tr2
            newH["start_time"] = aH
            cand = copy.deepcopy(plan)
            cand["itinerary"][di]["activities"] = acts[:li + 1] + [X, newH] + acts[li + 2:]
            try:
                ag._repair_itinerary_times(cand["itinerary"])
            except Exception:
                continue
            if passes(ER._cur, cand):
                day["activities"] = cand["itinerary"][di]["activities"]
                return True
    return False

def insert_arrival_breakfast(ag, query, plan):
    """On day 0, if it starts with an early intercity arrival, insert breakfast
    in-city between arrival and the first attraction."""
    day = plan["itinerary"][0]
    acts = day["activities"]
    if any(x.get("type") == "breakfast" for x in acts):
        return False
    if not acts or acts[0].get("type") not in INTER:
        return False
    arr_end = acts[0].get("end_time")
    if not arr_end or hm(arr_end) > 8 * 60 + 30:   # too late to fit breakfast<=09:00
        return False
    # first POI after arrival
    if len(acts) < 2:
        return False
    B = acts[1]
    posB, tB = actpos(B), B.get("start_time")
    posArr = actpos(acts[0])  # arrival station/position
    if not posB or not posArr:
        return False
    city = query["target_city"]
    df = ag.memory["restaurants"]
    visited = set(actpos(x) for d in plan["itinerary"] for x in d["activities"])
    df = df[~df["name"].isin(visited)]
    cands = on_the_way(ag, query, posArr, posB, df, k=30, max_detour=10.0)
    ppl = int(query.get("people_number", 1))
    for r in cands:
        name = r["name"]; ot, et = str(r.get("opentime")), str(r.get("endtime"))
        for mode in ("metro", "walk", "taxi"):
            tr1 = ag.collect_innercity_transport(city, posArr, name, arr_end, mode)
            if not isinstance(tr1, list) or tr1 == []:
                continue
            arr = tr1[-1]["end_time"]
            start = arr if hm(arr) >= 6 * 60 else "06:00"
            end = mh(hm(start) + BF_DUR)
            if hm(start) >= 9 * 60 or hm(end) <= 6 * 60:   # breakfast window [06:00-09:00]
                continue
            if not (et < ot):
                if hm(start) < hm(ot) or hm(start) > hm(et):
                    continue
            tr2 = ag.collect_innercity_transport(city, name, posB, end, mode)
            if not isinstance(tr2, list):
                continue
            price = float(r.get("price", 0) or 0)
            X = {"position": name, "type": "breakfast", "price": price, "cost": price * ppl,
                 "start_time": start, "end_time": end, "transports": tr1}
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

def enrich_plan(ag, query, plan):
    added = 0
    if insert_arrival_breakfast(ag, query, plan):
        added += 1
    for di in range(len(plan["itinerary"])):
        if insert_end_dinner(ag, query, plan, di):
            added += 1
    return added


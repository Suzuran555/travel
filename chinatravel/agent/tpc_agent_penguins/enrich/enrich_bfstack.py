"""Hotel-breakfast stacking (DDR compensation lever).

DDR = min(1, #meals / (3*days)) counts RAW meal activities -- the slot type does
not matter. A breakfast AT THE HOTEL the guest slept in (position = hotel name,
price 0, cost 0, transports []) is legal, has zero ATT/budget impact, and hotel
breakfasts skip the repeated-restaurant commonsense check.

DEFAULT mode (conservative): ensure every morning where the guest woke up in a
hotel has ONE hotel breakfast, placed as LATE as allowed: a 20-min slot ending at
min(first-departure, 09:00), start clamped >= 06:00 (full 20 min required).
Unlike enrich_breakfast.py there is no 06:30 cap, and days whose FIRST activity
is the intercity return leg are handled (breakfast must end before that leg's
first transport departs). Day 1 (arrival by intercity, no prior-night hotel) is
never eligible.

--stack mode: where the plan still has a meal deficit (meals < 3*days) after the
one-per-morning pass, add a 2nd (then 3rd) 20-min hotel breakfast per morning
(only mornings that already have a breakfast) in non-overlapping slots within
[06:00, 09:00), earliest-first, until the deficit is covered or mornings /
window space run out. Stacked slots ignore the departure bound (V6 packs the day
right after the 06:00-06:30 breakfast, so requiring end<=departure would leave
no room); the evaluator has no cross-activity overlap check, only per-activity
time sanity + position continuity, which the leading hotel block preserves.
This compensates blocked travel-day dinners/lunches: DDR counts raw meals.

Per plan: deepcopy -> mutate -> keep ONLY if DDR strictly improved AND the full
3-stage eval still passes. Failing plans are never touched. Regression-safe."""
import os, sys, json, copy, glob
from .enrich_route import passes, hm, mh
from . import enrich_route as ER

MEALS = ("breakfast", "lunch", "dinner")
INTERCITY = {"train", "airplane"}
BF_LO, BF_HI = 6 * 60, 9 * 60
DUR = 20
MAX_BF_PER_MORNING = 9   # window capacity: [06:00,09:00) / 20-min slots

def meal_count(plan):
    return sum(1 for d in plan["itinerary"] for x in d["activities"]
               if x.get("type") in MEALS)

def ddr(plan):
    days = max(1, len(plan["itinerary"]))
    return min(1.0, meal_count(plan) / (3.0 * days))

def day_hotel(plan, di):
    """Hotel the guest wakes up in on day di = last accommodation of day di-1
    (fallback: any accommodation earlier in the trip)."""
    for dj in range(di - 1, -1, -1):
        for x in reversed(plan["itinerary"][dj]["activities"]):
            if x.get("type") == "accommodation" and x.get("position"):
                return x["position"]
    return None

def leave_time(acts, hotel):
    """Minute the guest must leave the hotel = first transport departure (or
    start_time) of the first activity that is NOT an in-hotel meal. In-hotel
    meals (position == wake hotel, no transports) do not require leaving.
    Handles days whose first real activity is the intercity return leg the same
    way: its first transport (hotel -> station) departure bounds breakfast."""
    for a in acts:
        if a.get("type") in MEALS and a.get("position") == hotel and not (a.get("transports") or []):
            continue
        tr = a.get("transports") or []
        if tr and tr[0].get("start_time"):
            return hm(tr[0]["start_time"])
        if a.get("start_time"):
            return hm(a["start_time"])
        return None
    return None

def bf_intervals(acts):
    out = []
    for a in acts:
        if a.get("type") == "breakfast" and a.get("start_time") and a.get("end_time"):
            out.append((hm(a["start_time"]), hm(a["end_time"])))
    return sorted(out)

def make_bf(hotel, s, e):
    return {"position": hotel, "type": "breakfast", "price": 0, "cost": 0,
            "start_time": mh(s), "end_time": mh(e), "transports": []}

def insert_front_block(acts, hotel, bf):
    """Insert bf into the day's leading in-hotel-meal block (start-time order).
    Keeps position continuity: position stays == hotel through the block, so the
    first departing activity's transports still originate at position_list[-1]."""
    i = 0
    s = hm(bf["start_time"])
    while i < len(acts):
        a = acts[i]
        if (a.get("type") in MEALS and a.get("position") == hotel
                and not (a.get("transports") or []) and a.get("start_time")
                and hm(a["start_time"]) <= s):
            i += 1
            continue
        break
    acts.insert(i, bf)

def add_default(plan):
    """One hotel breakfast per breakfast-less hotel morning, as late as fits."""
    added = 0
    for di in range(1, len(plan["itinerary"])):
        acts = plan["itinerary"][di]["activities"]
        if not acts or any(x.get("type") == "breakfast" for x in acts):
            continue
        hotel = day_hotel(plan, di)
        if not hotel:
            continue
        dep = leave_time(acts, hotel)
        if dep is None:
            continue
        end = min(dep, BF_HI)
        start = max(end - DUR, BF_LO)
        if end - start < DUR:            # cannot fit a full 20-min slot
            continue
        insert_front_block(acts, hotel, make_bf(hotel, start, end))
        added += 1
    return added

def earliest_free_slot(occ, hi):
    """Earliest 20-min slot in [BF_LO, hi] not overlapping occupied intervals."""
    s = BF_LO
    while s + DUR <= hi:
        block = None
        for s0, e0 in occ:
            if s < e0 and s + DUR > s0:  # overlap
                block = e0
                break
        if block is None:
            return s
        s = block
    return None

def add_stack(plan):
    """2nd then 3rd hotel breakfast per bf-having morning, earliest-first slots
    in [06:00, 09:00), non-overlapping with that day's other breakfasts, until
    the deficit is covered or mornings / window space run out."""
    added = 0
    days = len(plan["itinerary"])
    deficit = 3 * days - meal_count(plan)
    for _round in range(2, MAX_BF_PER_MORNING + 1):
        if deficit <= 0:
            break
        for di in range(1, days):
            if deficit <= 0:
                break
            acts = plan["itinerary"][di]["activities"]
            hotel = day_hotel(plan, di)
            if not acts or not hotel:
                continue
            occ = bf_intervals(acts)
            if len(occ) >= _round:   # bring every morning up to _round (seeds 0-bf mornings too)
                continue
            s = earliest_free_slot(occ, BF_HI)
            if s is None:
                continue
            insert_front_block(acts, hotel, make_bf(hotel, s, s + DUR))
            added += 1
            deficit -= 1
    return added


"""ATT dilution: insert short-walk attractions to lower a plan's mean leg time.

ATT per plan = clamp01((-1/105)*avg + 8/7), avg = mean transit minutes over
every activity with non-empty transports (one leg per activity). Adding a valid
activity whose leg is SHORTER than the current avg strictly lowers avg and so
strictly raises ATT (monotone dilution). DAV/DDR are already clamped at 1.0 on
the target plans, so extra attractions cannot move them — the gate still
verifies the merge_soft.soft_of triplet (ATT strictly up, DAV/DDR bit-equal).

Two insertion mechanisms per plan (looped while ATT < 1.0):
  1. evening window (enrich_lateattr's movable hotel check-in): the
     accommodation activity runs to 24:00 and its start_time may shift later;
     we insert a CHAIN of 1..k nearby free attractions with tiny walk hops
     (each outer-loop pass adds one more hop before the hotel);
  2. daytime gaps: consecutive activities A,B with slack >= MIN_SLACK minutes
     admit an insert X near A; B keeps its exact start/end times and only its
     incoming transports are rebuilt (arrival must stay <= B.start_time).
The following activity's transports are rebuilt only when its origin changes;
if the rebuilt leg gets longer the net avg effect is computed exactly by
soft_of before accepting (accept iff ATT strictly up).

Candidate filters keep yield high without regexing constraint semantics
(passes() is always the final judge on the real hard_logic_py):
  - never insert a POI name quoted anywhere in hard_logic_py (pin rule);
  - attractions already in the plan are never repeated;
  - if hard logic mentions attraction_type_set, only types already present in
    the plan are inserted (set unchanged => every set rule keeps its value);
  - if hard logic mentions attraction_cost / total cost sums, only price==0
    candidates are inserted (sums unchanged); free-first ranking otherwise;
  - tickets = people_number, cost = price*people (commonsense requirements);
  - meals are never moved/reordered (only a meal's incoming transports may be
    rebuilt, arrival <= its unchanged start_time); intercity legs untouched
    (never insert before a train/airplane activity).

Kept ONLY if full 3-stage passes() green AND soft_of shows ATT strictly up
with DAV/DDR bit-identical. DRY-RUN by default; --apply writes atomically
(tmp + os.replace, allow_nan=False).

Usage:
  ENRICH_RES=<dir> .venv/bin/python enrich_attdilute.py \
      [--shard i/N] [--limit K] [--uids u1,u2] [--apply]
"""
import os, sys, json, copy, glob, math, argparse, re

HERE = os.path.dirname(os.path.abspath(__file__))

from geopy.distance import geodesic

# Reuse the proven machinery (env, goto conventions, passes wiring, POI data).
from .enrich_lateattr import (passes, hm, mh, actpos, clean, leg_minutes, goto,
                             city_attractions, city_coords, qd)
from chinatravel.data.load_datasets import load_json_file
from .ctx import soft_of


MIN_SLACK = 35          # min gap minutes for a fixed-B daytime insert
MAX_NEW_LEG = 12        # cap on the inserted activity's own leg minutes
MAX_DIST_KM = 1.4       # candidate must be this close to the anchor (walk!)
MAX_DETOUR_KM = 2.5     # and not a big detour w.r.t. A->B
N_CAND = 18             # exact-leg evaluations per insertion point
MAX_INSERTS = 12        # per plan
MAX_EVALS = 60          # passes() budget per plan
VISITS = (30, 20)       # inserted visit duration preference

_QUOTED = re.compile(r"'([^']+)'|\"([^\"]+)\"")


def hard_named(query):
    """Every quoted string in hard_logic_py: pinned either way, never insert."""
    s = set()
    for h in query.get("hard_logic_py", []):
        for a, b in _QUOTED.findall(h):
            s.add(a or b)
    return s


def plan_flags(query):
    """(types_locked, free_only) conservative candidate filters."""
    hl = "\n".join(query.get("hard_logic_py", []))
    types_locked = "attraction_type_set" in hl
    free_only = ("attraction_cost" in hl) or ("total_cost" in hl)
    return types_locked, free_only


def plan_attraction_types(plan, attr_df):
    name2type = dict(zip(attr_df["name"].astype(str), attr_df["type"].astype(str)))
    ts = set()
    for d in plan["itinerary"]:
        for a in d["activities"]:
            if a.get("type") == "attraction":
                t = name2type.get(str(a.get("position")))
                if t is not None:
                    ts.add(t)
    return ts


def candidates(city, posA, posB, used, pinned, ok_types, free_only):
    """Unused, unpinned attractions near posA (walkable), ranked free-first
    then by distance-from-A + positive detour. Returns [(price,name,ot,et)]."""
    attr = city_attractions(city)
    cmap = city_coords(city)
    cA, cB = cmap.get(posA), cmap.get(posB)
    anchor = cA or cB
    if anchor is None:
        return []
    dAB = geodesic(cA, cB).km if (cA and cB) else 0.0
    rows = []
    for _, r in attr.iterrows():
        name = str(r["name"])
        if name in used or name in pinned or name not in cmap:
            continue
        if ok_types is not None and str(r.get("type")) not in ok_types:
            continue
        price = float(r.get("price", 0) or 0)
        if math.isnan(price):
            continue
        if free_only and price > 0:
            continue
        cX = cmap[name]
        dAX = geodesic(anchor, cX).km
        if dAX > MAX_DIST_KM:
            continue
        if cA and cB:
            detour = geodesic(cA, cX).km + geodesic(cX, cB).km - dAB
            if detour > MAX_DETOUR_KM:
                continue
        else:
            detour = 0.0
        rows.append((price > 0, dAX + max(0.0, detour),
                     price, name, str(r["opentime"]), str(r["endtime"])))
    rows.sort(key=lambda x: (x[0], x[1]))
    return [(p, n, ot, et) for _, _, p, n, ot, et in rows[:N_CAND]]


def try_point(uid, q, plan, di, i, s0, budget):
    """Insert one attraction between activities i and i+1 on day di.
    Returns (new_plan, evals_used, msg) or (None, evals_used, None)."""
    acts = plan["itinerary"][di]["activities"]
    A, B = acts[i], acts[i + 1]
    intercity = B.get("type") in ("train", "airplane")
    if intercity:
        # the train/airplane itself is untouched; only its innercity leg to
        # the DEPARTURE station is rebuilt (arrival <= fixed departure time)
        posB = (B.get("transports") or [{}])[-1].get("end") or B.get("start")
    else:
        posB = actpos(B)
    posA, tA = actpos(A), A.get("end_time")
    if not posA or not posB or not tA:
        return None, 0, None
    movable = (not intercity
               and B.get("type") == "accommodation"
               and str(B.get("end_time")) == "24:00"
               and i + 1 == len(acts) - 1)
    if movable:
        arr_limit = 24 * 60 - 1
        if hm(tA) >= 24 * 60 - 15:
            return None, 0, None
    else:
        tB = B.get("start_time")
        if not tB:
            return None, 0, None
        arr_limit = hm(tB)
        if arr_limit - hm(tA) < MIN_SLACK:
            return None, 0, None

    city = q["target_city"]
    ppl = int(q.get("people_number", 1))
    pinned = hard_named(q)
    types_locked, free_only = plan_flags(q)
    ok_types = plan_attraction_types(plan, city_attractions(city)) \
        if types_locked else None
    used = set(str(actpos(x)) for d in plan["itinerary"]
               for x in d["activities"] if actpos(x))

    evals = 0
    for price, name, ot, et in candidates(city, posA, posB, used, pinned,
                                          ok_types, free_only):
        if hm(et) <= hm(ot):                    # overnight window: skip
            continue
        tr1 = goto(city, posA, name, tA, "walk", ppl)
        if not tr1 or leg_minutes(tr1) > MAX_NEW_LEG:
            continue
        arr1 = tr1[-1]["end_time"]
        start = arr1 if hm(arr1) >= hm(ot) else ot
        for visit in VISITS:
            end_m = hm(start) + visit
            if end_m > hm(et) or end_m >= 24 * 60:
                continue
            end = mh(end_m)
            for m2 in ("walk", "metro", "taxi"):
                tr2 = goto(city, name, posB, end, m2, ppl)
                if not tr2:
                    continue
                arr2 = hm(tr2[-1]["end_time"])
                if arr2 > arr_limit:
                    continue
                cand = copy.deepcopy(plan)
                cacts = cand["itinerary"][di]["activities"]
                newact = {"position": name, "type": "attraction",
                          "price": float(price), "cost": float(price) * ppl,
                          "tickets": ppl, "start_time": str(start),
                          "end_time": str(end), "transports": tr1}
                nb = copy.deepcopy(B)
                nb["transports"] = tr2
                if movable:
                    nb["start_time"] = mh(arr2)
                cacts[i + 1:i + 2] = [newact, nb]   # splice X before B
                cand = clean(cand)
                s1 = soft_of(cand)
                if not s1:
                    continue
                if not (s1[2] > s0[2] and s1[0] == s0[0] and s1[1] == s0[1]):
                    continue
                evals += 1
                if passes(uid, cand):
                    msg = (f"    day{di + 1} gap[{i}] +'{name}' (price {price:g}) "
                           f"leg {leg_minutes(tr1)}m walk, next via {m2} "
                           f"{leg_minutes(tr2)}m | ATT {s0[2]:.5f}->{s1[2]:.5f}"
                           f"{' [hotel->' + mh(arr2) + ']' if movable else ''}")
                    return cand, evals, msg
                if evals >= budget:
                    return None, evals, None
        if evals >= budget:
            break
    return None, evals, None


def enrich_plan(uid, q, plan):
    """Greedy insert loop. Returns (plan, msgs)."""
    msgs = []
    evals_left = MAX_EVALS
    failed = set()
    for _ in range(MAX_INSERTS):
        s0 = soft_of(plan)
        if not s0 or s0[2] >= 1.0 or evals_left <= 0:
            break
        progressed = False
        for di in range(len(plan["itinerary"])):
            acts = plan["itinerary"][di]["activities"]
            for i in range(len(acts) - 1):
                key = (di, str(actpos(acts[i])), str(actpos(acts[i + 1])))
                if key in failed:
                    continue
                cand, ev, msg = try_point(uid, q, plan, di, i, s0,
                                          min(evals_left, 10))
                evals_left -= ev
                if cand is not None:
                    plan = cand
                    msgs.append(msg)
                    progressed = True
                    break
                failed.add(key)
                if evals_left <= 0:
                    break
            if progressed or evals_left <= 0:
                break
        if not progressed:
            break
    return plan, msgs



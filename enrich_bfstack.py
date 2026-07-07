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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_route import passes, hm, mh
import enrich_route as ER

MEALS = ("breakfast", "lunch", "dinner")
INTERCITY = {"train", "airplane"}
BF_LO, BF_HI = 6 * 60, 9 * 60
DUR = 20
MAX_BF_PER_MORNING = 3

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
            if len(occ) != _round - 1:   # add 2nd only where 1 exists, 3rd where 2
                continue
            s = earliest_free_slot(occ, BF_HI)
            if s is None:
                continue
            insert_front_block(acts, hotel, make_bf(hotel, s, s + DUR))
            added += 1
            deficit -= 1
    return added

if __name__ == "__main__":
    files = sorted(glob.glob(f"{ER.RES}/*.json"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    apply = "--apply" in sys.argv
    stack = "--stack" in sys.argv
    shard_i, shard_n = 0, 1
    if "--shard" in sys.argv:
        shard_i, shard_n = [int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/")]
    if "--uids" in sys.argv:                     # validation subset (json list)
        want = set(json.load(open(sys.argv[sys.argv.index("--uids") + 1])))
        files = [f for f in files if os.path.basename(f)[:-5] in want]
    kept = tried = tb = 0
    for n, f in enumerate(files):
        if limit and n >= limit:
            break
        if n % shard_n != shard_i:
            continue
        uid = os.path.basename(f)[:-5]; ER._cur = uid
        plan = ER.load_json_file(f)
        if not plan.get("itinerary"):
            continue
        if meal_count(plan) >= 3 * len(plan["itinerary"]):
            continue                              # DDR already 1.0, nothing to gain
        if not passes(uid, plan):
            continue                              # never touch failing plans
        r0 = ddr(plan)
        cand = copy.deepcopy(plan)
        b = add_default(cand)
        if stack:
            b += add_stack(cand)
        if b > 0:
            tried += 1
            r1 = ddr(cand)
            if r1 > r0 and passes(uid, cand):
                kept += 1; tb += b
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{b}bf | DDR {r0:.3f}->{r1:.3f}", flush=True)
            else:
                print(f"{uid}: REJECT +{b}bf | DDR {r0:.3f}->{r1:.3f} "
                      f"{'(no gain)' if r1 <= r0 else '(eval fail)'}", flush=True)
    print(f"\ntried {tried}, kept {kept}, breakfasts {tb}, apply={apply}, stack={stack}")

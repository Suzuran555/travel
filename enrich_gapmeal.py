"""Gap lunch enrichment via NEAREST-restaurant recall (workflow Patch 2, post-proc).

262 middle-day lunches sit in a >=40min idle gap but were missed by the on-the-way
enricher: when the two flanking activities are close, almost no restaurant lies
"between" them (tiny detour budget). Here we instead recall the NEAREST open,
unvisited restaurants to the gap and drop the lunch INTO the existing idle time --
no schedule cascade (meal + both transports fit before the next activity's current
start). Every insertion re-validated by the full 3-stage eval; kept only if the
plan still passes ALL hard constraints and DDR improved. Regression-safe."""
import os, sys, json, copy, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_route import (passes, agent_for, hm, mh, actpos, qd)
import enrich_route as ER

INTER = {"train", "airplane"}
WIN = {"lunch": (11 * 60, 14 * 60), "dinner": (17 * 60, 20 * 60), "breakfast": (6 * 60, 9 * 60)}
DUR = 45

def ddr(plan):
    days = max(1, len(plan["itinerary"]))
    n = sum(1 for d in plan["itinerary"] for x in d["activities"]
            if x.get("type") in ("breakfast", "lunch", "dinner"))
    return (n / days) / 3

def nearest_open(ag, query, anchor, lo, hi, visited, k=30):
    df = ag.memory["restaurants"]
    df = df[~df["name"].isin(visited)]
    rows = []
    for _, r in df.iterrows():
        ot, et = str(r.get("opentime")), str(r.get("endtime"))
        # open sometime within [lo,hi]
        try:
            if not (et < ot):  # normal hours
                if hm(et) <= lo or hm(ot) >= hi:
                    continue
        except Exception:
            pass
        try:
            d = ag.calculate_distance(query, anchor, r["name"])
        except Exception:
            continue
        if d is None:
            continue
        rows.append((d, r))
    rows.sort(key=lambda z: z[0])
    return [r for _, r in rows[:k]]

def insert_gap_meal(ag, query, plan, di, meal):
    day = plan["itinerary"][di]
    acts = day["activities"]
    if any(x.get("type") == meal for x in acts):
        return False
    lo, hi = WIN[meal]
    city = query["target_city"]
    visited = set(actpos(x) for d in plan["itinerary"] for x in d["activities"])
    ppl = int(query.get("people_number", 1))
    for i in range(len(acts) - 1):
        x, y = acts[i], acts[i + 1]
        if x.get("type") in INTER or y.get("type") in INTER:
            continue
        posX, posY = actpos(x), actpos(y)
        xe, ys = x.get("end_time"), y.get("start_time")
        if not posX or not posY or not xe or not ys:
            continue
        if min(hm(ys), hi) - max(hm(xe), lo) < 40:      # no usable idle window here
            continue
        for r in nearest_open(ag, query, posX, lo, hi, visited):
            name = r["name"]; ot, et = str(r.get("opentime")), str(r.get("endtime"))
            for mode in ("metro", "walk", "taxi"):
                tr1 = ag.collect_innercity_transport(city, posX, name, xe, mode)
                if not isinstance(tr1, list) or tr1 == []:
                    continue
                arr = tr1[-1]["end_time"]
                start = arr if hm(arr) >= lo else mh(lo)
                if hm(start) >= hi:
                    continue
                if not (et < ot) and (hm(start) < hm(ot) or hm(start) > hm(et)):
                    continue
                end = mh(hm(start) + DUR)
                if hm(end) > hi or hm(end) <= lo:
                    continue
                tr2 = ag.collect_innercity_transport(city, name, posY, end, mode)
                if not isinstance(tr2, list) or tr2 == []:
                    continue
                if hm(tr2[-1]["end_time"]) > hm(ys):      # must fit before next start (no cascade)
                    continue
                price = float(r.get("price", 0) or 0)
                X = {"position": name, "type": meal, "price": price, "cost": price * ppl,
                     "start_time": start, "end_time": end, "transports": tr1}
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

def enrich_plan(ag, query, plan):
    added = 0
    for di in range(len(plan["itinerary"])):
        for meal in ("lunch", "dinner", "breakfast"):
            if insert_gap_meal(ag, query, plan, di, meal):
                added += 1
    return added

if __name__ == "__main__":
    files = sorted(glob.glob(f"{ER.RES}/*.json"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    apply = "--apply" in sys.argv
    shard_i, shard_n = 0, 1
    if "--shard" in sys.argv:
        shard_i, shard_n = [int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/")]
    kept = tried = tot = 0
    for n, f in enumerate(files):
        if limit and n >= limit:
            break
        if n % shard_n != shard_i:
            continue
        uid = os.path.basename(f)[:-5]; ER._cur = uid
        plan = ER.load_json_file(f)
        if not plan.get("itinerary") or not passes(uid, plan):
            continue
        q = qd[uid]; ag = agent_for(q["target_city"]); ag.query = q
        r0 = ddr(plan); cand = copy.deepcopy(plan)
        a = enrich_plan(ag, q, cand)
        if a > 0:
            tried += 1
            r1 = ddr(cand)
            if r1 > r0 and passes(uid, cand):
                kept += 1; tot += a
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{a}meal | DDR {r0:.3f}->{r1:.3f}")
    print(f"\ntried {tried}, kept {kept}, meals {tot}, apply={apply}")

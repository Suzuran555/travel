"""ATT enrichment: swap slow inner-city legs to a faster mode (taxi) where budget
allows (workflow Patch 1, offline-simulated +0.24 Overall).

ATT = clamp((-1/105)*avg_transit_min + 8/7), avg over activities that have
transports. A walk/metro/walk sandwich carries station-walking overhead; the
direct taxi is usually strictly faster, cutting that activity's transit time and
raising ATT. A faster leg only moves THIS activity's start earlier (repair aligns
start to arrival); the next activity departs from this one's unchanged end_time,
so there is NO downstream cascade. Every swap is re-validated with the full
3-stage eval (budget caps included); kept only if the plan still passes ALL hard
constraints and its ATT strictly improved. Regression-safe by construction.

EXTENSION (station legs): activities of type train/airplane carry the in-city
legs TO the station/airport in their transports. Previously skipped; now, when
those legs are slow (walk/metro/walk sandwich, no taxi), we try the direct taxi
via collect_innercity_transport(city, tr[0]['start'], tr[-1]['end'],
tr[0]['start_time'], 'taxi'). Activity start/end times are NEVER touched --
arriving at the station early and waiting is legal -- so no time repair is run
for these swaps. Kept only if the new legs are strictly faster, per-plan ATT
strictly improves, and the full eval still passes."""
import os, sys, json, copy, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_route import (passes, agent_for, actpos, qd)
import enrich_route as ER
from chinatravel.symbol_verification.concept_func import innercity_transport_time

def att_of(plan):
    tc = n = 0
    for d in plan["itinerary"]:
        for a in d["activities"]:
            tr = a.get("transports") or []
            if tr:
                tc += innercity_transport_time(tr); n += 1
    if n == 0:
        return 1.0
    avg = tc / n
    return max(0.0, min(1.0, (-1 / 105) * avg + 8 / 7))

def _modes(tr):
    return [l.get("mode") for l in (tr or [])]

def enrich_plan(ag, query, plan):
    city = query["target_city"]
    swaps = st_swaps = 0
    for di, day in enumerate(plan["itinerary"]):
        acts = day["activities"]
        # collect swap candidates on this day: activities whose current
        # leg is slower than a taxi alternative, biggest time-gain first
        cand = []
        for ai in range(len(acts)):
            cur = acts[ai]
            tr = cur.get("transports") or []
            if not tr:
                continue
            station = cur.get("type") in ("train", "airplane")
            if ai == 0 and not station:    # normal activities need a previous anchor
                continue
            modes = _modes(tr)
            if "taxi" in modes:            # already fastest mode
                continue
            if station:
                # intercity activity: fix the in-city legs TO the station/airport.
                # Origin/dest come from the legs themselves (the previous activity
                # may be in another city / on another day).
                origin = tr[0].get("start"); dest = tr[-1].get("end")
            else:
                origin = actpos(acts[ai - 1]); dest = actpos(cur)
            if not origin or not dest or origin == dest:
                continue
            depart = tr[0].get("start_time")
            if not depart:
                continue
            cur_t = innercity_transport_time(tr)
            cand.append((cur_t, di, ai, origin, dest, depart, station))
        cand.sort(reverse=True)            # largest current transit first
        for cur_t, di_, ai, origin, dest, depart, station in cand:
            # Normal activities: try metro FIRST (cheap, budget-safe -> rescues long
            # WALK legs where the taxi is faster but blows the innercity_transport_cost
            # cap), then taxi (fastest, for metro-sandwich legs with budget slack).
            # Station legs: taxi only (verified spec). Keep the first strictly-faster
            # alternative that still passes and raises ATT.
            for mode in (("taxi",) if station else ("metro", "taxi")):
                alt = ag.collect_innercity_transport(city, origin, dest, depart, mode)
                if not isinstance(alt, list) or alt == []:
                    continue
                if innercity_transport_time(alt) >= cur_t:   # must be strictly faster
                    continue
                trial = copy.deepcopy(plan)
                trial["itinerary"][di_]["activities"][ai]["transports"] = alt
                if not station:
                    try:
                        ag._repair_itinerary_times(trial["itinerary"])
                    except Exception:
                        continue
                # station legs: NO time repair -- departure time is unchanged and the
                # faster legs simply arrive at the station/airport earlier (waiting
                # there is legal); activity start/end times stay untouched.
                if att_of(trial) > att_of(plan) + 1e-9 and passes(ER._cur, trial):
                    plan["itinerary"][di_]["activities"] = trial["itinerary"][di_]["activities"]
                    swaps += 1
                    if station:
                        st_swaps += 1
                    break
    return swaps, st_swaps

if __name__ == "__main__":
    files = sorted(glob.glob(f"{ER.RES}/*.json"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    apply = "--apply" in sys.argv
    if "--uids" in sys.argv:                 # comma list or path to a JSON list
        v = sys.argv[sys.argv.index("--uids") + 1]
        want = set(json.load(open(v))) if os.path.exists(v) else set(v.split(","))
        files = [f for f in files if os.path.basename(f)[:-5] in want]
    shard_i, shard_n = 0, 1
    if "--shard" in sys.argv:
        shard_i, shard_n = [int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/")]
    kept = tried = tot = st_tot = 0
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
        a0 = att_of(plan); cand = copy.deepcopy(plan)
        s, st = enrich_plan(ag, q, cand)
        if s > 0:
            tried += 1
            a1 = att_of(cand)
            if a1 > a0 and passes(uid, cand):
                kept += 1; tot += s; st_tot += st
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{s}taxi ({st}station) | ATT {a0:.3f}->{a1:.3f}")
    print(f"\ntried {tried}, kept {kept}, swaps {tot} ({st_tot} station), apply={apply}")

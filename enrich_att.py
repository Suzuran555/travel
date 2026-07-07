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
strictly improves, and the full eval still passes.

EXTENSION 2 (slow-leg taxi upgrades): legs slower than SLOW_MIN (>15 min, the
ATT break-even) now try TAXI FIRST, metro as fallback. The old metro-first
ladder stranded slow legs: a marginally-faster metro swap was accepted and the
taxi upgrade never attempted in the same sweep, so walk/metro sandwiches stayed
above 15 min even when a budget-feasible taxi existed (242 such legs in the
top5 residual scan). Legs <=15 min keep the metro-first ordering (budget-safe).
A cheap budget precheck (binding total_cost / inner_city_transportation_cost
caps parsed from hard_logic_py, +10 yuan tolerance) skips taxi lookups that
clearly blow the caps -- passes() remains the authoritative budget guard.
ATT guard: swaps only ever shorten legs and acceptance requires plan-ATT to
strictly improve, so a plan at ATT>=1 (avg<=15) can never be dragged below it.
Idempotent: legs already containing a taxi segment are skipped."""
import os, sys, json, copy, glob, re
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

SLOW_MIN = 15          # ATT break-even: legs above this get taxi-preferred ordering
BUDGET_TOL = 10.0      # yuan tolerance on the cheap precheck (passes() is authoritative)

def _budget_caps(query):
    """Binding cost caps from hard_logic_py: (total_cost cap, inner_city cap).
    A total_cost cap only binds transport fares when the constraint actually
    sums innercity transport cost (same linkage the residual scan used)."""
    tot = inner = None
    for h in query.get("hard_logic_py", []):
        hh = re.sub(r"\s+", "", h)
        if "innercity_transport_cost" in hh:
            for m in re.finditer(r"total_cost<=(\d+(?:\.\d+)?)", hh):
                v = float(m.group(1)); tot = v if tot is None else min(tot, v)
        for m in re.finditer(r"inner_city_transportation_cost<=(\d+(?:\.\d+)?)", hh):
            v = float(m.group(1)); inner = v if inner is None else min(inner, v)
    return tot, inner

def _budget_slack(query, plan):
    """Remaining yuan under the tightest binding cap (inf if uncapped)."""
    tcap, icap = _budget_caps(query)
    if tcap is None and icap is None:
        return float("inf")
    total = inner = 0.0
    for day in plan["itinerary"]:
        for a in day["activities"]:
            total += float(a.get("cost", 0) or 0)
            for s in a.get("transports") or []:
                c = float(s.get("cost", 0) or 0)
                total += c; inner += c
    return min(tcap - total if tcap is not None else float("inf"),
               icap - inner if icap is not None else float("inf"))

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
            # Mode ladder:
            # - station legs: taxi only (verified spec, unchanged).
            # - slow legs (> SLOW_MIN): taxi FIRST (metro->taxi upgrades; the old
            #   metro-first order stranded these above 15 min), metro fallback.
            # - fast legs (<= SLOW_MIN): metro FIRST (cheap, budget-safe -> rescues
            #   long WALK legs where the taxi blows the innercity cost cap).
            # Keep the first strictly-faster alternative that still passes and
            # raises ATT.
            if station:
                ladder = ("taxi",)
            elif cur_t > SLOW_MIN:
                ladder = ("taxi", "metro")
            else:
                ladder = ("metro", "taxi")
            for mode in ladder:
                alt = ag.collect_innercity_transport(city, origin, dest, depart, mode)
                if not isinstance(alt, list) or alt == []:
                    continue
                if innercity_transport_time(alt) >= cur_t:   # must be strictly faster
                    continue
                if mode == "taxi" and not station:
                    # cheap budget precheck: skip fares that clearly exceed the
                    # binding caps' remaining slack (full eval still re-checks)
                    old_tr = plan["itinerary"][di_]["activities"][ai].get("transports") or []
                    fare_delta = (sum(float(s.get("cost", 0) or 0) for s in alt)
                                  - sum(float(s.get("cost", 0) or 0) for s in old_tr))
                    if fare_delta > _budget_slack(query, plan) + BUDGET_TOL:
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

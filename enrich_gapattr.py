"""Gap attraction enrichment v2 (DAV lever, post-proc).

For each PASSING plan whose attraction count falls short of the DAV cap
(shortfall = 4*days - total_attractions > 0), find idle windows >= 60 min
between consecutive same-day activities and drop an unvisited attraction INTO
the idle time -- no schedule cascade (attraction + both transports must fit
before the next activity's current start). Dwell is adaptive to the window:
60 min for gaps >= 100, 45 for >= 75, 30 for >= 60.

Candidate ranking (recall k=20): pure nearest distance to the gap anchor;
FREE-first ordering applies ONLY when a binding (non-disjunctive) total-cost
budget has <= 200 CNY slack. Tight plans are NOT skipped up front any more:
  - total-cost slack < 60 CNY  -> restrict to FREE attractions + WALK-only
    transports (zero cost, cannot violate the budget);
  - inner_city_transportation_cost cap with slack < 15 CNY -> same walk-only
    mode (walk transports cost 0; metro/taxi skipped entirely).
Candidates must be open during the gap and are pre-filtered against the
query's attraction name/type EXCLUSION hard constraints.

Acceptance: every insertion is re-validated by the full 3-stage eval; an
insert is additionally rejected if it would drag a plan from avg transit
<= 15 min (ATT == 1) to avg > 15 (ATT guard). A plan is kept only if per-plan
(dDAV + dATT) > 0 AND it still passes ALL hard constraints. Regression-safe
and idempotent (re-runs find no strict improvement on already-kept plans).

CLI: --shard i/N  --limit K  --apply  --uids <json-file with a uid list>.

Aggregate overfill (>4 attractions on one day) is legal: DAV counts the plan
total, and the eval has been shown to accept it. tickets == people_number."""
import os, sys, json, copy, glob, re, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_route import passes, agent_for, hm, mh, actpos, qd
import enrich_route as ER

INTER = {"train", "airplane"}
MIN_GAP = 60               # smallest usable idle window (minutes)
DWELL_TIERS = ((100, 60), (75, 45), (60, 30))   # (min gap -> dwell minutes)
MIN_BUDGET_SLACK = 60.0    # CNY; below this: FREE+WALK-only mode (not a skip)
FREE_FIRST_SLACK = 200.0   # CNY; free-first ranking only when slack <= this
IC_WALK_SLACK = 15.0       # CNY; inner-city cost slack below this: walk-only
ATT_GUARD_AVG = 15.0       # min; avg transit ceiling for plans already at ATT=1
K_RECALL = 20              # candidates per gap slot
ALL_MODES = ("metro", "walk", "taxi")
MAX_INSERTS = 12           # per-plan insert guard
MAX_EVALS_SLOT = 4         # passes() calls per gap-slot attempt
MAX_EVALS_PLAN = 60        # total passes() budget per plan

# ---------- metrics ----------
def attr_count(plan):
    return sum(1 for d in plan["itinerary"] for a in d["activities"]
               if a.get("type") == "attraction")

def dav(plan):
    days = max(1, len(plan["itinerary"]))
    return min(1.0, attr_count(plan) / (4.0 * days))

def avg_transit(plan):
    T = 0; n = 0
    for d in plan["itinerary"]:
        for a in d["activities"]:
            trs = a.get("transports") or []
            if trs:
                n += 1
                T += sum(hm(t["end_time"]) - hm(t["start_time"]) for t in trs)
    return (T / n) if n else 0.0

def att_metric(plan):
    return max(0.0, min(1.0, (-1.0 / 105.0) * avg_transit(plan) + 8.0 / 7.0))

def plan_cost(plan):
    tot = 0.0
    for d in plan["itinerary"]:
        for a in d["activities"]:
            tot += float(a.get("cost", 0) or 0)
            for tr in a.get("transports", []) or []:
                tot += float(tr.get("cost", 0) or 0)
    return tot

def ic_cost(plan):
    """Inner-city transport cost: sum of transport['cost'] (walk == 0)."""
    tot = 0.0
    for d in plan["itinerary"]:
        for a in d["activities"]:
            for tr in a.get("transports", []) or []:
                tot += float(tr.get("cost", 0) or 0)
    return tot

# ---------- query verifier pre-filters ----------
_NAME_EX = re.compile(r'not\(\s*(\{[^}]*\})\s*&\s*attraction_name_set\s*\)')
_TYPE_EX = re.compile(r'not\(\s*(\{[^}]*\})\s*&\s*attraction_type_set\s*\)')
_BUDGET = re.compile(r'(?<![A-Za-z_])total_cost\s*<=?\s*([0-9.]+)')
_IC_CAP = re.compile(r'(?<![A-Za-z_])inner_city_transportation_cost\s*<=?\s*([0-9.]+)')

def excl_sets(q):
    """Excluded attraction names/types (conservative: any block, even disjunctive)."""
    names, types = set(), set()
    for hl in q.get("hard_logic_py", []) or []:
        for m in _NAME_EX.finditer(hl):
            try:
                names |= set(ast.literal_eval(m.group(1)))
            except Exception:
                pass
        for m in _TYPE_EX.finditer(hl):
            try:
                types |= set(ast.literal_eval(m.group(1)))
            except Exception:
                pass
    return names, types

def _binding_cap(q, rx):
    """Smallest NON-disjunctive cap matched by rx, or None."""
    b = None
    for hl in q.get("hard_logic_py", []) or []:
        if "result=result or r" in hl:      # disjunctive block: bound not binding alone
            continue
        m = rx.search(hl)
        if m:
            v = float(m.group(1))
            b = v if b is None else min(b, v)
    return b

def binding_budget(q):
    return _binding_cap(q, _BUDGET)

def binding_ic_cap(q):
    return _binding_cap(q, _IC_CAP)

# ---------- candidate recall ----------
def nearest_attr(ag, query, anchor, lo, hi, visited, ex_names, ex_types, dwell,
                 free_first=False, free_only=False, k=K_RECALL):
    df = ag.memory["attractions"]
    df = df[~df["name"].isin(visited)]
    rows = []
    for _, r in df.iterrows():
        name = r["name"]
        if name in ex_names or str(r.get("type")) in ex_types:
            continue
        price = float(r.get("price", 0) or 0)
        if free_only and price > 0:
            continue
        try:
            ot, et = str(r.get("opentime")), str(r.get("endtime"))
            if not (et < ot):  # normal hours: must overlap [lo,hi] with room for dwell
                if hm(et) - 10 <= lo + 5 or hm(ot) >= hi - dwell:
                    continue
        except Exception:
            continue
        try:
            d = ag.calculate_distance(query, anchor, name)
        except Exception:
            continue
        if d is None:
            continue
        rows.append(((price > 0, d), r))
    if free_first:
        rows.sort(key=lambda z: z[0])          # (paid?, distance)
    else:
        rows.sort(key=lambda z: z[0][1])       # pure distance
    return [r for _, r in rows[:k]]

def gap_dwell(gap):
    for g, dw in DWELL_TIERS:
        if gap >= g:
            return dw
    return None

# ---------- insertion ----------
def insert_gap_attr(ag, query, plan, di, ex_names, ex_types,
                    modes=ALL_MODES, free_first=False, free_only=False, att_cap=None):
    """Try one attraction insertion into an idle gap on day di.
    Returns (inserted: bool, n_passes_calls: int)."""
    day = plan["itinerary"][di]
    acts = day["activities"]
    city = query["target_city"]
    visited = set(actpos(x) for d in plan["itinerary"] for x in d["activities"])
    ppl = int(query.get("people_number", 1))
    npass = 0
    for i in range(len(acts) - 1):
        x, y = acts[i], acts[i + 1]
        if y.get("type") in INTER:
            continue
        posX, posY = actpos(x), actpos(y)
        xe, ys = x.get("end_time"), y.get("start_time")
        if not posX or not posY or not xe or not ys:
            continue
        if hm(ys) < hm(xe):
            continue
        lo, hi = hm(xe), hm(ys)
        dwell = gap_dwell(hi - lo)
        if dwell is None:
            continue
        for r in nearest_attr(ag, query, posX, lo, hi, visited, ex_names, ex_types,
                              dwell, free_first=free_first, free_only=free_only):
            name = r["name"]; ot, et = str(r.get("opentime")), str(r.get("endtime"))
            for mode in modes:
                tr1 = ag.collect_innercity_transport(city, posX, name, xe, mode)
                if not isinstance(tr1, list) or tr1 == []:
                    continue
                arr = tr1[-1]["end_time"]
                start = arr
                if not (et < ot) and hm(arr) < hm(ot):
                    start = ot
                if not (et < ot) and hm(start) + dwell > hm(et):
                    continue
                end = mh(hm(start) + dwell)
                if hm(end) > hi:
                    continue
                tr2 = ag.collect_innercity_transport(city, name, posY, end, mode)
                if not isinstance(tr2, list) or tr2 == []:
                    continue
                if hm(tr2[-1]["end_time"]) > hm(ys):   # must fit before next start (no cascade)
                    continue
                price = float(r.get("price", 0) or 0)
                X = {"position": name, "type": "attraction", "price": price,
                     "cost": price * ppl, "tickets": ppl,
                     "start_time": start, "end_time": end, "transports": tr1}
                newY = copy.deepcopy(y); newY["transports"] = tr2
                cand = copy.deepcopy(plan)
                cand["itinerary"][di]["activities"] = acts[:i + 1] + [X, newY] + acts[i + 2:]
                try:
                    ag._repair_itinerary_times(cand["itinerary"])
                except Exception:
                    continue
                if att_cap is not None and avg_transit(cand) > att_cap:
                    continue                            # ATT guard (pre-eval, cheap)
                npass += 1
                if passes(ER._cur, cand):
                    plan["itinerary"] = cand["itinerary"]   # adopt validated repaired state
                    return True, npass
                if npass >= MAX_EVALS_SLOT:
                    return False, npass
    return False, npass

def enrich_plan(ag, query, plan, shortfall, ex_names, ex_types,
                modes=ALL_MODES, free_first=False, free_only=False, att_cap=None):
    added = evals = 0
    days = len(plan["itinerary"])
    progress = True
    while shortfall - added > 0 and added < MAX_INSERTS and progress and evals < MAX_EVALS_PLAN:
        progress = False
        for di in range(days):
            if shortfall - added <= 0 or evals >= MAX_EVALS_PLAN:
                break
            ok, np_ = insert_gap_attr(ag, query, plan, di, ex_names, ex_types,
                                      modes=modes, free_first=free_first,
                                      free_only=free_only, att_cap=att_cap)
            evals += np_
            if ok:
                added += 1; progress = True
    return added, evals

if __name__ == "__main__":
    files = sorted(glob.glob(f"{ER.RES}/*.json"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    apply = "--apply" in sys.argv
    shard_i, shard_n = 0, 1
    if "--shard" in sys.argv:
        shard_i, shard_n = [int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/")]
    uid_filter = None
    if "--uids" in sys.argv:
        uid_filter = set(json.load(open(sys.argv[sys.argv.index("--uids") + 1])))

    # cheap prefilter: shard/limit over DAV-short plans ONLY, so shards stay balanced
    short_files = []
    for f in files:
        if uid_filter is not None and os.path.basename(f)[:-5] not in uid_filter:
            continue
        try:
            p = json.load(open(f))
        except Exception:
            continue
        it = p.get("itinerary") or []
        if not it:
            continue
        if 4 * len(it) - sum(1 for d in it for a in d["activities"]
                             if a.get("type") == "attraction") > 0:
            short_files.append(f)

    kept = tried = tot = 0
    done = 0
    for n, f in enumerate(short_files):
        if n % shard_n != shard_i:
            continue
        if limit and done >= limit:
            break
        done += 1
        uid = os.path.basename(f)[:-5]; ER._cur = uid
        plan = ER.load_json_file(f)
        if not plan.get("itinerary"):
            continue
        days = len(plan["itinerary"])
        shortfall = 4 * days - attr_count(plan)
        if shortfall <= 0:
            continue
        q = qd[uid]
        if not passes(uid, plan):
            continue
        # per-plan mode from budget / inner-city-cost slack (no up-front skips)
        bud = binding_budget(q)
        slack = (bud - plan_cost(plan)) if bud is not None else None
        icap = binding_ic_cap(q)
        ic_slack = (icap - ic_cost(plan)) if icap is not None else None
        modes, free_first, free_only, tags = ALL_MODES, False, False, []
        if slack is not None and slack < MIN_BUDGET_SLACK:
            modes, free_only = ("walk",), True
            tags.append(f"budget-tight({slack:.0f})->free+walk")
        elif slack is not None and slack <= FREE_FIRST_SLACK:
            free_first = True
            tags.append(f"free-first({slack:.0f})")
        if ic_slack is not None and ic_slack < IC_WALK_SLACK:
            modes, free_only = ("walk",), True
            tags.append(f"ic-tight({ic_slack:.0f})->free+walk")
        base_avg = avg_transit(plan)
        att_cap = ATT_GUARD_AVG if base_avg <= ATT_GUARD_AVG else None
        ag = agent_for(q["target_city"]); ag.query = q
        ex_names, ex_types = excl_sets(q)
        d0, a0 = dav(plan), att_metric(plan)
        cand = copy.deepcopy(plan)
        added, evals = enrich_plan(ag, q, cand, shortfall, ex_names, ex_types,
                                   modes=modes, free_first=free_first,
                                   free_only=free_only, att_cap=att_cap)
        note = (" [" + ",".join(tags) + "]") if tags else ""
        if added > 0:
            tried += 1
            d1, a1 = dav(cand), att_metric(cand)
            guard_ok = att_cap is None or avg_transit(cand) <= att_cap
            if (d1 + a1) > (d0 + a0) and guard_ok and passes(uid, cand):
                kept += 1; tot += added
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{added}attr (evals={evals}) | DAV {d0:.3f}->{d1:.3f} "
                      f"ATT {a0:.3f}->{a1:.3f}{note}")
            else:
                why = "att-guard" if not guard_ok else "delta"
                print(f"{uid}: rejected +{added}attr ({why}, evals={evals}) | "
                      f"dDAV+dATT={(d1 + a1) - (d0 + a0):+.4f}{note}")
    print(f"\ntried {tried}, kept {kept}, attractions {tot}, apply={apply}")

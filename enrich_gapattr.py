"""Gap attraction enrichment (DAV lever, post-proc).

For each PASSING plan whose attraction count falls short of the DAV cap
(shortfall = 4*days - total_attractions > 0), find idle windows >= 100 min
between consecutive same-day activities and drop an unvisited attraction INTO
the idle time -- no schedule cascade (attraction + both transports must fit
before the next activity's current start). Candidates are recalled nearest to
the gap anchor with FREE attractions first (sort key: price>0, distance), must
be open during the gap, and are pre-filtered against the query's attraction
name/type EXCLUSION hard constraints. Plans whose binding (non-disjunctive)
total-cost budget has < 60 CNY slack are skipped up front. Every insertion is
re-validated by the full 3-stage eval; a plan is kept only if per-plan
(dDAV + dATT) > 0 AND it still passes ALL hard constraints. Regression-safe.

Aggregate overfill (>4 attractions on one day) is legal: DAV counts the plan
total, and the eval has been shown to accept it. tickets == people_number."""
import os, sys, json, copy, glob, re, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_route import passes, agent_for, hm, mh, actpos, qd
import enrich_route as ER

INTER = {"train", "airplane"}
DWELL = 60                 # attraction dwell minutes
MIN_GAP = 100              # need transit + dwell + transit
MIN_BUDGET_SLACK = 60.0    # CNY; skip tighter plans
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

def att_metric(plan):
    T = 0; n = 0
    for d in plan["itinerary"]:
        for a in d["activities"]:
            trs = a.get("transports") or []
            if trs:
                n += 1
                T += sum(hm(t["end_time"]) - hm(t["start_time"]) for t in trs)
    avg = (T / n) if n else 0.0
    return max(0.0, min(1.0, (-1.0 / 105.0) * avg + 8.0 / 7.0))

def plan_cost(plan):
    tot = 0.0
    for d in plan["itinerary"]:
        for a in d["activities"]:
            tot += float(a.get("cost", 0) or 0)
            for tr in a.get("transports", []) or []:
                tot += float(tr.get("cost", 0) or 0)
    return tot

# ---------- query verifier pre-filters ----------
_NAME_EX = re.compile(r'not\(\s*(\{[^}]*\})\s*&\s*attraction_name_set\s*\)')
_TYPE_EX = re.compile(r'not\(\s*(\{[^}]*\})\s*&\s*attraction_type_set\s*\)')
_BUDGET = re.compile(r'(?<![A-Za-z_])total_cost\s*<=?\s*([0-9.]+)')

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

def binding_budget(q):
    """Smallest NON-disjunctive total_cost bound, or None."""
    b = None
    for hl in q.get("hard_logic_py", []) or []:
        if "result=result or r" in hl:      # disjunctive block: bound not binding alone
            continue
        m = _BUDGET.search(hl)
        if m:
            v = float(m.group(1))
            b = v if b is None else min(b, v)
    return b

# ---------- candidate recall ----------
def nearest_attr(ag, query, anchor, lo, hi, visited, ex_names, ex_types, k=10):
    df = ag.memory["attractions"]
    df = df[~df["name"].isin(visited)]
    rows = []
    for _, r in df.iterrows():
        name = r["name"]
        if name in ex_names or str(r.get("type")) in ex_types:
            continue
        try:
            ot, et = str(r.get("opentime")), str(r.get("endtime"))
            if not (et < ot):  # normal hours: must overlap [lo,hi] with room for DWELL
                if hm(et) - 10 <= lo + 5 or hm(ot) >= hi - DWELL:
                    continue
        except Exception:
            continue
        try:
            d = ag.calculate_distance(query, anchor, name)
        except Exception:
            continue
        if d is None:
            continue
        price = float(r.get("price", 0) or 0)
        rows.append(((price > 0, d), r))
    rows.sort(key=lambda z: z[0])
    return [r for _, r in rows[:k]]

# ---------- insertion ----------
def insert_gap_attr(ag, query, plan, di, ex_names, ex_types):
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
        if hi - lo < MIN_GAP:
            continue
        for r in nearest_attr(ag, query, posX, lo, hi, visited, ex_names, ex_types):
            name = r["name"]; ot, et = str(r.get("opentime")), str(r.get("endtime"))
            for mode in ("metro", "walk", "taxi"):
                tr1 = ag.collect_innercity_transport(city, posX, name, xe, mode)
                if not isinstance(tr1, list) or tr1 == []:
                    continue
                arr = tr1[-1]["end_time"]
                start = arr
                if not (et < ot) and hm(arr) < hm(ot):
                    start = ot
                if not (et < ot) and hm(start) + DWELL > hm(et):
                    continue
                end = mh(hm(start) + DWELL)
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
                npass += 1
                if passes(ER._cur, cand):
                    plan["itinerary"] = cand["itinerary"]   # adopt validated repaired state
                    return True, npass
                if npass >= MAX_EVALS_SLOT:
                    return False, npass
    return False, npass

def enrich_plan(ag, query, plan, shortfall, ex_names, ex_types):
    added = evals = 0
    days = len(plan["itinerary"])
    progress = True
    while shortfall - added > 0 and added < MAX_INSERTS and progress and evals < MAX_EVALS_PLAN:
        progress = False
        for di in range(days):
            if shortfall - added <= 0 or evals >= MAX_EVALS_PLAN:
                break
            ok, np_ = insert_gap_attr(ag, query, plan, di, ex_names, ex_types)
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

    # cheap prefilter: shard/limit over DAV-short plans ONLY, so shards stay balanced
    short_files = []
    for f in files:
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
        bud = binding_budget(q)
        if bud is not None and bud - plan_cost(plan) < MIN_BUDGET_SLACK:
            print(f"{uid}: skip budget slack {bud - plan_cost(plan):.0f} < {MIN_BUDGET_SLACK:.0f}")
            continue
        if not passes(uid, plan):
            continue
        ag = agent_for(q["target_city"]); ag.query = q
        ex_names, ex_types = excl_sets(q)
        d0, a0 = dav(plan), att_metric(plan)
        cand = copy.deepcopy(plan)
        added, evals = enrich_plan(ag, q, cand, shortfall, ex_names, ex_types)
        if added > 0:
            tried += 1
            d1, a1 = dav(cand), att_metric(cand)
            if (d1 + a1) > (d0 + a0) and passes(uid, cand):
                kept += 1; tot += added
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{added}attr (evals={evals}) | DAV {d0:.3f}->{d1:.3f} ATT {a0:.3f}->{a1:.3f}")
            else:
                print(f"{uid}: rejected +{added}attr (evals={evals}) | dDAV+dATT={(d1 + a1) - (d0 + a0):+.4f}")
    print(f"\ntried {tried}, kept {kept}, attractions {tot}, apply={apply}")

"""Evening-append attraction enrichment (DAV lever, zero-window plans).

The gap-based attraction enrichers (enrich_gapattr etc.) need an idle window
BETWEEN two activities. Structural plans (travel-day starved, dense days) have
no such windows -- but many end the day early: last real activity done by
~18:00, then straight back to the hotel. The hotel arrival time is ELASTIC
(accommodation end is always 24:00; enrich_endday proved the shift-later
cascade passes the full eval).

This enricher appends ONE attraction per day after the last non-accommodation
activity, when that activity ends <= 20:15 (>= 75 min before the 21:30 cap):

  last_act -> [tr1] -> NEW attraction (dwell 45-60 adaptive, ends <= 21:30,
  within open hours) -> [tr2 rebuilt] -> accommodation (arrival shifts later)

Candidates: ag.memory['attractions'], unvisited, not name/type-excluded by the
query's hard logic, open at that hour; nearest-first from the last activity
(free-first when budget slack <= 200). Transports tried metro -> walk -> taxi.

Acceptance (regression-safe):
  * dDAV > 0 (only plans with attr < 4*days are touched; capped per plan)
  * ATT guard: a plan whose avg transit <= 15 min (ATT == 1) must stay <= 15
  * full 3-stage passes() (schema + commonsense + hard logic) per insertion
    and again on the final plan.

tickets = people_number, cost = price * people_number (commonsense requires
cost == price * tickets for attractions).
"""
import os, sys, json, copy, glob, re, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_route import passes, agent_for, hm, mh, actpos, qd
import enrich_route as ER

INTER = {"train", "airplane"}
DAY_CAP = 21 * 60 + 30      # appended attraction must end by 21:30
LAST_END = 20 * 60 + 15     # last real activity must end by 20:15
MAX_CAND = 14               # candidates tried per day slot
ATT_ONE = 15.0              # avg transit minutes at which ATT == 1

_NAME_EX = re.compile(r'not\(\s*(\{[^}]*\})\s*&\s*attraction_name_set\s*\)')
_TYPE_EX = re.compile(r'not\(\s*(\{[^}]*\})\s*&\s*attraction_type_set\s*\)')
_BUDGET = re.compile(r'(?<![A-Za-z_])total_cost\s*<=?\s*([0-9.]+)')


def query_meta(query):
    """(excluded names, excluded types, budget or None) from hard_logic_py."""
    names, types, bud = set(), set(), None
    for hl in query.get("hard_logic_py", []) or []:
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
        if "result=result or r" not in hl:
            m = _BUDGET.search(hl)
            if m:
                v = float(m.group(1))
                bud = v if bud is None else min(bud, v)
    return names, types, bud


def plan_cost(plan):
    tot = 0.0
    for d in plan["itinerary"]:
        for a in d["activities"]:
            tot += float(a.get("cost", 0) or 0)
            for tr in a.get("transports", []) or []:
                tot += float(tr.get("cost", 0) or 0)
    return tot


def attr_count(plan):
    return sum(1 for d in plan["itinerary"] for a in d["activities"]
               if a.get("type") == "attraction")


def dav(plan):
    days = max(1, len(plan["itinerary"]))
    return min(1.0, attr_count(plan) / (4 * days))


def att_avg(plan):
    """avg transit minutes over activities WITH transports (the ATT input)."""
    tot = n = 0
    for d in plan["itinerary"]:
        for a in d["activities"]:
            trs = a.get("transports") or []
            if trs:
                n += 1
                tot += sum(hm(t["end_time"]) - hm(t["start_time"]) for t in trs)
    return tot / n if n else 0.0


def _last_real_idx(acts):
    for i in range(len(acts) - 1, -1, -1):
        t = acts[i].get("type")
        if t != "accommodation" and t not in INTER:
            return i
    return -1


def append_evening_attraction(ag, query, plan, di, ex_names, ex_types, budget):
    """Append one attraction between the last real activity of day di and the
    following accommodation (hotel arrival shifts later). Validated in-place."""
    day = plan["itinerary"][di]
    acts = day["activities"]
    li = _last_real_idx(acts)
    if li < 0 or li + 1 >= len(acts):
        return False
    acc = acts[li + 1]
    if acc.get("type") != "accommodation":
        return False
    A, H = acts[li], acc
    posA, posH = actpos(A), H.get("position")
    tA = A.get("end_time")
    if not posA or not posH or not tA or hm(tA) > LAST_END:
        return False

    city = query["target_city"]
    ppl = int(query.get("people_number", 1))
    visited = set(actpos(x) for d in plan["itinerary"] for x in d["activities"])
    slack = (budget - plan_cost(plan)) if budget is not None else None

    df = ag.memory["attractions"]
    df = df[~df["name"].isin(visited)]
    if ex_names:
        df = df[~df["name"].isin(ex_names)]
    if ex_types and "type" in df.columns:
        df = df[~df["type"].astype(str).isin(ex_types)]

    scored = []
    for _, r in df.iterrows():
        name = r["name"]
        try:
            ot, et = hm(str(r.get("opentime"))), hm(str(r.get("endtime")))
        except Exception:
            continue
        overnight = et <= ot
        close = DAY_CAP if overnight else min(DAY_CAP, et)
        # cheap viability gate: leave 20:15+45min minimum even before transit
        if not overnight and max(hm(tA), ot) + 45 > close:
            continue
        price = float(r.get("price", 0) or 0)
        if slack is not None and price * ppl > max(0.0, slack):
            continue
        try:
            dist = ag.calculate_distance(query, posA, name)
        except Exception:
            continue
        if dist is None:
            continue
        free_first = slack is not None and slack <= 200
        key = ((0 if price == 0 else 1, dist) if free_first else (dist,))
        scored.append((key, name, ot, et, overnight, price))
    scored.sort(key=lambda x: x[0])

    base_att = att_avg(plan)
    for _, name, ot, et, overnight, price in scored[:MAX_CAND]:
        for mode in ("metro", "walk", "taxi"):
            tr1 = ag.collect_innercity_transport(city, posA, name, tA, mode)
            if not isinstance(tr1, list) or tr1 == []:
                continue
            arr = tr1[-1]["end_time"]
            start_m = max(hm(arr), ot)
            close = DAY_CAP if overnight else min(DAY_CAP, et)
            dwell = 60 if start_m + 60 <= close else 45
            if start_m + dwell > close:
                continue
            start, end = mh(start_m), mh(start_m + dwell)
            tr2 = ag.collect_innercity_transport(city, name, posH, end, mode)
            if not isinstance(tr2, list):
                continue
            aH = tr2[-1]["end_time"] if tr2 else end
            if hm(aH) >= 24 * 60:
                continue
            X = {"position": name, "type": "attraction", "price": price,
                 "cost": price * ppl, "tickets": ppl,
                 "start_time": start, "end_time": end, "transports": tr1}
            newH = copy.deepcopy(H)
            newH["transports"] = tr2
            newH["start_time"] = aH
            cand = copy.deepcopy(plan)
            cand["itinerary"][di]["activities"] = (acts[:li + 1] + [X, newH]
                                                   + acts[li + 2:])
            try:
                ag._repair_itinerary_times(cand["itinerary"])
            except Exception:
                continue
            # ATT guard: never drag an ATT==1 plan (avg<=15) below the cap
            if base_att <= ATT_ONE and att_avg(cand) > ATT_ONE:
                continue
            if passes(ER._cur, cand):
                day["activities"] = cand["itinerary"][di]["activities"]
                return True
    return False


def enrich_plan(ag, query, plan, ex_names, ex_types, budget):
    added = 0
    cap = 4 * len(plan["itinerary"])
    for di in range(len(plan["itinerary"])):
        if attr_count(plan) >= cap:
            break
        if append_evening_attraction(ag, query, plan, di, ex_names, ex_types,
                                     budget):
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
    done = 0
    for n, f in enumerate(files):
        if n % shard_n != shard_i:
            continue
        if limit and done >= limit:
            break
        done += 1
        uid = os.path.basename(f)[:-5]
        ER._cur = uid
        plan = ER.load_json_file(f)
        it = plan.get("itinerary") or []
        if not it:
            continue
        days = len(it)
        if attr_count(plan) >= 4 * days:
            continue                     # DAV already 1: nothing to gain
        # evening slot pre-check before the expensive baseline passes()
        has_slot = False
        for d in it:
            acts = d["activities"]
            li = _last_real_idx(acts)
            if (li >= 0 and li + 1 < len(acts)
                    and acts[li + 1].get("type") == "accommodation"
                    and acts[li].get("end_time")
                    and hm(acts[li]["end_time"]) <= LAST_END):
                has_slot = True
                break
        if not has_slot:
            continue
        if not passes(uid, plan):
            continue
        q = qd[uid]
        ag = agent_for(q["target_city"])
        ag.query = q
        ex_names, ex_types, budget = query_meta(q)
        d0, a0 = dav(plan), att_avg(plan)
        cand = copy.deepcopy(plan)
        add = enrich_plan(ag, q, cand, ex_names, ex_types, budget)
        if add > 0:
            tried += 1
            d1, a1 = dav(cand), att_avg(cand)
            att_ok = not (a0 <= ATT_ONE and a1 > ATT_ONE)
            if d1 > d0 and att_ok and passes(uid, cand):
                kept += 1
                tot += add
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{add}attr | DAV {d0:.3f}->{d1:.3f} | "
                      f"ATTavg {a0:.1f}->{a1:.1f}", flush=True)
    print(f"\ntried {tried}, kept {kept}, attractions {tot}, apply={apply}")

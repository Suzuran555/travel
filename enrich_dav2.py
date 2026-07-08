"""DAV enrichment v2 -- four validated residual angles in one pass (post-proc).

Only PASSING plans with attraction shortfall (4*days - attr_count > 0) are
touched. Per plan, five deficit-gated phases run in order (each stops when the
shortfall is filled), every candidate insert is validated by the FULL 3-stage
eval (schema + commonsense + hard logic) before adoption:

  1. gap    -- enrich_gapattr semantics extended to 30-59 min idle windows
               (dwell tiers 60/45/30/25/20; transit must fit inside the window,
               no cascade). Also retries >=60 windows (wide-retry).
  2. eve    -- evening append to FIXED POINT: enrich_endattr's per-day append
               wrapped in a while-progress loop, so 2x-capacity evenings absorb
               two attractions in one run (elastic hotel arrival).
  3. station-- idle windows adjacent to intercity legs, invisible to all other
               enrichers: pre_dep (last positioned act -> departure station,
               dest = ic_act['start'], NO cascade) and post_arr (arrival
               station -> next act; elastic when only accommodation follows;
               chains multiple inserts). Mode pairs walk/metro/taxi mixed,
               ic-cost delta pre-check for binding inner-city caps.
  4. morning-- post-breakfast prepend with bounded SHIFT-Y: insert after the
               LAST breakfast of the 06:00 stack when the first real activity
               starts >=60 min later; Y may start up to 90 min later (duration
               preserved, must still fit its DB close time); only the single
               next activity Z gets rebuilt transports (accommodation elastic);
               no deeper cascade.
  5. shrink -- trade 30 min of an existing attraction's dwell (>=90 kept >=60)
               to widen the following gap, adopt ONLY if the insert into the
               widened gap validates end-to-end.

Shared gates (identical to the applied v2 enrichers):
  * per-plan mode from budget slack (<60 CNY -> FREE+WALK-only; <=200 ->
    free-first ranking) and inner-city cost slack (<15 CNY -> walk-only);
  * ATT guard: a plan with avg transit <= 15 min (ATT == 1) must stay <= 15;
  * tickets = people_number, cost = price * tickets;
  * name/type exclusion prefilter from the query's hard logic;
  * plan keep rule: dDAV > 0 AND ATT guard AND final full passes().
Idempotent: shortfall <= 0 plans are skipped; re-runs keep 0.

CLI: --shard i/N  --limit K  --apply  --uids <json-file with a uid list>.
Run with ENRICH_RES exported BEFORE python starts (it is read at import time).
"""
import os, sys, json, copy, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_route import passes, agent_for, hm, mh, actpos, qd
import enrich_route as ER
import enrich_gapattr as GA
import enrich_endattr as EA

INTER = GA.INTER
TIERS = ((100, 60), (75, 45), (60, 30), (45, 25), (30, 20))  # min gap -> dwell
SHRINK, MIN_KEEP = 30, 60      # shrink step / minimum dwell an attraction keeps
MAX_SHIFT = 90                 # minutes Y may start later (morning phase)
DAY_CAP = 21 * 60 + 30         # appended/station attraction must end by 21:30
ATT_ONE = 15.0                 # avg transit minutes at which ATT == 1
K_STATION = 16                 # station-phase candidate recall
MIN_STATION_WIN = 50           # smallest usable station-adjacent window
MAX_EVALS_SLOT = 4             # passes() calls per gap/morning slot attempt
MAX_EVALS_PLAN = 120           # passes() budget per plan (phases 1,3,4,5)
MAX_INSERTS = 12               # per-plan insert guard
MODE_PAIRS = (("metro", "metro"), ("walk", "walk"), ("taxi", "taxi"),
              ("walk", "metro"), ("metro", "walk"),
              ("walk", "taxi"), ("taxi", "walk"))


def gap_dwell(gap):
    for g, dw in TIERS:
        if gap >= g:
            return dw
    return None


def tcost(trs):
    return sum(float(t.get("cost", 0) or 0) for t in trs or [])


def db_hours(ag, name):
    df = ag.memory["attractions"]
    m = df[df["name"] == name]
    if len(m) == 0:
        return None, None
    r = m.iloc[0]
    return str(r["opentime"]), str(r["endtime"])


# ---------------------------------------------------------------- gap / shrink
def insert_at_gap(uid, ag, query, plan, di, i, ex_names, ex_types,
                  modes, free_first, free_only, att_cap):
    """One attraction into the idle window between acts[i] and acts[i+1] of
    day di (no cascade). Mutates plan on success. Returns (ok, n_evals)."""
    acts = plan["itinerary"][di]["activities"]
    city = query["target_city"]
    visited = set(actpos(x) for d in plan["itinerary"] for x in d["activities"])
    ppl = int(query.get("people_number", 1))
    x, y = acts[i], acts[i + 1]
    if y.get("type") in INTER:
        return False, 0
    posX, posY = actpos(x), actpos(y)
    xe, ys = x.get("end_time"), y.get("start_time")
    if not posX or not posY or not xe or not ys or hm(ys) < hm(xe):
        return False, 0
    lo, hi = hm(xe), hm(ys)
    dwell = gap_dwell(hi - lo)
    if dwell is None:
        return False, 0
    npass = 0
    for r in GA.nearest_attr(ag, query, posX, lo, hi, visited, ex_names,
                             ex_types, dwell, free_first=free_first,
                             free_only=free_only):
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
            if hm(tr2[-1]["end_time"]) > hm(ys):    # no cascade
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
            if att_cap is not None and GA.avg_transit(cand) > att_cap:
                continue
            npass += 1
            if passes(uid, cand):
                plan["itinerary"] = cand["itinerary"]
                return True, npass
            if npass >= MAX_EVALS_SLOT:
                return False, npass
    return False, npass


def phase_gap(uid, ag, query, plan, room, budget, kw):
    """Sweep all idle gaps (>=30 min) until no progress. (added, evals)."""
    added = evals = 0
    progress = True
    while room - added > 0 and progress and evals < budget:
        progress = False
        for di in range(len(plan["itinerary"])):
            acts = plan["itinerary"][di]["activities"]
            for i in range(len(acts) - 1):
                if evals >= budget:
                    return added, evals
                ok, np_ = insert_at_gap(uid, ag, query, plan, di, i, **kw)
                evals += np_
                if ok:
                    added += 1; progress = True
                    break
            if progress:
                break
    return added, evals


def phase_shrink(uid, ag, query, plan, room, budget, kw):
    """Shrink one attraction dwell (>=90 -> >=60) by 30 min, insert into the
    widened gap; adopt only on a validated insert. (added, evals)."""
    added = evals = 0
    progress = True
    while room - added > 0 and progress and evals < budget:
        progress = False
        for di in range(len(plan["itinerary"])):
            acts = plan["itinerary"][di]["activities"]
            for i in range(len(acts) - 1):
                if evals >= budget:
                    return added, evals
                x, y = acts[i], acts[i + 1]
                if x.get("type") != "attraction" or y.get("type") in INTER:
                    continue
                xs, xe, ys = x.get("start_time"), x.get("end_time"), y.get("start_time")
                if not xs or not xe or not ys:
                    continue
                dur = hm(xe) - hm(xs)
                gap = hm(ys) - hm(xe)
                if dur < MIN_KEEP + SHRINK or gap < 0:
                    continue
                if gap_dwell(gap + SHRINK) is None:
                    continue
                cand = copy.deepcopy(plan)
                cand["itinerary"][di]["activities"][i]["end_time"] = mh(hm(xe) - SHRINK)
                ok, np_ = insert_at_gap(uid, ag, query, cand, di, i, **kw)
                evals += np_
                if ok:
                    plan["itinerary"] = cand["itinerary"]
                    added += 1; progress = True
                    break
            if progress:
                break
    return added, evals


# -------------------------------------------------------------------- evening
def phase_eve(uid, ag, query, plan, room, ex_names, ex_types, budget):
    """enrich_endattr's per-day evening append run to FIXED POINT (the
    while-progress wrapper the standalone script lacks). Every append is
    individually passes()-gated inside enrich_endattr. (added, 0)."""
    added = 0
    progress = True
    while room - added > 0 and progress:
        progress = False
        for di in range(len(plan["itinerary"])):
            if room - added <= 0:
                break
            if EA.append_evening_attraction(ag, query, plan, di,
                                            ex_names, ex_types, budget):
                added += 1; progress = True
    return added, 0


# -------------------------------------------------------------------- station
def station_candidates(ag, query, anchor, dest, lo, hi, visited,
                       ex_names, ex_types):
    """On-the-way anchor->dest (detour-ranked); anchor-distance fallback."""
    df = ag.memory["attractions"]
    df = df[~df["name"].isin(visited)]
    if ex_names:
        df = df[~df["name"].isin(ex_names)]
    if ex_types and "type" in df.columns:
        df = df[~df["type"].astype(str).isin(ex_types)]
    try:
        dAB = ag.calculate_distance(query, anchor, dest) if dest else None
    except Exception:
        dAB = None
    rows = []
    for _, r in df.iterrows():
        name = r["name"]
        try:
            ot, et = hm(str(r.get("opentime"))), hm(str(r.get("endtime")))
        except Exception:
            continue
        overnight = et <= ot
        if not overnight and (et - 10 <= lo + 5 or ot >= hi - 30):
            continue                     # cannot host even dwell30
        try:
            d1 = ag.calculate_distance(query, anchor, name)
        except Exception:
            continue
        if d1 is None:
            continue
        key = d1
        if dAB is not None:
            try:
                d2 = ag.calculate_distance(query, name, dest)
                if d2 is not None:
                    key = d1 + d2 - dAB  # detour
            except Exception:
                pass
        rows.append((key, name, ot, et, overnight, float(r.get("price", 0) or 0)))
    rows.sort(key=lambda z: z[0])
    return rows[:K_STATION]


def station_insert(uid, ag, query, plan, di, i, hi_cap, dest, elastic,
                   base_att, ex_names, ex_types, slack, ic_slack, stats):
    """Insert one attraction between acts[i] and acts[i+1] (station-adjacent).
    dest: successor position (departure station for pre_dep). elastic:
    successor is accommodation and its start may shift later."""
    acts = plan["itinerary"][di]["activities"]
    a, b = acts[i], acts[i + 1]
    posA = actpos(a) if a.get("type") not in INTER else a.get("end")
    tA = a.get("end_time")
    if not posA or not tA or not dest:
        return False
    city = query["target_city"]
    ppl = int(query.get("people_number", 1))
    visited = set(actpos(x) for d in plan["itinerary"] for x in d["activities"])
    lo = hm(tA)
    removed = tcost(b.get("transports"))    # successor's old transports go away
    for _, name, ot, et, overnight, price in station_candidates(
            ag, query, posA, dest, lo, hi_cap, visited, ex_names, ex_types):
        if slack is not None and price * ppl > max(0.0, slack):
            continue
        for m1, m2 in MODE_PAIRS:
            tr1 = ag.collect_innercity_transport(city, posA, name, tA, m1)
            if not isinstance(tr1, list) or tr1 == []:
                continue
            arr = hm(tr1[-1]["end_time"])
            start_m = max(arr, ot) if not overnight else arr
            close = hi_cap if overnight else min(hi_cap, et)
            dwell = next((dw for dw in (60, 45, 30) if start_m + dw <= close), None)
            if dwell is None:
                continue
            end_m = start_m + dwell
            start, end = mh(start_m), mh(end_m)
            tr2 = ag.collect_innercity_transport(city, name, dest, end, m2)
            if not isinstance(tr2, list) or (tr2 == [] and name != dest):
                continue
            arr2 = hm(tr2[-1]["end_time"]) if tr2 else end_m
            if elastic:
                if arr2 >= 24 * 60 or end_m > DAY_CAP:
                    continue
            else:
                if arr2 > hi_cap:
                    continue
            if ic_slack is not None:
                if tcost(tr1) + tcost(tr2) - removed > ic_slack:
                    continue                 # ic-cost delta pre-check
            X = {"position": name, "type": "attraction", "price": price,
                 "cost": price * ppl, "tickets": ppl,
                 "start_time": start, "end_time": end, "transports": tr1}
            newb = copy.deepcopy(b)
            newb["transports"] = tr2
            if elastic and b.get("type") == "accommodation":
                newb["start_time"] = mh(arr2)
            cand = copy.deepcopy(plan)
            cand["itinerary"][di]["activities"] = acts[:i + 1] + [X, newb] + acts[i + 2:]
            try:
                ag._repair_itinerary_times(cand["itinerary"])
            except Exception:
                continue
            if base_att <= ATT_ONE and GA.avg_transit(cand) > ATT_ONE:
                continue
            stats["evals"] += 1
            if passes(uid, cand):
                plan["itinerary"] = cand["itinerary"]
                return True
            if stats["evals"] >= stats["budget"]:
                return False
    return False


def phase_station(uid, ag, query, plan, room, budget, base_att,
                  ex_names, ex_types, bud, icap):
    """Pre-departure and post-arrival windows around intercity legs."""
    added = 0
    stats = {"evals": 0, "budget": budget}
    progress = True
    while room - added > 0 and progress and stats["evals"] < stats["budget"]:
        progress = False
        for di, d in enumerate(plan["itinerary"]):
            acts = d["activities"]
            for j, a in enumerate(acts):
                if a.get("type") not in INTER:
                    continue
                slack = (bud - GA.plan_cost(plan)) if bud is not None else None
                ic_slack = (icap - GA.ic_cost(plan)) if icap is not None else None
                # pre_dep: between acts[j-1] and the intercity leg; the tr2
                # destination is the DEPARTURE station a['start'] (actpos on an
                # intercity act returns the wrong-city arrival station).
                if j > 0 and acts[j - 1].get("position") and acts[j - 1].get("end_time"):
                    w = hm(a["start_time"]) - hm(acts[j - 1]["end_time"])
                    if w >= MIN_STATION_WIN and station_insert(
                            uid, ag, query, plan, di, j - 1, hm(a["start_time"]),
                            a.get("start"), False, base_att, ex_names, ex_types,
                            slack, ic_slack, stats):
                        added += 1; progress = True
                        break
                # post_arr: after the ic leg, chaining past attractions already
                # inserted there ([ic, X1, X2, ..., acc] keeps growing).
                k = j
                while k + 1 < len(acts) and acts[k + 1].get("type") == "attraction":
                    k += 1
                if k + 1 < len(acts):
                    anchor = acts[k]
                    tail = [x.get("type") for x in acts[k + 1:]]
                    elastic = all(t == "accommodation" for t in tail)
                    nxt = acts[k + 1]
                    hi_cap = DAY_CAP if elastic else (
                        hm(nxt["start_time"]) if nxt.get("start_time") else None)
                    if hi_cap is None:
                        continue
                    ae = anchor.get("end_time")
                    w = (hi_cap - hm(ae)) if ae else 0
                    if nxt.get("type") in INTER:
                        dest = nxt.get("start")     # departure station, this city
                    else:
                        dest = nxt.get("position") or actpos(nxt)
                    if w >= MIN_STATION_WIN and dest and station_insert(
                            uid, ag, query, plan, di, k, hi_cap, dest, elastic,
                            base_att, ex_names, ex_types, slack, ic_slack, stats):
                        added += 1; progress = True
                        break
            if progress:
                break
    return added, stats["evals"]


# -------------------------------------------------------------------- morning
def morning_slot(uid, ag, query, plan, di, i, ex_names, ex_types,
                 modes, free_first, free_only, att_cap):
    """Post-breakfast insert with dwell fallback and bounded SHIFT-Y.
    Returns (validated cand plan or None, n_evals)."""
    acts = plan["itinerary"][di]["activities"]
    x, y = acts[i], acts[i + 1]
    posX, posY = actpos(x), actpos(y)
    xe, ys = x.get("end_time"), y.get("start_time")
    lo, hi = hm(xe), hm(ys)
    gap = hi - lo
    city = query["target_city"]; ppl = int(query.get("people_number", 1))
    visited = set(actpos(a) for d in plan["itinerary"] for a in d["activities"])
    y_dur = hm(y["end_time"]) - hm(y["start_time"])
    y_open, y_close = (db_hours(ag, posY) if y.get("type") == "attraction"
                       else (None, None))
    z = acts[i + 2] if i + 2 < len(acts) else None
    npass = 0
    for dwell in (60, 45, 30):
        if gap < 60 or dwell > gap:
            continue
        cands = GA.nearest_attr(ag, query, posX, lo, hi + MAX_SHIFT, visited,
                                ex_names, ex_types, dwell,
                                free_first=free_first, free_only=free_only)
        for r in cands:
            name = r["name"]; ot, et = str(r.get("opentime")), str(r.get("endtime"))
            for mode in modes:
                tr1 = ag.collect_innercity_transport(city, posX, name, xe, mode)
                if not isinstance(tr1, list) or tr1 == []:
                    continue
                arr = tr1[-1]["end_time"]; start = arr
                if not (et < ot) and hm(arr) < hm(ot):
                    start = ot
                if not (et < ot) and hm(start) + dwell > hm(et):
                    continue
                end = mh(hm(start) + dwell)
                tr2 = ag.collect_innercity_transport(city, name, posY, end, mode)
                if not isinstance(tr2, list) or tr2 == []:
                    continue
                arr2 = hm(tr2[-1]["end_time"])
                newY = copy.deepcopy(y); newY["transports"] = tr2
                newZ = None
                if arr2 > hm(ys):                     # SHIFT-Y branch
                    if arr2 - hm(ys) > MAX_SHIFT:
                        continue
                    ny_start, ny_end = arr2, arr2 + y_dur
                    if y_close is not None and not (y_close < y_open) \
                       and ny_end > hm(y_close):
                        continue                      # Y would close
                    newY["start_time"] = mh(ny_start)
                    newY["end_time"] = mh(ny_end)
                    if z is not None:
                        ztr = z.get("transports") or []
                        if ztr:
                            zmode = ztr[0]["mode"] if len(ztr) == 1 else "metro"
                            zmode = zmode if zmode in modes else modes[0]
                            tr3 = ag.collect_innercity_transport(
                                city, posY, actpos(z), newY["end_time"], zmode)
                            if not isinstance(tr3, list) or tr3 == []:
                                continue
                            arr3 = hm(tr3[-1]["end_time"])
                        else:
                            tr3, arr3 = ztr, ny_end
                        if z.get("type") != "accommodation" and arr3 > hm(z["start_time"]):
                            continue                  # no deeper cascade
                        if arr3 >= 24 * 60:
                            continue
                        newZ = copy.deepcopy(z); newZ["transports"] = tr3
                price = float(r.get("price", 0) or 0)
                X = {"position": name, "type": "attraction", "price": price,
                     "cost": price * ppl, "tickets": ppl,
                     "start_time": start, "end_time": end, "transports": tr1}
                cand = copy.deepcopy(plan)
                tail = acts[i + 3:] if newZ is not None else acts[i + 2:]
                mid = [X, newY] + ([newZ] if newZ is not None else [])
                cand["itinerary"][di]["activities"] = acts[:i + 1] + mid + tail
                try:
                    ag._repair_itinerary_times(cand["itinerary"])
                except Exception:
                    continue
                if att_cap is not None and GA.avg_transit(cand) > att_cap:
                    continue
                npass += 1
                if passes(uid, cand):
                    return cand, npass
                if npass >= MAX_EVALS_SLOT:
                    return None, npass
    return None, npass


def phase_morning(uid, ag, query, plan, room, budget, kw):
    """Sweep breakfast-left slots (gap >= 60 after the LAST breakfast of the
    morning stack) until no progress. (added, evals)."""
    added = evals = 0
    progress = True
    while room - added > 0 and progress and evals < budget:
        progress = False
        for di in range(len(plan["itinerary"])):
            acts = plan["itinerary"][di]["activities"]
            for i in range(len(acts) - 1):
                if evals >= budget:
                    return added, evals
                x, y = acts[i], acts[i + 1]
                if x.get("type") != "breakfast" or y.get("type") in INTER \
                        or y.get("type") == "breakfast":
                    continue              # only the LAST breakfast of the stack
                if not actpos(x) or not actpos(y):
                    continue
                xe, ys = x.get("end_time"), y.get("start_time")
                if not xe or not ys or hm(ys) - hm(xe) < 60:
                    continue
                cand, np_ = morning_slot(uid, ag, query, plan, di, i, **kw)
                evals += np_
                if cand is not None:
                    plan["itinerary"] = cand["itinerary"]
                    added += 1; progress = True
                    break
            if progress:
                break
    return added, evals


# ---------------------------------------------------------------------- driver
def enrich_plan(uid, ag, query, plan):
    days = len(plan["itinerary"])
    shortfall = min(4 * days - GA.attr_count(plan), MAX_INSERTS)
    ex_names, ex_types = GA.excl_sets(query)
    bud = GA.binding_budget(query)
    slack = (bud - GA.plan_cost(plan)) if bud is not None else None
    icap = GA.binding_ic_cap(query)
    ic_slack = (icap - GA.ic_cost(plan)) if icap is not None else None
    modes, free_first, free_only, tags = GA.ALL_MODES, False, False, []
    if slack is not None and slack < GA.MIN_BUDGET_SLACK:
        modes, free_only = ("walk",), True
        tags.append(f"budget-tight({slack:.0f})->free+walk")
    elif slack is not None and slack <= GA.FREE_FIRST_SLACK:
        free_first = True
        tags.append(f"free-first({slack:.0f})")
    if ic_slack is not None and ic_slack < GA.IC_WALK_SLACK:
        modes, free_only = ("walk",), True
        tags.append(f"ic-tight({ic_slack:.0f})->free+walk")
    base_att = GA.avg_transit(plan)
    att_cap = GA.ATT_GUARD_AVG if base_att <= GA.ATT_GUARD_AVG else None
    kw = dict(ex_names=ex_names, ex_types=ex_types, modes=modes,
              free_first=free_first, free_only=free_only, att_cap=att_cap)

    added = evals = 0
    parts = {}
    for pname in ("gap", "eve", "station", "morning", "shrink"):
        room = shortfall - added
        if room <= 0 or evals >= MAX_EVALS_PLAN:
            break
        left = MAX_EVALS_PLAN - evals
        if pname == "gap":
            a, e = phase_gap(uid, ag, query, plan, room, left, kw)
        elif pname == "eve":
            a, e = phase_eve(uid, ag, query, plan, room, ex_names, ex_types, bud)
        elif pname == "station":
            a, e = phase_station(uid, ag, query, plan, room, min(left, 40),
                                 base_att, ex_names, ex_types, bud, icap)
        elif pname == "morning":
            a, e = phase_morning(uid, ag, query, plan, room, left, kw)
        else:
            a, e = phase_shrink(uid, ag, query, plan, room, left, kw)
        added += a; evals += e
        if a:
            parts[pname] = a
    return added, evals, parts, tags, att_cap


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

    # cheap prefilter: shard/limit over DAV-short plans ONLY (balanced shards)
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
        q = qd[uid]
        if not passes(uid, plan):
            continue
        ag = agent_for(q["target_city"]); ag.query = q
        d0, a0 = GA.dav(plan), GA.avg_transit(plan)
        cand = copy.deepcopy(plan)
        added, evals, parts, tags, att_cap = enrich_plan(uid, ag, q, cand)
        note = (" [" + ",".join(tags) + "]") if tags else ""
        pstr = ",".join(f"{k}+{v}" for k, v in parts.items())
        if added > 0:
            tried += 1
            d1, a1 = GA.dav(cand), GA.avg_transit(cand)
            guard_ok = att_cap is None or a1 <= att_cap
            if d1 > d0 and guard_ok and passes(uid, cand):
                kept += 1; tot += added
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{added}attr ({pstr}; evals={evals}) | "
                      f"DAV {d0:.3f}->{d1:.3f} ATTavg {a0:.1f}->{a1:.1f}{note}",
                      flush=True)
            else:
                why = "att-guard" if not guard_ok else ("delta" if d1 <= d0 else "final-pass")
                print(f"{uid}: rejected +{added}attr ({why}; {pstr}; evals={evals}){note}",
                      flush=True)
        else:
            print(f"{uid}: +0 (evals={evals}){note}", flush=True)
    print(f"\ntried {tried}, kept {kept}, attractions {tot}, apply={apply}")

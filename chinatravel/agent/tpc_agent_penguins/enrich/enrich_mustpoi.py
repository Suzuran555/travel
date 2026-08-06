"""Required-POI repair pass (hard-logic recovery) for the Phase-2 evaluator.

Parses required-POI targets from the GENERATED hard_logic_py (ctx.qd -- never
oracle) and applies gated fixers:

  A hotel-window        : shift accommodation start later into its window
  B meal-at-hotel       : insert a zero-cost hotel breakfast inside the window
  C named-restaurant    : relocate an existing meal (DB price, goto legs)
  D attraction-window   : insert the attraction in-window before hotel return
  E cuisine / attrtype  : set-membership variants of C / D

Every candidate is adopted only if the GENERATED-hard satisfied-count strictly
improves and commonsense does not degrade. Measured on the DashScope live-NL
familiar-100: FPR 78 -> 84, C-LPR 95.5 -> 97.0, Overall 85.98 -> 89.03.
"""
import re
import copy
from copy import deepcopy

from . import ctx
from . import enrich_lateattr as LA
from chinatravel.symbol_verification.concept_func import normalize_poi_name as N


def hm(t):
    h, m = str(t).split(":")
    return int(h) * 60 + int(m)


def fmt(x):
    return "24:00" if x >= 1440 else f"{x//60:02d}:{x%60:02d}"


MEAL_WIN = {"breakfast": (360, 540), "lunch": (660, 840), "dinner": (1020, 1200)}

POS = r'activity_position\(activity\)==[\'"]([^\'"]+)[\'"]'
ST = r'activity_start_time\(activity\)>=[\'"]([0-9:]+)[\'"]'
ET = r'activity_end_time\(activity\)<=[\'"]([0-9:]+)[\'"]'

_REST = {}
_ATTR = {}


def _rest_db(city):
    if city not in _REST:
        from chinatravel.environment.tools.restaurants.apis import Restaurants
        _REST[city] = Restaurants(lang="en").select(city, key="name", func=lambda x: True)
    return _REST[city]


def _attr_db(city):
    if city not in _ATTR:
        from chinatravel.environment.tools.attractions.apis import Attractions
        _ATTR[city] = Attractions(lang="en").select(city, key="name", func=lambda x: True)
    return _ATTR[city]


def parse_targets(dsl_list):
    tg = []
    for c in dsl_list:
        c1 = re.sub(r"\s+", " ", c)
        if "result=False" not in c1: continue
        pm = re.search(POS, c1)
        if not pm: continue
        poi = pm.group(1)
        t1 = re.search(ST, c1); t2 = re.search(ET, c1)
        lo = hm(t1.group(1)) if t1 else None
        hi = hm(t2.group(1)) if t2 else None
        if "=='accommodation'" in c1:
            tg.append({"kind": "hotel_window", "poi": poi, "lo": lo, "hi": hi})
        elif "'breakfast'" in c1:
            tg.append({"kind": "meal_at", "poi": poi, "lo": lo, "hi": hi})
        elif "=='attraction'" in c1:
            tg.append({"kind": "attr_window", "poi": poi, "lo": lo, "hi": hi})
    # set-membership requirements: ({...} <= xxx_set), no window
    for c in dsl_list:
        c1 = re.sub(r"\s+", " ", c)
        m = re.search(r"result=\(\{([^}]+)\}\s*<=\s*(attraction_name_set|restaurant_name_set|restaurant_type_set|attraction_type_set)\)", c1)
        if not m: continue
        items = re.findall(r"[\'\"]([^\'\"]+)[\'\"]", m.group(1))
        kindmap = {"attraction_name_set": "attr_window", "restaurant_name_set": "meal_at",
                   "restaurant_type_set": "cuisine", "attraction_type_set": "attrtype"}
        for it in items:
            tg.append({"kind": kindmap[m.group(2)], "poi": it, "lo": None, "hi": None})
    return tg

def fix_hotel_window(plan, poi, lo, hi):
    p = deepcopy(plan); done = False
    for day in p["itinerary"]:
        for a in day["activities"]:
            if a.get("type") == "accommodation" and N(a.get("position", "")) == N(poi):
                st, ed = hm(a["start_time"]), hm(a["end_time"])
                nlo = st if lo is None else max(st, lo)
                if hi is not None and ed > hi: continue
                if nlo < ed and nlo != st:
                    a["start_time"] = fmt(nlo); done = True
                elif lo is None or st >= lo:
                    done = True
    return p if done else None

def fix_meal_at_hotel(plan, poi, lo, hi):
    """Insert a meal at the (already-booked) hotel: first slot of a morning
    (breakfast) whose previous day ends at that hotel -> transports []."""
    p = deepcopy(plan)
    days = p["itinerary"]
    for di in range(1, len(days)):
        prev_acts = days[di-1]["activities"]
        if not prev_acts: continue
        last = prev_acts[-1]
        if last.get("type") != "accommodation" or N(last.get("position","")) != N(poi):
            continue
        acts = days[di]["activities"]
        if any(a.get("type") == "breakfast" for a in acts): continue
        first_st = hm(acts[0]["start_time"]) if acts else 1440
        # consider the first activity's own transports: meal must end before they depart
        trs0 = (acts[0].get("transports") or []) if acts else []
        dep0 = hm(trs0[0]["start_time"]) if trs0 else first_st
        w_lo = max(MEAL_WIN["breakfast"][0], lo if lo is not None else 0)
        w_hi = min(MEAL_WIN["breakfast"][1], hi if hi is not None else 1440, dep0)
        if w_hi - w_lo < 20: continue
        st = w_lo; ed = min(st + 30, w_hi)
        meal = {"position": poi, "type": "breakfast", "price": 0, "cost": 0,
                "start_time": fmt(st), "end_time": fmt(ed), "transports": []}
        days[di]["activities"] = [meal] + acts
        return p
    return None

def _win_type(lo, hi):
    cands = []
    for ty, (wl, wh) in MEAL_WIN.items():
        l = max(wl, lo if lo is not None else 0); h = min(wh, hi if hi is not None else 1440)
        if h - l >= 25: cands.append((ty, l, h))
    return cands

def fix_meal_at_restaurant(plan, poi, lo, hi, city, ppl):
    """Relocate an existing meal to the required restaurant within window."""
    db = _rest_db(city); row = db[db["name"] == poi]
    if row.empty: return None
    price = float(row.iloc[0]["price"]); ot = hm(str(row.iloc[0]["opentime"])); et = hm(str(row.iloc[0]["endtime"]))
    for ty, wl, wh in _win_type(lo, hi):
        wl = max(wl, ot); wh = min(wh, et)
        if wh - wl < 25: continue
        p = deepcopy(plan)
        for day in p["itinerary"]:
            acts = day["activities"]
            for j, a in enumerate(acts):
                if a.get("type") != ty: continue
                prev = acts[j-1] if j > 0 else None
                nxt = acts[j+1] if j+1 < len(acts) else None
                prev_pos = prev.get("position") or prev.get("end") if prev else None
                prev_end = hm(prev["end_time"]) if prev and prev.get("end_time") else None
                if not prev_pos or prev_end is None: continue
                tr1 = None
                for mode in ("metro", "walk", "taxi"):
                    probe = LA.goto(city, prev_pos, poi, "09:00", mode, ppl)
                    if not probe: continue
                    d1 = hm(probe[-1]["end_time"]) - hm(probe[0]["start_time"])
                    st = max(wl, prev_end + d1)
                    if st + 30 > wh: continue
                    dep = st - d1
                    if dep < prev_end: continue
                    sh = dep - hm(probe[0]["start_time"])
                    tr1 = [dict(l, start_time=fmt(hm(l["start_time"])+sh), end_time=fmt(hm(l["end_time"])+sh)) for l in probe]
                    break
                if tr1 is None: continue
                st = hm(tr1[-1]["end_time"]); ed = st + 30
                # leg back to next activity
                tr2 = None; nxt_pos = None
                if nxt is not None:
                    nxt_pos = nxt.get("position") or (nxt.get("start") if nxt.get("start") else None)
                if nxt_pos and N(nxt_pos) != N(poi):
                    for mode in ("metro", "walk", "taxi"):
                        probe = LA.goto(city, poi, nxt_pos, "09:00", mode, ppl)
                        if not probe: continue
                        d2 = hm(probe[-1]["end_time"]) - hm(probe[0]["start_time"])
                        arr = ed + d2
                        nst = hm(nxt["start_time"]) if nxt.get("start_time") else None
                        if nst is None: continue
                        if arr <= nst:
                            sh = ed - hm(probe[0]["start_time"])
                            tr2 = [dict(l, start_time=fmt(hm(l["start_time"])+sh), end_time=fmt(hm(l["end_time"])+sh)) for l in probe]
                            break
                        if nxt.get("type") == "accommodation" and arr < 1440:
                            sh = ed - hm(probe[0]["start_time"])
                            tr2 = [dict(l, start_time=fmt(hm(l["start_time"])+sh), end_time=fmt(hm(l["end_time"])+sh)) for l in probe]
                            nxt = dict(nxt); nxt["start_time"] = fmt(arr)
                            break
                    if tr2 is None: continue
                na = dict(a); na.update(position=poi, price=price, cost=price*ppl,
                                        start_time=fmt(st), end_time=fmt(ed), transports=tr1)
                nacts = list(acts); nacts[j] = na
                if tr2 is not None and j+1 < len(nacts):
                    nn = dict(nacts[j+1]); nn["transports"] = tr2
                    if nn.get("type") == "accommodation" and nxt is not None and "start_time" in nxt:
                        nn["start_time"] = nxt["start_time"]
                    nacts[j+1] = nn
                day["activities"] = nacts
                return p
    return None

def fix_attr_window(plan, poi, lo, hi, city, ppl):
    """Insert the required attraction in-window before an evening hotel return."""
    db = _attr_db(city); row = db[db["name"] == poi]
    if row.empty: return None
    price = float(row.iloc[0].get("price", 0) or 0)
    ot = hm(str(row.iloc[0]["opentime"])); et = hm(str(row.iloc[0]["endtime"]))
    wl = max(lo if lo is not None else 0, ot); wh = min(hi if hi is not None else 1440, et)
    if wh - wl < 30: return None
    # already present? then just try shifting its time is complex — only insert if absent
    for day in plan["itinerary"]:
        for a in day["activities"]:
            if a.get("type") == "attraction" and N(a.get("position","")) == N(poi):
                return None
    p = deepcopy(plan)
    for day in p["itinerary"]:
        acts = day["activities"]
        if not acts: continue
        H = acts[-1]
        if H.get("type") != "accommodation": continue
        j = len(acts) - 1
        prev = acts[j-1] if j > 0 else None
        if prev is None: continue
        prev_pos = prev.get("position") or prev.get("end")
        prev_end = hm(prev["end_time"]) if prev.get("end_time") else None
        if not prev_pos or prev_end is None: continue
        for mode in ("metro", "walk", "taxi"):
            p1 = LA.goto(city, prev_pos, poi, "09:00", mode, ppl)
            if not p1: continue
            d1 = hm(p1[-1]["end_time"]) - hm(p1[0]["start_time"])
            st = max(wl, prev_end + d1)
            ed = st + 30
            if ed > wh: continue
            for m2 in ("metro", "walk", "taxi"):
                p2 = LA.goto(city, poi, H.get("position",""), "09:00", m2, ppl)
                if not p2: continue
                d2 = hm(p2[-1]["end_time"]) - hm(p2[0]["start_time"])
                arr = ed + d2
                if arr >= 1440: continue
                sh1 = (st - d1) - hm(p1[0]["start_time"])
                tr1 = [dict(l, start_time=fmt(hm(l["start_time"])+sh1), end_time=fmt(hm(l["end_time"])+sh1)) for l in p1]
                sh2 = ed - hm(p2[0]["start_time"])
                tr2 = [dict(l, start_time=fmt(hm(l["start_time"])+sh2), end_time=fmt(hm(l["end_time"])+sh2)) for l in p2]
                newact = {"position": poi, "type": "attraction", "price": price,
                          "cost": price*ppl, "tickets": ppl, "start_time": fmt(st),
                          "end_time": fmt(ed), "transports": tr1}
                nH = dict(H); nH["transports"] = tr2; nH["start_time"] = fmt(arr)
                day["activities"] = acts[:j] + [newact, nH]
                return p
    return None

def fix_cuisine(plan, cuisine, city, ppl):
    """Relocate a meal to any restaurant of the required cuisine (nearest fit)."""
    db = _rest_db(city)
    rows = db[db["cuisine"] == cuisine]
    if rows.empty: return None
    for _, r in rows.iterrows():
        cand = fix_meal_at_restaurant(plan, r["name"], None, None, city, ppl)
        if cand is not None: return cand
    return None

def fix_attrtype(plan, atype, city, ppl):
    """Insert any attraction of the required type into an evening slot."""
    db = _attr_db(city)
    col = "type" if "type" in db.columns else None
    if col is None: return None
    rows = db[db[col] == atype]
    if rows.empty: return None
    for _, r in rows.iterrows():
        cand = fix_attr_window(plan, r["name"], None, None, city, ppl)
        if cand is not None: return cand
    return None


def hard_count(uid, plan):
    """Satisfied-count against the GENERATED constraints (ctx.qd)."""
    from chinatravel.evaluation.hard_constraint import evaluate_constraints_py
    res = evaluate_constraints_py(ctx.qd[uid].get("hard_logic_py") or [], plan, verbose=False)
    return sum(1 for r in res if r is True)


def commonsense_ok(uid, plan):
    from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
    _, _, _, cp = evaluate_commonsense_constraints([uid], ctx.qd, {uid: plan}, verbose=False, lang=ctx._lang)
    return uid in cp


def repair(uid, plan):
    """Apply all fixers, each gated on hard-count improvement + commonsense."""
    q = ctx.qd[uid]
    city = q["target_city"]
    ppl = int(q.get("people_number", 1) or 1)
    targets = parse_targets(q.get("hard_logic_py") or [])
    if not targets:
        return plan
    cur = plan
    h0 = hard_count(uid, cur)
    cs0 = commonsense_ok(uid, cur)
    for t in targets:
        cand = None
        if t["kind"] == "hotel_window":
            cand = fix_hotel_window(cur, t["poi"], t["lo"], t["hi"])
        elif t["kind"] == "meal_at":
            cand = fix_meal_at_hotel(cur, t["poi"], t["lo"], t["hi"])
            if cand is None:
                cand = fix_meal_at_restaurant(cur, t["poi"], t["lo"], t["hi"], city, ppl)
        elif t["kind"] == "attr_window":
            cand = fix_attr_window(cur, t["poi"], t["lo"], t["hi"], city, ppl)
        elif t["kind"] == "cuisine":
            cand = fix_cuisine(cur, t["poi"], city, ppl)
        elif t["kind"] == "attrtype":
            cand = fix_attrtype(cur, t["poi"], city, ppl)
        if cand is None:
            continue
        if cs0 and not commonsense_ok(uid, cand):
            continue
        h1 = hard_count(uid, cand)
        if h1 > h0:
            cur, h0 = cand, h1
            cs0 = commonsense_ok(uid, cur)
    return cur

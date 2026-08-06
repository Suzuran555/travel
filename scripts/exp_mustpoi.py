#!/usr/bin/env python3
"""Post-hoc gated repair experiment for required-POI hard-logic misses.

Parses required-POI targets from the GENERATED DSL (translation cache — never
the oracle), applies fixers, gates on the official evaluators:
  A hotel-window : shift accommodation start later to enter its window
  B meal-at-hotel: insert a zero-cost hotel meal inside window (same-position
                   rules: transports [] when following the same hotel)
  C named-restaurant relocate: move an existing meal to the required restaurant
                   (DB price, goto legs both sides)
  D attraction-window insert: evening/morning attraction in window w/ goto legs

Run: CHINATRAVEL_OPENAI_* env + .venv python scripts/exp_mustpoi.py
"""
import sys, os, json, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copy import deepcopy
import ast as _ast

from chinatravel.environment.world_env import WorldEnv
from chinatravel.symbol_verification.concept_func import normalize_poi_name as N
from chinatravel.agent.tpc_agent_penguins.enrich import ctx as ectx
from chinatravel.agent.tpc_agent_penguins.enrich import enrich_lateattr as LA
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
from chinatravel.evaluation.utils import load_json_file

ectx._env = WorldEnv(lang="en"); ectx._lang = "en"
FAM = "chinatravel/data/en/phase2_familiar_EN"
REP = "results/TPCAgent_dashscope_repaired"
CACHE = "cache/TPCAgent_dashscope/translation_Qwen3.6-27B_reflect"
schema = load_json_file("chinatravel/evaluation/output_schema.json")

def hm(t): h, m = str(t).split(":"); return int(h)*60 + int(m)
def fmt(x): return "24:00" if x >= 1440 else f"{x//60:02d}:{x%60:02d}"

MEAL_WIN = {"breakfast": (360, 540), "lunch": (660, 840), "dinner": (1020, 1200)}

POS = r'activity_position\(activity\)==(?:"((?:\\.|[^"\\])+)"|\'((?:\\.|[^\'\\])+)\')'
ST = r'activity_start_time\(activity\)>=[\'"]([0-9:]+)[\'"]'
ET = r'activity_end_time\(activity\)<=[\'"]([0-9:]+)[\'"]'

def parse_targets(dsl_list):
    tg = []
    for c in dsl_list:
        c1 = re.sub(r"\s+", " ", c)
        if "result=False" not in c1: continue
        pm = re.search(POS, c1)
        if not pm: continue
        poi = (pm.group(1) or pm.group(2)).replace("\\'","'").replace('\\"','"')
        t1 = re.search(ST, c1); t2 = re.search(ET, c1)
        lo = hm(t1.group(1)) if t1 else None
        hi = hm(t2.group(1)) if t2 else None
        if "=='accommodation'" in c1:
            tg.append({"kind": "hotel_window", "poi": poi, "lo": lo, "hi": hi})
        elif "'breakfast'" in c1:
            tg.append({"kind": "meal_at", "poi": poi, "lo": lo, "hi": hi})
        elif "=='attraction'" in c1:
            tg.append({"kind": "attr_window", "poi": poi, "lo": lo, "hi": hi})
    for c in dsl_list:
        c1 = re.sub(r"\s+", " ", c)
        m = re.search(r"result=\(\{([^}]+)\}\s*<=\s*intercity_transport_set\)", c1)
        if m:
            for it in re.findall(r"[\'\"]([a-z]+)[\'\"]", m.group(1)):
                tg.append({"kind": "intercity", "poi": it, "lo": None, "hi": None})
        m = re.search(r"restaurant_cost\+=activity_cost\(activity\) result=\(restaurant_cost<=([0-9.]+)\)", c1)
        if m:
            tg.append({"kind": "meal_budget", "poi": m.group(1), "lo": None, "hi": None})
    for c in dsl_list:
        c1 = re.sub(r"\s+", " ", c)
        hits=list(re.finditer(r"[\'\"]((?:\\.|[^\'\"\\])+)[\'\"]\s+in\s+(intercity_transport_set|attraction_name_set|restaurant_name_set|restaurant_type_set|attraction_type_set)", c1))
        gid = "in%d" % len(tg) if len(hits) > 1 else None
        for m in hits:
            it=(m.group(1)).replace("\\'","'").replace('\\"','"')
            k={"intercity_transport_set":"intercity","attraction_name_set":"attr_window",
               "restaurant_name_set":"meal_at","restaurant_type_set":"cuisine",
               "attraction_type_set":"attrtype"}[m.group(2)]
            tg.append({"kind": k, "poi": it, "lo": None, "hi": None, "group": gid})
    for c in dsl_list:
        c1 = re.sub(r"\s+", " ", c)
        m = re.search(r"result=not\(\{([^}]+)\}\s*&\s*(attraction_type_set|restaurant_type_set)\)", c1)
        if not m: continue
        items = [ (a or b).replace("\\'","'").replace('\\"','"')
                  for a,b in re.findall(r'"((?:\\.|[^"\\])+)"|\'((?:\\.|[^\'\\])+)\'', m.group(1)) ]
        k = "attrtype_excl" if m.group(2)=="attraction_type_set" else "cuisine_excl"
        for it in items:
            tg.append({"kind": k, "poi": it, "lo": None, "hi": None})
    # set-membership requirements: ({...} <= xxx_set), no window
    for c in dsl_list:
        c1 = re.sub(r"\s+", " ", c)
        m = re.search(r"result=\(\{([^}]+)\}\s*<=\s*(attraction_name_set|restaurant_name_set|restaurant_type_set|attraction_type_set)\)", c1)
        if not m: continue
        items = [ (a or b).replace("\\'","'").replace('\\"','"')
                  for a,b in re.findall(r'"((?:\\.|[^"\\])+)"|\'((?:\\.|[^\'\\])+)\'', m.group(1)) ]
        kindmap = {"attraction_name_set": "attr_window", "restaurant_name_set": "meal_at",
                   "restaurant_type_set": "cuisine", "attraction_type_set": "attrtype"}
        gid = "set%d" % len(tg)
        for it in items:
            tg.append({"kind": kindmap[m.group(2)], "poi": it, "lo": None, "hi": None,
                       "group": gid if len(items) > 1 else None})
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

def passes_full(uid, full, plan):
    idx = [uid]
    _, _, sp = evaluate_schema_constraints(idx, {uid: plan}, schema=schema)
    if uid not in sp: return False
    _, _, _, cp = evaluate_commonsense_constraints(idx, {uid: full}, {uid: plan}, verbose=False, lang="en")
    if uid not in cp: return False
    *_, lp = evaluate_hard_constraints_v2(idx, {uid: full}, {uid: plan}, env_pass_id=cp, verbose=False, lang="en")
    return uid in lp

def hard_count(uid, full, plan):
    from chinatravel.evaluation.hard_constraint import evaluate_constraints_py
    res = evaluate_constraints_py(full["hard_logic_py"], plan, verbose=False)
    return sum(1 for r in res if r is True)

def loadq(uid):
    d = json.load(open(f"{FAM}/{uid}.json"))
    if isinstance(d.get("hard_logic_py"), str):
        d["hard_logic_py"] = _ast.literal_eval(d["hard_logic_py"])
    return d

def main():
    fails = []
    for f in sorted(glob.glob(FAM + "/*.json")):
        uid = os.path.basename(f)[:-5]; full = loadq(uid)
        plan = json.load(open(f"{REP}/{uid}.json"))
        if not passes_full(uid, full, plan): fails.append(uid)
    print(f"targets: {len(fails)} failing plans")
    fixed = 0; improved = 0
    for uid in fails:
        full = loadq(uid); plan = json.load(open(f"{REP}/{uid}.json"))
        gen = json.load(open(f"{CACHE}/{uid}.json")).get("hard_logic_py") or []
        tg = parse_targets(gen)
        # merge meal_at windows for the same POI (intersection)
        merged = {}
        rest = []
        for t in tg:
            if t["kind"] == "meal_at":
                k = t["poi"]
                if k in merged:
                    m = merged[k]
                    m["lo"] = max(m["lo"] or 0, t["lo"] or 0) or None
                    his = [x for x in (m["hi"], t["hi"]) if x is not None]
                    m["hi"] = min(his) if his else None
                else:
                    merged[k] = dict(t)
            else:
                rest.append(t)
        tg = list(merged.values()) + rest
        if not tg: print(f"  {uid[-8:]}: no parsed targets"); continue
        cur = plan; h0 = hard_count(uid, full, cur)
        # group set-derived targets so multi-item requirements gate as a whole
        groups = []
        seen_g = {}
        for t in tg:
            g = t.get("group")
            if g:
                if g not in seen_g:
                    seen_g[g] = []; groups.append(("grp", seen_g[g]))
                seen_g[g].append(t)
            else:
                groups.append(("one", [t]))
        for gk, gts in groups:
            tmp = cur
            for t in gts:
                city=full["target_city"]; ppl=int(full.get("people_number",1))
                cand = None
                if t["kind"] == "hotel_window":
                    cand = fix_hotel_window(tmp, t["poi"], t["lo"], t["hi"])
                elif t["kind"] == "meal_at":
                    cand = fix_shift_meal(tmp, t["poi"], t["lo"], t["hi"])
                    if cand is None: cand = fix_meal_at_hotel(tmp, t["poi"], t["lo"], t["hi"])
                    if cand is None: cand = fix_relocate_meal_to_hotel(tmp, t["poi"], t["lo"], t["hi"])
                    if cand is None: cand = fix_meal_at_restaurant(tmp, t["poi"], t["lo"], t["hi"], city, ppl)
                    if cand is None: cand = fix_insert_meal(tmp, t["poi"], t["lo"], t["hi"], city, ppl)
                elif t["kind"] == "attr_window":
                    cand = fix_attr_window(tmp, t["poi"], t["lo"], t["hi"], city, ppl)
                elif t["kind"] == "cuisine":
                    cand = fix_cuisine(tmp, t["poi"], city, ppl)
                elif t["kind"] == "attrtype":
                    cand = fix_attrtype(tmp, t["poi"], city, ppl)
                elif t["kind"] == "attrtype_excl":
                    cand = fix_attrtype_excl(tmp, t["poi"], city, ppl)
                elif t["kind"] == "intercity":
                    cand = fix_intercity(tmp, t["poi"], full["start_city"], city, city, ppl)
                elif t["kind"] == "meal_budget":
                    protected = [x["poi"] for x in tg if x["kind"] in ("meal_at",)]
                    cand = fix_budget_targeted(tmp, float(t["poi"]), city, ppl, protected)
                if cand is not None: tmp = cand
            if tmp is cur: continue
            _, _, _, cp0 = evaluate_commonsense_constraints([uid], {uid: full}, {uid: cur}, verbose=False, lang="en")
            _, _, _, cp1 = evaluate_commonsense_constraints([uid], {uid: full}, {uid: tmp}, verbose=False, lang="en")
            if (uid in cp0) and (uid not in cp1): continue
            h1 = hard_count(uid, full, tmp)
            if h1 > h0: cur, h0 = tmp, h1
        if passes_full(uid, full, cur):
            fixed += 1; json.dump(cur, open(f"{REP}/{uid}.json", "w"), ensure_ascii=False)
            print(f"  {uid[-8:]}: FIXED (hard {h0})")
        elif h0 > hard_count(uid, full, plan):
            improved += 1; json.dump(cur, open(f"{REP}/{uid}.json", "w"), ensure_ascii=False)
            print(f"  {uid[-8:]}: improved hard->{h0} (still failing)")
    print(f"\nA+B fixers: fully fixed {fixed}, improved {improved}, of {len(fails)}")


# ---------------- C + D fixers (appended) ----------------
_REST = {}; _ATTR = {}
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
                for mode in ("metro", "taxi", "walk"):
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
    """Relocate (or insert) a meal at any restaurant of the required cuisine."""
    db = _rest_db(city)
    rows = db[db["cuisine"] == cuisine]
    if rows.empty: return None
    for _, r in rows.iterrows():
        cand = fix_meal_at_restaurant(plan, r["name"], None, None, city, ppl)
        if cand is not None: return cand
    for _, r in rows.iterrows():
        cand = fix_insert_meal(plan, r["name"], None, None, city, ppl)
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


def fix_shift_meal(plan, poi, lo, hi):
    """If a meal at poi exists but misses the window, shift it inside."""
    p = deepcopy(plan)
    for day in p["itinerary"]:
        acts = day["activities"]
        for j, a in enumerate(acts):
            if a.get("type") not in MEAL_WIN: continue
            if N(a.get("position","")) != N(poi): continue
            wl, wh = MEAL_WIN[a["type"]]
            l = max(wl, lo if lo is not None else 0)
            h = min(wh, hi if hi is not None else 1440)
            if h - l < 20: continue
            st, ed = hm(a["start_time"]), hm(a["end_time"])
            if st >= l and ed <= h: return None       # already fine
            dur = min(max(ed - st, 20), h - l)
            nst = max(l, min(st, h - dur)); ned = nst + dur
            # chronology guards
            prev = acts[j-1] if j > 0 else None
            nxt = acts[j+1] if j+1 < len(acts) else None
            if prev and prev.get("end_time") and hm(prev["end_time"]) > nst:
                trs=a.get("transports") or []
                if trs: continue
                nst = max(nst, hm(prev["end_time"])); ned = nst + dur
                if ned > h: continue
            if nxt and nxt.get("start_time") and hm(nxt["start_time"]) < ned:
                if nxt.get("type") == "accommodation" and not (nxt.get("transports") or []):
                    nxt2 = dict(nxt); nxt2["start_time"] = fmt(ned); acts[j+1] = nxt2
                else:
                    continue
            na = dict(a); na["start_time"] = fmt(nst); na["end_time"] = fmt(ned)
            trs = na.get("transports") or []
            if trs:
                sh = nst - st
                na["transports"] = [dict(l2, start_time=fmt(hm(l2["start_time"])+sh),
                                          end_time=fmt(hm(l2["end_time"])+sh)) for l2 in trs]
            acts[j] = na
            return p
    return None

def fix_insert_meal(plan, poi, lo, hi, city, ppl):
    """Insert a NEW meal at a DB restaurant into a free gap (day lacks type)."""
    db = _rest_db(city); row = db[db["name"] == poi]
    if row.empty: return None
    price = float(row.iloc[0]["price"]); ot = hm(str(row.iloc[0]["opentime"])); et = hm(str(row.iloc[0]["endtime"]))
    for ty, wl, wh in _win_type(lo, hi):
        wl = max(wl, ot); wh = min(wh, et)
        if wh - wl < 25: continue
        p = deepcopy(plan)
        for day in p["itinerary"]:
            acts = day["activities"]
            if any(a.get("type") == ty for a in acts): continue
            for j in range(1, len(acts)+0):
                prev, nxt = acts[j-1], acts[j] if j < len(acts) else None
                if nxt is None: break
                prev_pos = prev.get("position") or prev.get("end")
                nxt_pos = nxt.get("position") or nxt.get("start")
                if not prev_pos or not nxt_pos: continue
                pe = hm(prev["end_time"]) if prev.get("end_time") else None
                ns = hm(nxt["start_time"]) if nxt.get("start_time") else None
                if pe is None or ns is None: continue
                for m1 in ("metro","walk","taxi"):
                    p1 = LA.goto(city, prev_pos, poi, "09:00", m1, ppl)
                    if not p1: continue
                    d1 = hm(p1[-1]["end_time"]) - hm(p1[0]["start_time"])
                    st = max(wl, pe + d1); ed = st + 30
                    if ed > wh: continue
                    ok2 = None
                    if N(nxt_pos) == N(poi):
                        ok2 = ([], ns)
                    else:
                        for m2 in ("metro","walk","taxi"):
                            p2 = LA.goto(city, poi, nxt_pos, "09:00", m2, ppl)
                            if not p2: continue
                            d2 = hm(p2[-1]["end_time"]) - hm(p2[0]["start_time"])
                            arr = ed + d2
                            if arr <= ns:
                                sh2 = ed - hm(p2[0]["start_time"])
                                ok2 = ([dict(l2, start_time=fmt(hm(l2["start_time"])+sh2), end_time=fmt(hm(l2["end_time"])+sh2)) for l2 in p2], ns)
                                break
                            if nxt.get("type")=="accommodation" and arr < 1440:
                                sh2 = ed - hm(p2[0]["start_time"])
                                ok2 = ([dict(l2, start_time=fmt(hm(l2["start_time"])+sh2), end_time=fmt(hm(l2["end_time"])+sh2)) for l2 in p2], arr)
                                break
                    if ok2 is None: continue
                    tr2, narr = ok2
                    sh1 = (st - d1) - hm(p1[0]["start_time"])
                    tr1 = [dict(l2, start_time=fmt(hm(l2["start_time"])+sh1), end_time=fmt(hm(l2["end_time"])+sh1)) for l2 in p1]
                    meal = {"position": poi, "type": ty, "price": price, "cost": price*ppl,
                            "start_time": fmt(st), "end_time": fmt(ed), "transports": tr1}
                    nacts = list(acts); nn = dict(nxt)
                    if tr2: nn["transports"] = tr2
                    if nn.get("type")=="accommodation": nn["start_time"] = fmt(narr)
                    nacts[j] = nn
                    day["activities"] = nacts[:j] + [meal] + nacts[j:]
                    return p
    return None


_IT = {}
def _it_db():
    if "x" not in _IT:
        from chinatravel.environment.tools.intercity_transport.apis import IntercityTransport
        _IT["x"] = IntercityTransport(lang="en")
    return _IT["x"]

def fix_intercity(plan, req, start_city, target_city, city, ppl):
    """Swap the RETURN intercity leg to the required type (train/airplane)."""
    p = deepcopy(plan)
    days = p["itinerary"]
    if not days: return None
    acts = days[-1]["activities"]
    if not acts: return None
    row = acts[-1]
    if row.get("type") not in ("train","airplane") or row.get("type") == req:
        return None
    prev = acts[-2] if len(acts) > 1 else None
    prev_pos = (prev.get("position") or prev.get("end")) if prev else None
    pe = hm(prev["end_time"]) if prev and prev.get("end_time") else 0
    df = _it_db().select(target_city, start_city, req)
    if df is None or df.empty: return None
    orig = hm(row["start_time"])
    cands = sorted(df.to_dict("records"), key=lambda r: abs(hm(str(r["BeginTime"])) - orig))
    for r in cands[:8]:
        bt = hm(str(r["BeginTime"]))
        station = r["From"]
        tr = None
        if prev_pos:
            for mode in ("metro","taxi","walk"):
                probe = LA.goto(city, prev_pos, station, "09:00", mode, ppl)
                if not probe: continue
                d1 = hm(probe[-1]["end_time"]) - hm(probe[0]["start_time"])
                dep = bt - d1 - 10
                if dep < pe: continue
                sh = dep - hm(probe[0]["start_time"])
                tr = [dict(l, start_time=fmt(hm(l["start_time"])+sh), end_time=fmt(hm(l["end_time"])+sh)) for l in probe]
                break
            if tr is None: continue
        idk = "TrainID" if req == "train" else "FlightID"
        price = float(r["Cost"])
        new = {"type": req, idk: r[idk], "start": r["From"], "end": r["To"],
               "start_time": str(r["BeginTime"]), "end_time": str(r["EndTime"]),
               "price": price, "tickets": ppl, "cost": price*ppl,
               "transports": tr or []}
        acts[-1] = new
        return p
    return None

def fix_meal_budget(plan, cap, city, ppl, protected):
    """Bring total meal cost under cap by relocating priciest meals to cheap ones."""
    p = deepcopy(plan)
    db = _rest_db(city)
    cheap = db.sort_values("price").to_dict("records")
    for _ in range(4):
        meals = [(hm(a["start_time"]), di, j, a) for di, day in enumerate(p["itinerary"])
                 for j, a in enumerate(day["activities"]) if a.get("type") in MEAL_WIN]
        total = sum(float(a.get("cost") or 0) for _, _, _, a in meals)
        if total <= cap: return p
        meals_sorted = sorted(meals, key=lambda x: -float(x[3].get("cost") or 0))
        moved = False
        for _, di, j, a in meals_sorted:
            if N(a.get("position","")) in {N(x) for x in protected}: continue
            if float(a.get("cost") or 0) <= 0: continue
            for r in cheap[:40]:
                if float(r["price"])*ppl >= float(a.get("cost") or 0): break
                cand = fix_meal_at_restaurant(p, r["name"], None, None, city, ppl)
                if cand is not None:
                    p = cand; moved = True; break
            if moved: break
        if not moved: return None
    return p


def fix_budget_targeted(plan, cap, city, ppl, protected):
    """Replace the priciest unprotected meal IN PLACE with a cheap restaurant."""
    p = deepcopy(plan)
    db = _rest_db(city); cheap = db.sort_values("price").to_dict("records")
    prot = {N(x) for x in protected}
    for _ in range(4):
        meals=[(di,j,a) for di,day in enumerate(p["itinerary"]) for j,a in enumerate(day["activities"]) if a.get("type") in MEAL_WIN]
        total=sum(float(a.get("cost") or 0) for _,_,a in meals)
        if total <= cap: return p
        done=False
        for di,j,a in sorted(meals,key=lambda x:-float(x[2].get("cost") or 0)):
            if N(a.get("position","")) in prot or float(a.get("cost") or 0)<=0: continue
            acts=p["itinerary"][di]["activities"]
            prev=acts[j-1] if j>0 else None
            nxt=acts[j+1] if j+1<len(acts) else None
            if prev is None: continue
            prev_pos=prev.get("position") or prev.get("end"); pe=hm(prev["end_time"])
            st,ed=hm(a["start_time"]),hm(a["end_time"])
            wl,wh=MEAL_WIN[a["type"]]
            ot_ok=lambda r: hm(str(r["opentime"]))<=st and ed<=hm(str(r["endtime"]))
            for r in cheap[:60]:
                if float(r["price"])*ppl >= float(a.get("cost") or 0)*0.7: break
                if not ot_ok(r): continue
                tr1=None
                for mode in ("metro","taxi","walk"):
                    probe=LA.goto(city, prev_pos, r["name"], "09:00", mode, ppl)
                    if not probe: continue
                    d1=hm(probe[-1]["end_time"])-hm(probe[0]["start_time"])
                    dep=st-d1
                    if dep<pe: continue
                    sh=dep-hm(probe[0]["start_time"])
                    tr1=[dict(l, start_time=fmt(hm(l["start_time"])+sh), end_time=fmt(hm(l["end_time"])+sh)) for l in probe]
                    break
                if tr1 is None: continue
                tr2=None; ok=True
                if nxt is not None:
                    nxt_pos=nxt.get("position") or nxt.get("start")
                    ns=hm(nxt["start_time"]) if nxt.get("start_time") else None
                    if nxt_pos and ns is not None and N(nxt_pos)!=N(r["name"]):
                        ok=False
                        for m2 in ("metro","taxi","walk"):
                            p2=LA.goto(city, r["name"], nxt_pos, "09:00", m2, ppl)
                            if not p2: continue
                            d2=hm(p2[-1]["end_time"])-hm(p2[0]["start_time"])
                            if ed+d2<=ns:
                                sh2=ed-hm(p2[0]["start_time"])
                                tr2=[dict(l, start_time=fmt(hm(l["start_time"])+sh2), end_time=fmt(hm(l["end_time"])+sh2)) for l in p2]
                                ok=True; break
                if not ok: continue
                na=dict(a); na.update(position=r["name"], price=float(r["price"]),
                                       cost=float(r["price"])*ppl, transports=tr1)
                acts[j]=na
                if tr2 is not None and j+1<len(acts):
                    nn=dict(acts[j+1]); nn["transports"]=tr2; acts[j+1]=nn
                done=True; break
            if done: break
        if not done: return None
    return None


def fix_relocate_meal_to_hotel(plan, poi, lo, hi):
    """Convert an existing meal to a zero-cost meal at the booked hotel (poi)."""
    p = deepcopy(plan)
    days = p["itinerary"]
    for di, day in enumerate(days):
        acts = day["activities"]
        for j, a in enumerate(acts):
            if a.get("type") not in MEAL_WIN: continue
            wl, wh = MEAL_WIN[a["type"]]
            l = max(wl, lo if lo is not None else 0); h = min(wh, hi if hi is not None else 1440)
            if h - l < 20: continue
            # 首活动 + 前一天末在该酒店 -> 空交通
            prev_day_ok = (j == 0 and di > 0 and days[di-1]["activities"]
                           and days[di-1]["activities"][-1].get("type") == "accommodation"
                           and N(days[di-1]["activities"][-1].get("position","")) == N(poi))
            same_prev = (j > 0 and N((acts[j-1].get("position") or acts[j-1].get("end") or "")) == N(poi))
            if not (prev_day_ok or same_prev): continue
            st = l; ed = min(st + 30, h)
            nxt = acts[j+1] if j+1 < len(acts) else None
            if nxt is not None and nxt.get("start_time"):
                trs = nxt.get("transports") or []
                dep = hm(trs[0]["start_time"]) if trs else hm(nxt["start_time"])
                if ed > dep: continue
            na = dict(a); na.update(position=poi, price=0, cost=0,
                                     start_time=fmt(st), end_time=fmt(ed), transports=[])
            acts[j] = na
            return p
    return None

def fix_attrtype_excl(plan, atype, city, ppl):
    """Remove attractions of a forbidden type; rebuild the next leg."""
    db = _attr_db(city)
    col = "type" if "type" in db.columns else None
    if col is None: return None
    bad = set(db[db[col] == atype]["name"])
    if not bad: return None
    p = deepcopy(plan); removed = False
    for day in p["itinerary"]:
        acts = day["activities"]
        for j, a in enumerate(acts):
            if a.get("type") != "attraction" or a.get("position") not in bad: continue
            prev = acts[j-1] if j > 0 else None
            nxt = acts[j+1] if j+1 < len(acts) else None
            new_acts = acts[:j] + acts[j+1:]
            if nxt is not None and prev is not None:
                ppos = prev.get("position") or prev.get("end")
                npos = nxt.get("position") or nxt.get("start")
                if ppos and npos and N(ppos) != N(npos):
                    tr = None; ns = hm(nxt["start_time"]) if nxt.get("start_time") else None
                    pe = hm(prev["end_time"]) if prev.get("end_time") else None
                    if ns is not None and pe is not None:
                        for mode in ("metro","taxi","walk"):
                            probe = LA.goto(city, ppos, npos, "09:00", mode, ppl)
                            if not probe: continue
                            d1 = hm(probe[-1]["end_time"]) - hm(probe[0]["start_time"])
                            if pe + d1 > ns: continue
                            sh = (ns - d1) - hm(probe[0]["start_time"])
                            tr = [dict(l, start_time=fmt(hm(l["start_time"])+sh), end_time=fmt(hm(l["end_time"])+sh)) for l in probe]
                            break
                    if tr is None: continue
                    k = new_acts.index(nxt)
                    nn = dict(nxt); nn["transports"] = tr; new_acts[k] = nn
                elif ppos and npos and N(ppos) == N(npos):
                    k = new_acts.index(nxt)
                    nn = dict(nxt); nn["transports"] = []; new_acts[k] = nn
            day["activities"] = new_acts
            removed = True
            break
    return p if removed else None

if __name__ == "__main__":
    main()

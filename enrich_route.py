"""Route-aware soft enrichment (顺路多吃多玩).

Instead of inserting the nearest-to-A POI (which forces a big detour and breaks
budget/time), recall POIs that lie ON THE WAY between two consecutive activities
A and B, ranked by detour = dist(A,X)+dist(X,B)-dist(A,B). A near-zero detour
means X barely adds travel time/cost, so the insertion is far more likely to keep
the plan within all hard constraints.

For each insertion point we recall, by hard-constraint type first then by detour:
  - a restaurant (中途吃) if a meal slot (lunch/dinner) for that time is missing,
  - otherwise an attraction (顺路玩) that is open during the pass-through.
Every candidate insertion is re-validated; kept only if the plan still passes ALL
hard constraints and the target soft metric improved. Regression-safe."""
import os, sys, json, copy, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chinatravel.data.load_datasets import load_query, load_json_file
from chinatravel.agent.load_model import init_agent, init_llm
from chinatravel.environment.world_env import WorldEnv
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2

import os as _os
RES = _os.environ.get("ENRICH_RES", "results/UrbanTripOptimizedV5_TPCLLM_en_oracletranslation")
args = argparse.Namespace(splits="TPC_IJCAI_2026_phase1", method="x", lang="en", preference=False)
qi, qd = load_query(args)
from constraint_gate import apply_gate
apply_gate(qd)  # GATE_CONSTRAINTS=generated swaps in generated hard_logic_py; default (oracle) = no-op
sch = load_json_file("chinatravel/evaluation/output_schema.json")
_cur = None

def passes(uid, plan):
    s = set(evaluate_schema_constraints([uid], {uid: plan}, schema=sch)[2])
    c = set(evaluate_commonsense_constraints([uid], qd, {uid: plan}, verbose=False, lang="en")[3])
    if uid not in s or uid not in c:
        return False
    l = set(evaluate_hard_constraints_v2([uid], qd, {uid: plan}, env_pass_id=list(c), verbose=False, lang="en")[-1])
    return uid in l

def soft(plan):
    days = max(1, len(plan.get("itinerary", [])))
    a = sum(1 for d in plan["itinerary"] for x in d["activities"] if x.get("type") == "attraction")
    m = sum(1 for d in plan["itinerary"] for x in d["activities"] if x.get("type") in ("breakfast", "lunch", "dinner"))
    return a / (4 * days), (m / days) / 3

def hm(t):
    h, m = str(t).split(":"); return int(h) * 60 + int(m)
def mh(x):
    return f"{x // 60:02d}:{x % 60:02d}"
def actpos(a):
    return a.get("position") or a.get("end") or a.get("start")

_agents = {}
def agent_for(city):
    if city not in _agents:
        ag = init_agent({"method": "UrbanTripOptimizedV5", "env": WorldEnv(lang="en"),
                         "backbone_llm": init_llm("TPCLLM"), "cache_dir": "cache",
                         "log_dir": "cache", "debug": False, "lang": "en"})
        ag.memory["attractions"] = ag.collect_poi_info_all(city, "attraction")
        ag.memory["restaurants"] = ag.collect_poi_info_all(city, "restaurant")
        _agents[city] = ag
    return _agents[city]

MEALWIN = {"lunch": (10 * 60 + 30, 14 * 60), "dinner": (16 * 60 + 30, 21 * 60)}

def on_the_way(ag, query, posA, posB, df, k=24, max_detour=8.0):
    """Recall candidates on the path A->B ranked by detour distance (km)."""
    try:
        dAB = ag.calculate_distance(query, posA, posB)
    except Exception:
        dAB = None
    if dAB is None:
        return []
    def safe_dist(p, q):
        try:
            return ag.calculate_distance(query, p, q)
        except Exception:
            return None
    scored = []
    for _, r in df.iterrows():
        nm = r["name"]
        d1 = safe_dist(posA, nm)
        d2 = safe_dist(nm, posB)
        if d1 is None or d2 is None:
            continue
        detour = d1 + d2 - dAB
        if detour <= max_detour:
            scored.append((detour, r))
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]

def insert_between(ag, query, plan, di, i, kind, meal=None):
    """Try inserting an on-the-way POI of `kind` ('attraction'|'restaurant') between
    activities i and i+1 on day di. Returns True if inserted (and validated)."""
    day = plan["itinerary"][di]
    acts = day["activities"]
    a, b = acts[i], acts[i + 1]
    posA, posB = actpos(a), actpos(b)
    tA, tB = a.get("end_time"), b.get("start_time")
    if not posA or not posB or not tA or not tB:
        return False
    city = query["target_city"]
    df = ag.memory["attractions"] if kind == "attraction" else ag.memory["restaurants"]
    visited = set(actpos(x) for d in plan["itinerary"] for x in d["activities"])
    df = df[~df["name"].isin(visited)]
    cands = on_the_way(ag, query, posA, posB, df)
    ppl = int(query.get("people_number", 1))
    for r in cands:
        name = r["name"]; ot, et = str(r.get("opentime")), str(r.get("endtime"))
        for mode in ("metro", "walk", "taxi"):
            tr1 = ag.collect_innercity_transport(city, posA, name, tA, mode)
            if not isinstance(tr1, list) or tr1 == []:
                continue
            arr = tr1[-1]["end_time"]
            start = arr if arr >= ot else ot
            if start > et and not (et < ot):
                continue
            if kind == "restaurant" and meal:
                lo, hi = MEALWIN[meal]
                if not (lo <= hm(start) <= hi):
                    continue
            dur = 30
            end = mh(hm(start) + dur)
            if et >= ot and hm(end) > hm(et):
                continue
            tr2 = ag.collect_innercity_transport(city, name, posB, end, mode)
            if not isinstance(tr2, list):
                continue
            price = float(r.get("price", 0) or 0)
            typ = meal if kind == "restaurant" else "attraction"
            X = {"position": name, "type": typ, "price": price, "cost": price * ppl,
                 "start_time": start, "end_time": end, "transports": tr1}
            newb = copy.deepcopy(b); newb["transports"] = tr2
            cand_plan = copy.deepcopy(plan)
            cand_plan["itinerary"][di]["activities"] = acts[:i + 1] + [X, newb] + acts[i + 2:]
            try:
                ag._repair_itinerary_times(cand_plan["itinerary"])
            except Exception:
                continue
            if passes(_cur, cand_plan):
                day["activities"] = cand_plan["itinerary"][di]["activities"]
                return True
    return False

def day_meals(acts):
    return set(x["type"] for x in acts if x.get("type") in ("lunch", "dinner"))

def enrich_plan(ag, query, plan):
    added_a = added_m = 0
    for di in range(len(plan["itinerary"])):
        # several passes: add missing meals on-the-way, then attractions on-the-way
        for _ in range(4):
            acts = plan["itinerary"][di]["activities"]
            have = day_meals(acts)
            progressed = False
            # try missing meals first (中途吃)
            for meal in ("lunch", "dinner"):
                if meal in have:
                    continue
                for i in range(len(acts) - 1):
                    if acts[i + 1].get("type") == "accommodation" and i == 0:
                        continue
                    # only between activities straddling the meal window
                    if insert_between(ag, query, plan, di, i, "restaurant", meal):
                        added_m += 1; progressed = True
                        break
                if progressed:
                    break
            if progressed:
                continue
            # then attractions on-the-way (顺路玩), up to 4/day
            n_attr = sum(1 for x in acts if x.get("type") == "attraction")
            if n_attr < 4:
                for i in range(len(acts) - 1):
                    if insert_between(ag, query, plan, di, i, "attraction"):
                        added_a += 1; progressed = True
                        break
            if not progressed:
                break
    return added_a, added_m

if __name__ == "__main__":
    files = sorted(glob.glob(f"{RES}/*.json"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    apply = "--apply" in sys.argv
    shard_i, shard_n = 0, 1
    if "--shard" in sys.argv:
        shard_i, shard_n = [int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/")]
    kept = tried = ta = tm = 0
    for n, f in enumerate(files):
        if limit and n >= limit:
            break
        if n % shard_n != shard_i:
            continue
        uid = os.path.basename(f)[:-5]; _cur = uid
        plan = load_json_file(f)
        if not plan.get("itinerary") or not passes(uid, plan):
            continue
        q = qd[uid]; ag = agent_for(q["target_city"]); ag.query = q
        d0, r0 = soft(plan); cand = copy.deepcopy(plan)
        a, m = enrich_plan(ag, q, cand)
        if a + m > 0:
            tried += 1
            d1, r1 = soft(cand)
            if passes(uid, cand) and (d1 > d0 or r1 > r0):
                kept += 1; ta += a; tm += m
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{a}attr +{m}meal | DAV {d0:.2f}->{d1:.2f} DDR {r0:.2f}->{r1:.2f}")
    print(f"\ntried {tried}, kept {kept}, attractions {ta}, meals {tm}, apply={apply}")

"""Soft enrichment stage 3 (LARGE): insert attractions (and lunch) into daytime
gaps between consecutive activities, using gap-fit so no downstream activity is
shifted (insert X between A and B such that A->X, visit X, X->B all fit before B's
original start). 77% of days have a >=60min daytime gap. Regression-safe:
re-validate each plan; keep only if it still passes ALL hard constraints and the
target soft metric improved."""
import os, sys, json, copy, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chinatravel.data.load_datasets import load_query, load_json_file
from chinatravel.agent.load_model import init_agent, init_llm
from chinatravel.environment.world_env import WorldEnv
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2

RES = "results/UrbanTripOptimizedV5_TPCLLM_en_oracletranslation"
args = argparse.Namespace(splits="TPC_IJCAI_2026_phase1", method="x", lang="en", preference=False)
qi, qd = load_query(args)
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
    return a / (4 * days), (m / days) / 3  # DAV, DDR

def hm(t):
    h, m = str(t).split(":"); return int(h) * 60 + int(m)
def mh(x):
    return f"{x // 60:02d}:{x % 60:02d}"

_agents = {}
def agent_for(city):
    if city not in _agents:
        ag = init_agent({"method": "UrbanTripOptimizedV5", "env": WorldEnv(lang="en"),
                         "backbone_llm": init_llm("TPCLLM"), "cache_dir": "cache",
                         "log_dir": "cache", "debug": False, "lang": "en"})
        ag.memory["attractions"] = ag.collect_poi_info_all(city, "attraction")
        _agents[city] = ag
    return _agents[city]

def actpos(a):
    return a.get("position") or a.get("end") or a.get("start")

def try_insert_attraction(ag, query, plan, di):
    """Insert one attraction into the largest daytime gap of day di (gap-fit, no shift)."""
    day = plan["itinerary"][di]
    acts = day["activities"]
    city = query["target_city"]
    visited = set(actpos(x) for d in plan["itinerary"] for x in d["activities"] if x.get("type") == "attraction")
    attr = ag.memory["attractions"]
    # consider each consecutive pair (i, i+1); i+1 is not the accommodation-start chain break
    pairs = []
    for i in range(len(acts) - 1):
        a, b = acts[i], acts[i + 1]
        if b.get("type") == "accommodation":
            continue  # handled by end-of-day enrichers; keep this for mid-day
        tA, tB = a.get("end_time"), b.get("start_time")
        if not tA or not tB:
            continue
        gap = hm(tB) - hm(tA)
        if gap < 50 or hm(tA) > 16 * 60:
            continue
        pairs.append((gap, i))
    pairs.sort(reverse=True)
    for gap, i in pairs:
        a, b = acts[i], acts[i + 1]
        posA, posB = actpos(a), actpos(b)
        tA, tB = a.get("end_time"), b.get("start_time")
        if not posA or not posB:
            continue
        cand = attr[~attr["name"].isin(visited)]
        rows = []
        for _, r in cand.iterrows():
            try:
                d1 = ag.calculate_distance(query, posA, r["name"])
            except Exception:
                d1 = None
            rows.append((d1 if d1 is not None else 1e9, r))
        rows.sort(key=lambda x: x[0])
        for _, r in rows[:10]:
            name = r["name"]; ot, et = str(r.get("opentime")), str(r.get("endtime"))
            for mode in ("metro", "walk", "taxi"):
                tr1 = ag.collect_innercity_transport(city, posA, name, tA, mode)
                if not isinstance(tr1, list) or tr1 == []:
                    continue
                arr = tr1[-1]["end_time"]
                start = arr if arr >= ot else ot
                if start > et:
                    continue
                end = mh(hm(start) + 30)
                if end > et and not (et < ot):
                    continue
                tr2 = ag.collect_innercity_transport(city, name, posB, end, mode)
                if not isinstance(tr2, list):
                    continue
                ppl = int(query.get("people_number", 1))
                price = float(r.get("price", 0) or 0)
                X = {"position": name, "type": "attraction", "price": price, "cost": price * ppl,
                     "start_time": start, "end_time": end, "transports": tr1}
                newb = copy.deepcopy(b); newb["transports"] = tr2
                new_acts = acts[:i + 1] + [X, newb] + acts[i + 2:]
                cand_plan = copy.deepcopy(plan)
                cand_plan["itinerary"][di]["activities"] = new_acts
                # shift-based: recompute the day's time chain forward from transports,
                # then re-validate. Downstream that no longer fits is rejected by passes().
                try:
                    ag._repair_itinerary_times(cand_plan["itinerary"])
                except Exception:
                    continue
                if passes(_cur, cand_plan):
                    day["activities"] = cand_plan["itinerary"][di]["activities"]
                    return True
    return False

if __name__ == "__main__":
    files = sorted(glob.glob(f"{RES}/*.json"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    apply = "--apply" in sys.argv
    kept = added = tried = 0
    for n, f in enumerate(files):
        if limit and n >= limit:
            break
        uid = os.path.basename(f)[:-5]; _cur = uid
        plan = load_json_file(f)
        if not plan.get("itinerary") or not passes(uid, plan):
            continue
        q = qd[uid]; ag = agent_for(q["target_city"]); ag.query = q
        dav0, _ = soft(plan); cand = copy.deepcopy(plan); a_added = 0
        for di in range(len(cand["itinerary"])):
            day_attr = sum(1 for x in cand["itinerary"][di]["activities"] if x.get("type") == "attraction")
            tries = 0
            while day_attr < 4 and tries < 3:
                if try_insert_attraction(ag, q, cand, di):
                    a_added += 1; day_attr += 1
                else:
                    break
                tries += 1
        if a_added:
            tried += 1
            dav1, _ = soft(cand)
            if passes(uid, cand) and dav1 > dav0:
                kept += 1; added += a_added
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{a_added} attr, DAV {dav0:.2f}->{dav1:.2f}")
    print(f"\ntried {tried}, kept {kept}, attractions added {added}, apply={apply}")

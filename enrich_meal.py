"""Soft enrichment stage 2 (meals): add a missing dinner (and lunch) to passing
plans, appended before the end-of-day hotel return. Restaurants stay open into the
evening, so they fit the end-of-day free slot (unlike attractions, which close).
Regression-safe: re-validate each plan; keep only if it still passes ALL hard
constraints and DDR improved."""
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
from constraint_gate import apply_gate
apply_gate(qd)  # GATE_CONSTRAINTS=generated swaps in generated hard_logic_py; default (oracle) = no-op
sch = load_json_file("chinatravel/evaluation/output_schema.json")
_cur_uid = None

def passes(uid, plan):
    s = set(evaluate_schema_constraints([uid], {uid: plan}, schema=sch)[2])
    c = set(evaluate_commonsense_constraints([uid], qd, {uid: plan}, verbose=False, lang="en")[3])
    if uid not in s or uid not in c:
        return False
    l = set(evaluate_hard_constraints_v2([uid], qd, {uid: plan}, env_pass_id=list(c), verbose=False, lang="en")[-1])
    return uid in l

def ddr(plan):
    m = sum(1 for d in plan.get("itinerary", []) for x in d.get("activities", []) if x.get("type") in ("breakfast", "lunch", "dinner"))
    days = max(1, len(plan.get("itinerary", [])))
    return (m / days) / 3

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
        ag.memory["restaurants"] = ag.collect_poi_info_all(city, "restaurant")
        _agents[city] = ag
    return _agents[city]

WIN = {"lunch": ("10:30", "14:00"), "dinner": ("16:30", "21:00")}

def try_add_meal(ag, query, plan, di, meal):
    day = plan["itinerary"][di]
    acts = day["activities"]
    if any(a.get("type") == meal for a in acts):
        return False
    acc_idx = next((i for i, a in enumerate(acts) if a.get("type") == "accommodation"), None)
    if acc_idx is None or acc_idx == 0:
        return False
    prev = acts[acc_idx - 1]
    posP = prev.get("position") or prev.get("end")
    tP = prev.get("end_time")
    acc = acts[acc_idx]; H = acc.get("position")
    if not posP or not tP or not H:
        return False
    lo, hi = WIN[meal]
    if hm(tP) > hm(hi):
        return False
    depart = max(hm(tP), hm(lo))
    city = query["target_city"]
    visited = set(a.get("position") for d in plan["itinerary"] for a in d.get("activities", []) if a.get("type") in ("breakfast", "lunch", "dinner"))
    res = ag.memory["restaurants"]
    cand = res[~res["name"].isin(visited)]
    rows = []
    for _, r in cand.iterrows():
        try:
            dist = ag.calculate_distance(query, posP, r["name"])
        except Exception:
            dist = None
        rows.append((dist if dist is not None else 1e9, r))
    rows.sort(key=lambda x: x[0])
    for _, r in rows[:10]:
        name = r["name"]; ot, et = str(r.get("opentime")), str(r.get("endtime"))
        for mode in ("metro", "walk", "taxi"):
            tr1 = ag.collect_innercity_transport(city, posP, name, mh(depart), mode)
            if not isinstance(tr1, list) or tr1 == []:
                continue
            arr = tr1[-1]["end_time"]
            start = arr if arr >= ot else ot
            hi_bound = min(hm(hi), hm(et) if et >= ot else hm(hi))
            if not (hm(lo) <= hm(start) <= hi_bound):
                continue
            end = mh(hm(start) + 30)
            if et >= ot and hm(end) > hm(et):
                continue
            tr2 = ag.collect_innercity_transport(city, name, H, end, mode)
            if not isinstance(tr2, list):
                continue
            ppl = int(query.get("people_number", 1))
            price = float(r.get("price", 0) or 0)
            newact = {"position": name, "type": meal, "price": price, "cost": price * ppl,
                      "start_time": start, "end_time": end, "transports": tr1}
            cand_acc = copy.deepcopy(acc); cand_acc["transports"] = tr2
            new_acts = acts[:acc_idx] + [newact, cand_acc] + acts[acc_idx + 1:]
            cand_plan = copy.deepcopy(plan)
            cand_plan["itinerary"][di]["activities"] = new_acts
            if passes(_cur_uid, cand_plan):
                day["activities"] = new_acts
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
        uid = os.path.basename(f)[:-5]; _cur_uid = uid
        plan = load_json_file(f)
        if not plan.get("itinerary") or not passes(uid, plan):
            continue
        q = qd[uid]; ag = agent_for(q["target_city"]); ag.query = q
        before = ddr(plan); cand = copy.deepcopy(plan); m_added = 0
        for di in range(len(cand["itinerary"])):
            for meal in ("dinner", "lunch"):
                if try_add_meal(ag, q, cand, di, meal):
                    m_added += 1
        if m_added:
            tried += 1
            if passes(uid, cand) and ddr(cand) > before:
                kept += 1; added += m_added
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{m_added} meals, DDR {before:.2f}->{ddr(cand):.2f}")
    print(f"\ntried {tried}, kept {kept}, meals added {added}, apply={apply}")

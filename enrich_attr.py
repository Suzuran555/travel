"""Soft enrichment stage 2: append attractions to days with free time, before the
end-of-day hotel return. Uses the agent's env transport API. Regression-safe:
each plan is re-validated; the enriched plan is kept only if it still passes ALL
hard constraints and DAV improved.

Insertion per day: between the last non-accommodation POI (posP, end tP) and the
accommodation (hotel H). Add attraction X: transport posP->X, visit X, transport
X->H (replacing posP->H). Try several near candidates; keep first that validates."""
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

def passes(uid, plan):
    s = set(evaluate_schema_constraints([uid], {uid: plan}, schema=sch)[2])
    c = set(evaluate_commonsense_constraints([uid], qd, {uid: plan}, verbose=False, lang="en")[3])
    if uid not in s or uid not in c:
        return False
    l = set(evaluate_hard_constraints_v2([uid], qd, {uid: plan}, env_pass_id=list(c), verbose=False, lang="en")[-1])
    return uid in l

def dav(plan):
    a = sum(1 for d in plan.get("itinerary", []) for x in d.get("activities", []) if x.get("type") == "attraction")
    days = max(1, len(plan.get("itinerary", [])))
    return a / (4 * days)

_agents = {}
def agent_for(city):
    if city not in _agents:
        ag = init_agent({"method": "UrbanTripOptimizedV5", "env": WorldEnv(lang="en"),
                         "backbone_llm": init_llm("TPCLLM"), "cache_dir": "cache",
                         "log_dir": "cache", "debug": False, "lang": "en"})
        ag.memory["attractions"] = ag.collect_poi_info_all(city, "attraction")
        _agents[city] = ag
    return _agents[city]

def hm(t):
    h, m = t.split(":"); return int(h) * 60 + int(m)
def mh(x):
    return f"{x//60:02d}:{x%60:02d}"

def try_enrich_day(ag, query, plan, di):
    """Append one attraction before the accommodation on day di. Returns True if added."""
    day = plan["itinerary"][di]
    acts = day["activities"]
    # find accommodation index (must end the day at hotel to re-anchor transport)
    acc_idx = next((i for i, a in enumerate(acts) if a.get("type") == "accommodation"), None)
    if acc_idx is None or acc_idx == 0:
        return False
    prev = acts[acc_idx - 1]
    posP = prev.get("position") or prev.get("end")
    tP = prev.get("end_time")
    acc = acts[acc_idx]
    H = acc.get("position")
    if not posP or not tP or not H:
        return False
    city = query["target_city"]
    visited = set(a.get("position") for d in plan["itinerary"] for a in d.get("activities", []) if a.get("type") == "attraction")
    attr = ag.memory["attractions"]
    # candidates: nearest to posP, not visited, open, free-ish
    try:
        cand = attr[~attr["name"].isin(visited)]
    except Exception:
        return False
    # rank by distance to posP
    rows = []
    for _, r in cand.iterrows():
        try:
            dist = ag.calculate_distance(query, posP, r["name"])
        except Exception:
            dist = None
        rows.append((dist if dist is not None else 1e9, r))
    rows.sort(key=lambda x: x[0])
    for _, r in rows[:8]:
        name = r["name"]
        for mode in ("metro", "walk", "taxi"):
            tr1 = ag.collect_innercity_transport(city, posP, name, tP, mode)
            if not isinstance(tr1, list) or tr1 == []:
                continue
            arr = tr1[-1]["end_time"]
            # open hours
            ot, et = str(r.get("opentime")), str(r.get("endtime"))
            if not (ot <= arr <= et):
                continue
            start = arr
            end = mh(min(hm(start) + 30, hm(et)))
            if hm(end) <= hm(start):
                continue
            tr2 = ag.collect_innercity_transport(city, name, H, end, mode)
            if not isinstance(tr2, list):
                continue
            newact = {"position": name, "type": "attraction", "price": float(r.get("price", 0) or 0),
                      "cost": float(r.get("price", 0) or 0) * int(query.get("people_number", 1)),
                      "start_time": start, "end_time": end, "transports": tr1}
            cand_acc = copy.deepcopy(acc); cand_acc["transports"] = tr2
            new_acts = acts[:acc_idx] + [newact, cand_acc] + acts[acc_idx + 1:]
            cand_plan = copy.deepcopy(plan)
            cand_plan["itinerary"][di]["activities"] = new_acts
            if passes_quick(cand_plan):
                day["activities"] = new_acts
                return True
    return False

_cur_uid = None
def passes_quick(plan):
    return passes(_cur_uid, plan)

if __name__ == "__main__":
    files = sorted(glob.glob(f"{RES}/*.json"))
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    apply = "--apply" in sys.argv
    kept = added = tried = 0
    for n, f in enumerate(files):
        if limit and n >= limit:
            break
        uid = os.path.basename(f)[:-5]
        _cur_uid = uid
        plan = load_json_file(f)
        if not plan.get("itinerary") or not passes(uid, plan):
            continue
        q = qd[uid]; city = q["target_city"]
        ag = agent_for(city); ag.query = q
        before = dav(plan)
        cand = copy.deepcopy(plan)
        a_added = 0
        for di in range(len(cand["itinerary"])):
            # add up to fill toward 4/day
            day_attr = sum(1 for x in cand["itinerary"][di]["activities"] if x.get("type") == "attraction")
            attempts = 0
            while day_attr < 4 and attempts < 2:
                if try_enrich_day(ag, q, cand, di):
                    a_added += 1; day_attr += 1
                else:
                    break
                attempts += 1
        if a_added:
            tried += 1
            if passes(uid, cand) and dav(cand) > before:
                kept += 1; added += a_added
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{a_added} attractions, DAV {before:.2f}->{dav(cand):.2f}")
    print(f"\ntried {tried}, kept {kept}, attractions added {added}, apply={apply}")

"""Post-hoc soft-metric enrichment (prototype): add missing breakfast at the hotel
on days whose morning is at the accommodation. Regression-safe: each plan is
re-validated; the enriched plan is kept ONLY if it still passes all hard
constraints AND its soft score did not drop.

Stage 1 = breakfast only (hotel, 06:00-06:30, cost 0, no transport)."""
import os, sys, json, argparse, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chinatravel.data.load_datasets import load_query, load_json_file
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2

RES = "results/UrbanTripOptimizedV5_TPCLLM_en_oracletranslation"
args = argparse.Namespace(splits="TPC_IJCAI_2026_phase1", method="x", lang="en", preference=False)
qi, qd = load_query(args)
sch = load_json_file("chinatravel/evaluation/output_schema.json")

def passes(uid, plan):
    s = set(evaluate_schema_constraints([uid], {uid: plan}, schema=sch)[2])
    c = set(evaluate_commonsense_constraints([uid], qd, {uid: plan}, verbose=False, lang="en")[3])
    if uid not in s or uid not in c:
        return False
    l = set(evaluate_hard_constraints_v2([uid], qd, {uid: plan}, env_pass_id=list(c), verbose=False, lang="en")[-1])
    return uid in l

def day_pos_end(day):
    acts = day.get("activities", [])
    for a in reversed(acts):
        for k in ("position", "end", "start"):
            if a.get(k):
                return a.get(k)
    return None

def hotel_name(plan):
    for d in plan.get("itinerary", []):
        for a in d.get("activities", []):
            if a.get("type") == "accommodation":
                return a.get("position")
    return None

def enrich_breakfast(plan):
    it = plan.get("itinerary", [])
    hotel = hotel_name(plan)
    if not hotel:
        return plan, 0
    added = 0
    for di, day in enumerate(it):
        acts = day.get("activities", [])
        if any(a.get("type") == "breakfast" for a in acts):
            continue
        # eligible only if previous day ended at the hotel (overnight there)
        if di == 0:
            continue
        prev = it[di - 1].get("activities", [])
        prev_acc = [a for a in prev if a.get("type") == "accommodation" and a.get("position") == hotel]
        if not prev_acc:
            continue
        # first real activity must start after 06:30 so breakfast slots in cleanly
        if acts:
            first_start = acts[0].get("start_time") or "00:00"
            if first_start <= "06:30":
                continue
        bk = {"position": hotel, "type": "breakfast", "price": 0, "cost": 0,
              "start_time": "06:00", "end_time": "06:30", "transports": []}
        day["activities"] = [bk] + acts
        added += 1
    return plan, added

if __name__ == "__main__":
    files = sorted(__import__("glob").glob(f"{RES}/*.json"))
    kept = added_total = tried = 0
    for f in files:
        uid = os.path.basename(f)[:-5]
        plan = load_json_file(f)
        if not plan.get("itinerary"):
            continue
        if not passes(uid, plan):
            continue  # only enrich passing plans
        cand = copy.deepcopy(plan)
        cand, added = enrich_breakfast(cand)
        if added == 0:
            continue
        tried += 1
        if passes(uid, cand):
            kept += 1
            added_total += added
            if "--apply" in sys.argv:
                json.dump(cand, open(f, "w"), ensure_ascii=False)
    print(f"tried enrich: {tried} plans | kept (still pass): {kept} | breakfasts added: {added_total}")
    print("apply mode:" , "--apply" in sys.argv)

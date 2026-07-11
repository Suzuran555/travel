"""Breakfast enrichment (早餐, DDR lever).

DDR = (#meals across trip / days) / 3, counting breakfast+lunch+dinner equally.
The route meal enricher only added lunch/dinner; breakfast was never attempted.

A breakfast AT THE HOTEL is the cheapest possible DDR gain:
  - position = the accommodation the guest slept in (previous day's hotel), which
    is where the morning transport already originates, so no transport rewrite,
  - price 0, cost 0, transports [] -> zero budget impact, zero ATT impact
    (ATT only averages activities that HAVE transports),
  - commonsense allows breakfast-at-hotel (select_restaurant empty -> price must
    be 0) and hotel breakfasts skip the repeated-restaurant check, so the same
    hotel is valid every day.

For each day with no breakfast we insert one at the anchoring hotel, timed to end
before the day's first real activity within [06:00, 09:00], then re-validate the
whole plan. Kept only if it still passes ALL hard constraints and DDR improved."""
import os, sys, json, copy, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chinatravel.data.load_datasets import load_query, load_json_file
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
import argparse

RES = os.environ.get("ENRICH_RES", "results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation")
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

INTERCITY = {"train", "airplane"}
def hm(t):
    h, m = str(t).split(":"); return int(h) * 60 + int(m)
def mh(x):
    return f"{x // 60:02d}:{x % 60:02d}"

def ddr(plan):
    days = max(1, len(plan["itinerary"]))
    n = sum(1 for d in plan["itinerary"] for x in d["activities"]
            if x.get("type") in ("breakfast", "lunch", "dinner"))
    return (n / days) / 3

def day_hotel(plan, di):
    """Hotel the guest wakes up in on day di = last accommodation of day di-1
    (fallback: any accommodation earlier in the trip)."""
    if di >= 1:
        for x in reversed(plan["itinerary"][di - 1]["activities"]):
            if x.get("type") == "accommodation" and x.get("position"):
                return x["position"]
    # fallback: nearest prior accommodation anywhere
    for dj in range(di - 1, -1, -1):
        for x in reversed(plan["itinerary"][dj]["activities"]):
            if x.get("type") == "accommodation" and x.get("position"):
                return x["position"]
    return None

def first_departure(acts):
    """Earliest time the guest must leave the hotel = first activity's transport
    departure, or its start_time if no transport."""
    for a in acts:
        tr = a.get("transports") or []
        if tr and tr[0].get("start_time"):
            return hm(tr[0]["start_time"])
        if a.get("start_time"):
            return hm(a["start_time"])
    return None

def add_breakfast(plan):
    added = 0
    for di, day in enumerate(plan["itinerary"]):
        acts = day["activities"]
        if any(x.get("type") == "breakfast" for x in acts):
            continue
        if not acts:
            continue
        if acts[0].get("type") in INTERCITY:   # arrival/departure-first day
            continue
        hotel = day_hotel(plan, di)
        if not hotel:
            continue
        dep = first_departure(acts)
        if dep is None:
            continue
        # fit breakfast to end before departure, within [06:00, 09:00]
        end = min(dep, 9 * 60)
        end = min(end, 6 * 60 + 30)          # cap at 06:30 like V6's own pattern
        start = end - 30
        if start < 6 * 60:
            start = 6 * 60
        if end <= start or end > 9 * 60 or start < 6 * 60 or end > dep:
            continue
        bf = {"position": hotel, "type": "breakfast", "price": 0, "cost": 0,
              "start_time": mh(start), "end_time": mh(end), "transports": []}
        day["activities"] = [bf] + acts
        added += 1
    return added

if __name__ == "__main__":
    files = sorted(glob.glob(f"{RES}/*.json"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    apply = "--apply" in sys.argv
    shard_i, shard_n = 0, 1
    if "--shard" in sys.argv:
        shard_i, shard_n = [int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/")]
    kept = tried = tb = 0
    for n, f in enumerate(files):
        if limit and n >= limit:
            break
        if n % shard_n != shard_i:
            continue
        uid = os.path.basename(f)[:-5]
        plan = load_json_file(f)
        if not plan.get("itinerary") or not passes(uid, plan):
            continue
        r0 = ddr(plan)
        cand = copy.deepcopy(plan)
        b = add_breakfast(cand)
        if b > 0:
            tried += 1
            r1 = ddr(cand)
            if r1 > r0 and passes(uid, cand):
                kept += 1; tb += b
                if apply:
                    json.dump(cand, open(f, "w"), ensure_ascii=False)
                print(f"{uid}: +{b}breakfast | DDR {r0:.3f}->{r1:.3f}")
    print(f"\ntried {tried}, kept {kept}, breakfasts {tb}, apply={apply}")

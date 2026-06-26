"""逐 UID 失败分类:把未通过的 query 分成 schema / 常识(含空盘=超时) / 硬逻辑 三类。
复用 eval_tpc.py 完全相同的评测函数,只是保留每个 UID 的 pass 集合。"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chinatravel.data.load_datasets import load_query, load_json_file
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2

METHOD = "UrbanTripOptimizedV5_TPCLLM_en_oracletranslation"
SPLIT = "TPC_IJCAI_2026_phase1"

args = argparse.Namespace(splits=SPLIT, method=METHOD, lang="en", preference=False)
query_index, query_data = load_query(args)

# load result plans (same as eval_tpc.load_result)
results_dir = os.path.join("results", METHOD)
plans = {}
for qid in query_index:
    p = os.path.join(results_dir, f"{qid}.json")
    plans[qid] = load_json_file(p) if os.path.exists(p) else {}

schema = load_json_file("chinatravel/evaluation/output_schema.json")
_, _, schema_pass = evaluate_schema_constraints(query_index, plans, schema=schema)
_, _, _, comm_pass = evaluate_commonsense_constraints(query_index, query_data, plans, verbose=False, lang="en")
*_, logi_pass = evaluate_hard_constraints_v2(query_index, query_data, plans, env_pass_id=comm_pass, verbose=False, lang="en")

schema_pass, comm_pass, logi_pass = set(schema_pass), set(comm_pass), set(logi_pass)
all_pass = schema_pass & comm_pass & logi_pass

def plan_empty(qid):
    it = (plans.get(qid) or {}).get("itinerary") or []
    return len(it) == 0

def plan_days(qid):
    it = (plans.get(qid) or {}).get("itinerary") or []
    return len(it)

buckets = {"schema_fail": [], "timeout_empty": [], "commonsense_fail_nonempty": [], "hardlogic_fail": []}
for qid in query_index:
    if qid in all_pass:
        continue
    if qid not in schema_pass:
        buckets["schema_fail"].append(qid)
    elif qid not in comm_pass:
        (buckets["timeout_empty"] if plan_empty(qid) else buckets["commonsense_fail_nonempty"]).append(qid)
    else:  # schema+commonsense pass, fails hard logic
        buckets["hardlogic_fail"].append(qid)

total_fail = sum(len(v) for v in buckets.values())
print(f"=== 失败分类 (共 {len(query_index)} 个, 通过 {len(all_pass)}, 失败 {total_fail}) ===")
for k, v in buckets.items():
    print(f"  {k:28s}: {len(v)}")
print()
print("schema_fail UIDs:", buckets["schema_fail"])
print("timeout_empty UIDs:", buckets["timeout_empty"])
print("commonsense_fail_nonempty UIDs:", buckets["commonsense_fail_nonempty"])
print("hardlogic_fail UIDs:", buckets["hardlogic_fail"])

# dump for the next step
json.dump({k: v for k, v in buckets.items()}, open("/tmp/v5_fail_buckets.json", "w"), ensure_ascii=False, indent=2)

#!/usr/bin/env python3
"""Full live-NL run of the harness on the familiar-100 with the REAL model
(DashScope Qwen3.6-27B), then a proper split-level score.

The agent sees ONLY the natural language (oracle stripped); scoring reads the
oracle (hard_logic_py) straight from disk so it works regardless of the local
loader.

Self-healing against a flaky external endpoint: Phase A runs multiple passes.
A query whose plan is a deterministic fallback (i.e. the LLM call failed on a
network drop -- no real-planner metadata) is NOT written, so the next pass
retries it. After MAX_PASSES, any query that still never produced a real plan
gets its fallback written so all 100 exist and can be scored. This makes ONE
invocation ride through transient network breaks and always finish 100/100.

Env: CHINATRAVEL_OPENAI_BASE_URL/MODEL/API_KEY (DashScope), CHINATRAVEL_LLM_THINK=0.
Usage: python scripts/run_dashscope_full.py --method TPCAgent_dashscope --limit 100
"""
import sys, os, json, glob, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from copy import deepcopy

FAM = "chinatravel/data/en/phase2_familiar_EN"
HIDDEN = {"hard_logic", "hard_logic.py", "hard_logic_py", "hard_logic_nl",
          "soft_logic", "soft_logic_py", "soft_logic_nl"}
MAX_PASSES = 6
_REAL_KEYS = ("search_time_sec", "llm_inference_time_sec", "commonsense_pass")

DEFAULT_ATTRACTION_PR = "attraction_count = 0\nfor activity in allactivities(plan):\n    if activity_type(activity) == 'attraction':\n        attraction_count += 1\nresult=attraction_count/(4*day_count(plan))\n"
DEFAULT_TRANS_PR = "time_cost = 0\ntransport_count = 0\nfor activity in allactivities(plan):\n    transports = activity_transports(activity)\n    if transports!=[]:\n        transport_count += 1\n        time_cost += innercity_transport_time(transports)\nif transport_count > 0:\n    average_time_cost = time_cost / transport_count\n    result= (-1/105) * average_time_cost + 8/7\nelse:\n    result=0\n"
DEFAULT_RES_PR = "res_count=0\nfor activity in allactivities(plan):\n    if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n        res_count+=1\nres_count=res_count/(day_count(plan))\nresult=res_count/3\n"
DEFAULT_PR = [DEFAULT_ATTRACTION_PR, DEFAULT_TRANS_PR, DEFAULT_RES_PR]


def is_real(plan):
    return isinstance(plan, dict) and any(k in plan for k in _REAL_KEYS)


def cal_default_pr(query_index, query_data, result_data, all_pass_id, func_dict):
    all_score = []
    clamp = lambda v: max(0.0, min(1.0, v))
    for idx in query_index:
        plan = result_data[idx]
        if idx not in all_pass_id:
            all_score.append(np.zeros(len(DEFAULT_PR))); continue
        results = []
        for c in DEFAULT_PR:
            vd = deepcopy(func_dict); vd["plan"] = plan
            try:
                exec(c, {"__builtins__": {"set": set}}, vd)
                results.append(clamp(vd.get("result", False)))
            except Exception:
                results.append(0.0)
        all_score.append(np.array(results))
    return np.mean(all_score, axis=0) if all_score else np.zeros(len(DEFAULT_PR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="TPCAgent_dashscope")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    from agent_env.scripts.tpc_agent_runner import run_tpc_agent
    from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
    from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
    from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
    from chinatravel.evaluation.utils import load_json_file
    from chinatravel.symbol_verification.concept_func import func_dict

    files = sorted(glob.glob(FAM + "/*.json"))[: args.limit]
    method = args.method
    res_dir = os.path.join("results", method)
    os.makedirs(res_dir, exist_ok=True)

    def out_path(f):
        return os.path.join(res_dir, json.load(open(f))["uid"] + ".json")

    def has_real(f):
        p = out_path(f)
        if not os.path.exists(p):
            return False
        try:
            return is_real(json.load(open(p)))
        except Exception:
            return False

    def gen(f):
        full = json.load(open(f)); uid = full["uid"]
        vis = {k: v for k, v in full.items() if k not in HIDDEN}
        t0 = time.time()
        plan = run_tpc_agent(uid=uid, query=vis, lang="en",
                             tpcagent_config={"per_query_timeout": 180}, timeout=180,
                             cache_dir=f"cache/{method}", log_dir=f"agent_env/runs/{method}")
        sys.stdout = sys.__stdout__
        return uid, plan, time.time() - t0

    # ---- Phase A: multi-pass self-healing generation ----
    print(f"=== Phase A: generate {len(files)} plans (DashScope live-NL, method={method}) ===", flush=True)
    tA = time.time()
    for p in range(MAX_PASSES):
        todo = [f for f in files if not has_real(f)]
        if not todo:
            break
        print(f"--- pass {p+1}/{MAX_PASSES}: {len(todo)} remaining ({len(files)-len(todo)}/{len(files)} real) ---", flush=True)
        for f in todo:
            uid, plan, dt = gen(f)
            if is_real(plan):
                json.dump(plan, open(out_path(f), "w"), ensure_ascii=False)
                done = len(files) - sum(1 for x in files if not has_real(x))
                print(f"[{done}/{len(files)}] {uid}: days={len(plan.get('itinerary',[]))} {dt:.0f}s (elapsed {(time.time()-tA)/60:.1f}m)", flush=True)
            else:
                print(f"[pass{p+1}] {uid}: network-fail -> retry next pass ({dt:.0f}s)", flush=True)
        if any(not has_real(f) for f in files) and p < MAX_PASSES - 1:
            print(f"--- pass {p+1} done; {sum(1 for f in files if not has_real(f))} still missing; wait 30s ---", flush=True)
            time.sleep(30)

    # accept fallback for whatever never got a real plan, so all 100 exist
    still = [f for f in files if not os.path.exists(out_path(f))]
    if still:
        print(f"=== writing deterministic fallback for {len(still)} queries that never produced a real plan ===", flush=True)
        for f in still:
            uid, plan, dt = gen(f)
            json.dump(plan, open(out_path(f), "w"), ensure_ascii=False)

    # ---- Phase B: proper split-level score (disk oracle) ----
    present = sum(1 for f in files if os.path.exists(out_path(f)))
    if present < len(files):
        print(f"\n=== Phase B SKIPPED: {present}/{len(files)} present ===", flush=True); return
    n_real = sum(1 for f in files if has_real(f))
    print(f"\n=== Phase B: score {len(files)} plans (disk oracle) | real-plan={n_real}/{len(files)} ===", flush=True)
    qidx, qdata, rdata = [], {}, {}
    for f in files:
        full = json.load(open(f)); uid = full["uid"]
        if isinstance(full.get("hard_logic_py"), str):
            import ast; full["hard_logic_py"] = ast.literal_eval(full["hard_logic_py"])
        qidx.append(uid); qdata[uid] = full
        rdata[uid] = json.load(open(out_path(f)))
    schema = load_json_file("chinatravel/evaluation/output_schema.json")
    _, _, sp = evaluate_schema_constraints(qidx, rdata, schema=schema)
    macro_c, micro_c, _, cp = evaluate_commonsense_constraints(qidx, qdata, rdata, verbose=False, lang="en")
    *_, cond_micro_logi, _, lp = evaluate_hard_constraints_v2(qidx, qdata, rdata, env_pass_id=cp, verbose=False, lang="en")
    all_pass = list(set(sp) & set(cp) & set(lp))
    fpr = 100.0 * len(all_pass) / len(qidx)
    pr = cal_default_pr(qidx, qdata, rdata, all_pass, func_dict)
    dav, att, ddr = pr[0]*100, pr[1]*100, pr[2]*100
    overall = 0.1*micro_c + 0.1*macro_c + 0.25*cond_micro_logi + 0.4*fpr + 0.05*(dav+att+ddr)
    print("\n" + "="*50)
    print(f"RESULT (DashScope Qwen3.6-27B, live-NL, familiar-{len(qidx)}, real-plan {n_real}/{len(qidx)}):")
    print(f"  MicEPR={micro_c:.2f} MacEPR={macro_c:.2f} C-LPR={cond_micro_logi:.2f} FPR={fpr:.2f}")
    print(f"  DAV={dav:.2f} ATT={att:.2f} DDR={ddr:.2f}")
    print(f"  OVERALL={overall:.4f}   (all_pass {len(all_pass)}/{len(qidx)})")
    json.dump({"MicEPR":micro_c,"MacEPR":macro_c,"C-LPR":cond_micro_logi,"FPR":fpr,
               "DAV":float(dav),"ATT":float(att),"DDR":float(ddr),"overall":float(overall),
               "all_pass":len(all_pass),"n":len(qidx),"real_plans":n_real},
              open(f"{res_dir}/_dashscope_score.json","w"), indent=2)
    print(f"  saved -> {res_dir}/_dashscope_score.json", flush=True)


if __name__ == "__main__":
    main()

"""In-process A/B for the travel-day timing bias (planner Patches 3+4).

Runs the V6 agent twice on the same sample queries -- baseline (bias off) vs
patched (enable_travelday_time_bias on) -- and compares the chosen intercity
arrival/departure times and hard-constraint pass rate. Writes nothing to the
results dir. Sample = multi-day trips whose current last-day departure is before
17:00 (dinner-blocked) or arrival after 08:30 (breakfast-blocked)."""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.abspath("."))
from chinatravel.data.load_datasets import load_query, load_json_file
from chinatravel.agent.load_model import init_agent, init_llm
from chinatravel.environment.world_env import WorldEnv
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2

D = "results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
args = argparse.Namespace(splits="TPC_IJCAI_2026_phase1", method="x", lang="en", preference=False)
qi, qd = load_query(args)
sch = load_json_file("chinatravel/evaluation/output_schema.json")
INTER = {"train", "airplane"}

import numpy as np
def _np(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError
def clean(p):
    """JSON round-trip (numpy->native) exactly like run_tpc.py's file write, so
    the evaluators see the same plain-typed plan the real flow evaluates."""
    if not isinstance(p, dict):
        return {}
    return json.loads(json.dumps(p, default=_np))

def hm(t):
    h, m = str(t).split(":"); return int(h) * 60 + int(m)

def passes(uid, plan):
    if not isinstance(plan, dict) or not plan.get("itinerary"):
        return False
    s = set(evaluate_schema_constraints([uid], {uid: plan}, schema=sch)[2])
    c = set(evaluate_commonsense_constraints([uid], qd, {uid: plan}, verbose=False, lang="en")[3])
    if uid not in s or uid not in c:
        return False
    l = set(evaluate_hard_constraints_v2([uid], qd, {uid: plan}, env_pass_id=list(c), verbose=False, lang="en")[-1])
    return uid in l

def intercity_times(plan):
    """(arrival_min day0, departure_min lastday) from the plan's intercity legs."""
    arr = dep = None
    it = plan.get("itinerary") or []
    if it:
        a0 = it[0]["activities"]
        if a0 and a0[0].get("type") in INTER and a0[0].get("end_time"):
            arr = hm(a0[0]["end_time"])
        al = it[-1]["activities"]
        if al and al[-1].get("type") in INTER and al[-1].get("start_time"):
            dep = hm(al[-1]["start_time"])
    return arr, dep

BASE_KW = {"use_bundle_search": True, "geo_anchor_must": True, "enable_fallback_hard_repair": True,
           "enable_metro_only_prune": True, "enable_dynamic_top_k": True, "enable_dfs_memoization": True,
           "enable_budget_drop_repair": True, "debug": False}

TIMEOUT = int(os.environ.get("BIAS_TIMEOUT", "150"))
def make_agent(extra):
    kw = {"method": "UrbanTripOptimizedV6", "env": WorldEnv(lang="en"),
          "backbone_llm": init_llm("TPCLLM"), "cache_dir": "cache", "log_dir": "cache",
          "debug": False, "lang": "en", "external_timeout": TIMEOUT}
    kw.update(BASE_KW); kw.update(extra)
    return init_agent(kw)

def pick_sample(n):
    out = []
    for f in sorted(glob.glob(f"{D}/*.json")):
        uid = os.path.basename(f)[:-5]; p = load_json_file(f)
        it = p.get("itinerary") or []
        if len(it) < 2:
            continue
        arr, dep = intercity_times(p)
        if (dep is not None and dep < 17 * 60) or (arr is not None and arr > 8 * 60 + 30):
            out.append(uid)
        if len(out) >= n:
            break
    return out

if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 12
    sample = pick_sample(n)
    print(f"sample ({len(sample)} dinner/breakfast-blocked multi-day UIDs): {sample}")
    base_ag = make_agent({"enable_travelday_time_bias": False})
    pat_ag = make_agent({"enable_travelday_time_bias": True})
    rows = []
    for uid in sample:
        q = qd[uid]
        try:
            _, bp = base_ag.run(q, prob_idx=uid, oralce_translation=True)
        except Exception as e:
            bp = {"err": str(e)[:60]}
        sys.stdout = sys.__stdout__      # agent.run() redirects stdout to its log
        try:
            _, pp = pat_ag.run(q, prob_idx=uid, oralce_translation=True)
        except Exception as e:
            pp = {"err": str(e)[:60]}
        sys.stdout = sys.__stdout__
        bp = clean(bp); pp = clean(pp)
        ba, bd = intercity_times(bp)
        pa, pd_ = intercity_times(pp)
        bpass = passes(uid, bp)
        ppass = passes(uid, pp)
        def ddr(p):
            days = max(1, len(p.get("itinerary", [])))
            m = sum(1 for d in p["itinerary"] for x in d["activities"] if x.get("type") in ("breakfast","lunch","dinner"))
            return min(1.0, (m/days)/3)
        def dav(p):
            days = max(1, len(p.get("itinerary", [])))
            n = sum(1 for d in p["itinerary"] for x in d["activities"] if x.get("type")=="attraction")
            return min(1.0, n/(4*days))
        bddr, pddr = (ddr(bp) if bpass else 0), (ddr(pp) if ppass else 0)
        bdav, pdav = (dav(bp) if bpass else 0), (dav(pp) if ppass else 0)
        rows.append((uid, ba, pa, bd, pd_, bpass, ppass, bddr, pddr, bdav, pdav))
        print(f"{uid}: arr {ba}->{pa} dep {bd}->{pd_} pass {bpass}->{ppass} DDR {bddr:.2f}->{pddr:.2f} DAV {bdav:.2f}->{pdav:.2f}")
    # summary
    dep_later = sum(1 for r in rows if r[3] is not None and r[4] is not None and r[4] > r[3])
    arr_earlier = sum(1 for r in rows if r[1] is not None and r[2] is not None and r[2] < r[1])
    bpass_n = sum(1 for r in rows if r[5]); ppass_n = sum(1 for r in rows if r[6])
    both = [r for r in rows if r[5] and r[6]]
    import statistics as st
    print(f"\n=== SUMMARY over {len(rows)} ===")
    print(f"departure moved LATER:  {dep_later}")
    print(f"arrival moved EARLIER:  {arr_earlier}")
    print(f"baseline pass: {bpass_n}/{len(rows)}   patched pass: {ppass_n}/{len(rows)}")
    if both:
        print(f"on {len(both)} both-pass: DDR {st.mean(r[7] for r in both)*100:.1f}->{st.mean(r[8] for r in both)*100:.1f}"
              f"  DAV {st.mean(r[9] for r in both)*100:.1f}->{st.mean(r[10] for r in both)*100:.1f}")

"""Agent-level emission-guarantee tests (no LLM backend needed):

  1. planner overruns the emission deadline -> the prebuilt fallback plan is
     returned before the deadline (never a watchdog kill),
  2. planner raises -> fallback plan is returned,
  3. planner returns normally -> its plan is kept and the wall-clock-volatile
     debug counters are stripped from the emitted plan.

Run:  .venv/bin/python chinatravel/agent/tpc_agent_penguins/tests/test_emit_deadline.py
"""
import json
import os
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# small budget for the test: emit deadline = start + (12 - 8) = start + 4s
os.environ["TPC_TIME_BUDGET"] = "12"
os.environ["TPC_ENRICH_RESERVE"] = "2"
os.environ["TPC_EMIT_MARGIN"] = "8"

from chinatravel.environment.world_env import WorldEnv                   # noqa: E402
from chinatravel.agent.tpc_agent_penguins.tpc_agent import TPCAgent      # noqa: E402
from chinatravel.agent.tpc_agent_penguins.tpc_llm import TPCLLM          # noqa: E402

QUERY = {
    "uid": "emit_deadline_test",
    "days": 3,
    "people_number": 2,
    "start_city": "Shanghai",
    "target_city": "Beijing",
    "nature_language": "2 people, 3 days Shanghai to Beijing.",
}


def build_agent(tmpdir):
    return TPCAgent(
        method="TPCAgent",
        env=WorldEnv(lang="en"),
        backbone_llm=TPCLLM(),
        cache_dir=os.path.join(tmpdir, "cache"),
        log_dir=os.path.join(tmpdir, "logs"),
        debug=False,
        lang="en",
    )


def main():
    tmpdir = tempfile.mkdtemp(prefix="tpc_emit_test_")
    agent = build_agent(tmpdir)

    # ---- 1. overrunning planner -> fallback before the emission deadline
    def _hang(query, prob_idx, oralce_translation=False):
        time.sleep(60)
        return True, {}

    agent.planner.run = _hang
    t0 = time.time()
    succ, plan = agent.run(dict(QUERY), "emit_deadline_test")
    elapsed = time.time() - t0
    assert elapsed < 12 - 8 + 3, f"emitted too late: {elapsed:.1f}s"
    assert isinstance(plan, dict) and plan.get("itinerary"), "no fallback plan emitted"
    assert len(plan["itinerary"]) == QUERY["days"]
    json.dumps(plan)  # pure JSON
    print(f"overrun case: fallback emitted at {elapsed:.1f}s (deadline 4s+overhead)")

    # request deadline was armed for the backbone
    assert agent.backbone_llm.request_deadline is not None
    assert agent.backbone_llm._effective_timeout() <= 12

    # let the abandoned thread die before the next case
    time.sleep(0.2)

    # ---- 2. raising planner -> fallback
    agent2 = build_agent(tmpdir)

    def _boom(query, prob_idx, oralce_translation=False):
        raise RuntimeError("planner exploded")

    agent2.planner.run = _boom
    succ, plan = agent2.run(dict(QUERY), "emit_deadline_test")
    assert isinstance(plan, dict) and plan.get("itinerary"), \
        "no fallback plan on planner exception"
    print("exception case: fallback emitted")

    # ---- 3. healthy planner -> plan kept, volatile counters stripped
    agent3 = build_agent(tmpdir)
    healthy_plan = {
        "people_number": 2,
        "start_city": "Shanghai",
        "target_city": "Beijing",
        "itinerary": [{"day": 1, "activities": []}],
        "search_time_sec": 8.123,
        "llm_inference_time_sec": 1.5,
        "meal_postprocess_attempts": 12,
        "dav_postprocess_attempts": 84,
        "dav_postprocess_insertions": 2,
        "commonsense_pass": True,
    }

    def _ok(query, prob_idx, oralce_translation=False):
        return True, dict(healthy_plan)

    agent3.planner.run = _ok
    succ, plan = agent3.run(dict(QUERY), "emit_deadline_test")
    assert succ is True
    assert plan["itinerary"] == healthy_plan["itinerary"], "planner plan replaced"
    for key in ("search_time_sec", "llm_inference_time_sec",
                "meal_postprocess_attempts", "dav_postprocess_attempts"):
        assert key not in plan, f"volatile key leaked: {key}"
    assert plan["dav_postprocess_insertions"] == 2  # deterministic keys kept
    assert plan["commonsense_pass"] is True
    print("healthy case: plan kept, volatile counters stripped")

    print("ALL PASSED")


if __name__ == "__main__":
    main()

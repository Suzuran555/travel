"""Validate the deterministic fallback plan against the STOCK evaluators
(schema + commonsense) for three different query shapes, and check that two
builds are byte-identical (determinism).

Run:  .venv/bin/python chinatravel/agent/tpc_agent_penguins/tests/test_fallback_plan.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from chinatravel.environment.world_env import WorldEnv                    # noqa: E402
from chinatravel.evaluation.utils import load_json_file                   # noqa: E402
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints  # noqa: E402
from chinatravel.evaluation.commonsense_constraint import (               # noqa: E402
    evaluate_commonsense_constraints,
)
from chinatravel.agent.tpc_agent_penguins.fallback_plan import build_fallback_plan  # noqa: E402

QUERIES = {
    "fb_short_pair": {
        "uid": "fb_short_pair",
        "days": 2,
        "people_number": 1,
        "start_city": "Beijing",
        "target_city": "Suzhou",
        "nature_language": "1 person, 2 days Beijing to Suzhou.",
        "hard_logic_py": [],
    },
    "fb_mid_pair": {
        "uid": "fb_mid_pair",
        "days": 3,
        "people_number": 2,
        "start_city": "Shanghai",
        "target_city": "Beijing",
        "nature_language": "2 people, 3 days Shanghai to Beijing.",
        "hard_logic_py": [],
    },
    "fb_long_pair": {
        "uid": "fb_long_pair",
        "days": 5,
        "people_number": 5,
        "start_city": "Guangzhou",
        "target_city": "Chengdu",
        "nature_language": "5 people, 5 days Guangzhou to Chengdu.",
        "hard_logic_py": [],
    },
}


def main():
    env = WorldEnv(lang="en")
    plans = {}
    for uid, q in QUERIES.items():
        plan = build_fallback_plan(dict(q), env)
        assert plan is not None, f"fallback build failed for {uid}"
        # must already be pure JSON (the agent round-trips anyway, but the
        # builder should not leak numpy scalars)
        plan = json.loads(json.dumps(plan, ensure_ascii=False))
        assert len(plan["itinerary"]) == q["days"]
        plans[uid] = plan

        # determinism: a second build is byte-identical
        plan2 = build_fallback_plan(dict(q), env)
        plan2 = json.loads(json.dumps(plan2, ensure_ascii=False))
        assert json.dumps(plan, sort_keys=True) == json.dumps(plan2, sort_keys=True), \
            f"fallback plan for {uid} is not deterministic"

    index = list(QUERIES.keys())
    schema = load_json_file(
        os.path.join(ROOT, "chinatravel/evaluation/output_schema.json"))
    _, _, schema_pass = evaluate_schema_constraints(index, plans, schema=schema)
    assert set(schema_pass) == set(index), \
        f"schema failures: {sorted(set(index) - set(schema_pass))}"
    print("schema: all 3 fallback plans pass")

    macro, micro, _, comm_pass = evaluate_commonsense_constraints(
        index, QUERIES, plans, verbose=False, lang="en")
    assert set(comm_pass) == set(index), \
        f"commonsense failures: {sorted(set(index) - set(comm_pass))} " \
        f"(macro={macro}, micro={micro})"
    print("commonsense: all 3 fallback plans pass")
    print("ALL PASSED")


if __name__ == "__main__":
    main()

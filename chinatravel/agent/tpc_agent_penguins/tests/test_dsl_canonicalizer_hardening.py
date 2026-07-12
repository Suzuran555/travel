"""Unit tests for the phase-2 hardening of dsl_canonicalizer:

  1. generic transport-budget guard repair (real/partial activity-type guards
     around pure transport-cost accumulations are removed),
  2. legacy pseudo-type ('transportation') guard repair still works,
  3. non-budget constraints with real type guards are NOT touched,
  4. deterministic content-keyed ordering of hard_logic_py.

Run:  .venv/bin/python chinatravel/agent/tpc_agent_penguins/tests/test_dsl_canonicalizer_hardening.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", ".."))

from chinatravel.agent.tpc_agent_penguins.dsl_canonicalizer import (   # noqa: E402
    canonicalize_hard_logic_py,
    canonicalize_query_hard_logic,
)
from chinatravel.agent.UrbanTrip import dsl_canonicalizer as twin      # noqa: E402


# the exact jitter-sensitive variant from uid 20250321030111150684
GUARDED_BUDGET = (
    "inner_city_transportation_cost=0\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner', "
    "'attraction', 'accommodation']:\n"
    "    inner_city_transportation_cost+="
    "innercity_transport_cost(activity_transports(activity))\n"
    "result=(inner_city_transportation_cost<=50.0)"
)
CANONICAL_BUDGET = (
    "inner_city_transportation_cost=0\n"
    "for activity in allactivities(plan):\n"
    "  inner_city_transportation_cost+="
    "innercity_transport_cost(activity_transports(activity))\n"
    "result=(inner_city_transportation_cost<=50.0)"
)

# inline single-type variant of the same category
GUARDED_BUDGET_INLINE = (
    "cost=0\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='attraction': "
    "cost+=innercity_transport_cost(activity_transports(activity))\n"
    "result=(cost<=100)"
)

# legacy pseudo-type guard (must still be repaired)
PSEUDO_GUARD = (
    "cost=0\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='transportation':\n"
    "    cost+=innercity_transport_cost(activity_transports(activity))\n"
    "result=(cost<=50)"
)

# must NOT be rewritten: real type guard whose body is not a pure
# transport accumulation (the benchmark tickets idiom)
TICKETS_CONSTRAINT = (
    "result=True\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['attraction', 'airplane', 'train'] "
    "and activity_tickets(activity)!=2: result=False\n"
    "  if innercity_transport_type(activity_transports(activity))=='metro' "
    "and metro_tickets(activity_transports(activity))!=2: result=False"
)

# must NOT be rewritten: transport-mode restriction guarded by real type
# (body is a predicate, not an accumulation)
MODE_CONSTRAINT = (
    "result=True\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='attraction':\n"
    "    result=result and "
    "innercity_transport_type(activity_transports(activity))=='metro'"
)

# must NOT be rewritten: real-type guard accumulating activity_cost
MEAL_BUDGET = (
    "restaurant_cost=0\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n"
    "    restaurant_cost+=activity_cost(activity)\n"
    "result=(restaurant_cost<=500)"
)


def test_guard_widening(mod):
    out = mod.canonicalize_hard_logic_py(GUARDED_BUDGET)
    assert out == CANONICAL_BUDGET, "partial-type budget guard not removed:\n" + out

    out = mod.canonicalize_hard_logic_py(GUARDED_BUDGET_INLINE)
    assert "if activity_type" not in out, "inline single-type budget guard kept:\n" + out
    assert "cost+=innercity_transport_cost(activity_transports(activity))" in out

    out = mod.canonicalize_hard_logic_py(PSEUDO_GUARD)
    assert "activity_type" not in out, "pseudo-type guard not removed:\n" + out

    assert mod.canonicalize_hard_logic_py(TICKETS_CONSTRAINT) == TICKETS_CONSTRAINT
    assert mod.canonicalize_hard_logic_py(MODE_CONSTRAINT) == MODE_CONSTRAINT
    assert mod.canonicalize_hard_logic_py(MEAL_BUDGET) == MEAL_BUDGET


def test_deterministic_order(mod):
    constraints = [
        "result=(people_count(plan)==2)",
        GUARDED_BUDGET,
        "result=(day_count(plan)==3)",
        TICKETS_CONSTRAINT,
    ]
    q1 = mod.canonicalize_query_hard_logic(
        {"hard_logic_py": list(constraints), "target_city": "Beijing"})
    q2 = mod.canonicalize_query_hard_logic(
        {"hard_logic_py": list(reversed(constraints)), "target_city": "Beijing"})
    q3 = mod.canonicalize_query_hard_logic(
        {"hard_logic_py": [constraints[2], constraints[0], constraints[3],
                           constraints[1]], "target_city": "Beijing"})
    assert q1["hard_logic_py"] == q2["hard_logic_py"] == q3["hard_logic_py"], \
        "constraint order not canonical"
    assert sorted(q1["hard_logic_py"]) == q1["hard_logic_py"]
    # content preserved (canonicalized budget + others)
    assert CANONICAL_BUDGET in q1["hard_logic_py"]
    assert len(q1["hard_logic_py"]) == 4


# the exact Qwen3.6-27B emission for uid h20241029143911770965: the type
# literal is mis-cased vs the Hangzhou EN DB ('University campus' vs
# 'university campus'), which starves planner-side POI selection
H965_TYPE_CONSTRAINT = (
    "attraction_type_set = set()\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='attraction':\n"
    "    attraction_type_set.add(attraction_type(activity, target_city(plan)))\n"
    "result=({'University campus'}<=attraction_type_set)"
)

POI_NAME_CONSTRAINT = (
    "attraction_name_set = set()\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='attraction':\n"
    "    attraction_name_set.add(activity_position(activity))\n"
    "result=({'West Lake'}<=attraction_name_set)"
)


def test_type_literal_db_normalization(mod):
    q = mod.canonicalize_query_hard_logic({
        "uid": "h20241029143911770965",
        "target_city": "Hangzhou",
        "hard_logic_py": [H965_TYPE_CONSTRAINT, POI_NAME_CONSTRAINT],
    })
    joined = "\n".join(q["hard_logic_py"])
    assert "'university campus'" in joined, \
        "type literal not folded onto DB spelling:\n" + joined
    assert "'University campus'" not in joined
    # POI names are never rewritten
    assert POI_NAME_CONSTRAINT in q["hard_logic_py"]


if __name__ == "__main__":
    import types
    pkg_mod = types.SimpleNamespace(
        canonicalize_hard_logic_py=canonicalize_hard_logic_py,
        canonicalize_query_hard_logic=canonicalize_query_hard_logic,
    )
    for name, mod in (("package", pkg_mod), ("main-tree twin", twin)):
        test_guard_widening(mod)
        test_deterministic_order(mod)
        test_type_literal_db_normalization(mod)
        print(f"[{name}] canonicalizer hardening tests passed")
    print("ALL PASSED")

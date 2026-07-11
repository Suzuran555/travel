"""Unit tests for the DSL canonicalizer and the widened V6 constraint extractor.

Run directly (no pytest needed):
    .venv/bin/python tests/test_dsl_canonicalizer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from chinatravel.agent.UrbanTrip.dsl_canonicalizer import (
    canonicalize_hard_logic_py,
    canonicalize_hard_logic_list,
    canonicalize_query_hard_logic,
)


# ---------------------------------------------------------------------------
# activity_price vs activity_cost: they are DIFFERENT DSL functions.
# ---------------------------------------------------------------------------

def test_price_and_cost_are_different_dsl_functions():
    from chinatravel.symbol_verification.concept_func import (
        activity_cost,
        activity_price,
    )

    # activity_price reads the per-person/room `price` field; activity_cost
    # reads the total `cost` field.  A canonicalizer must therefore never
    # rewrite one spelling into the other.
    activity = {"price": 100, "cost": 300, "tickets": 3}
    assert activity_price(activity) == 100
    assert activity_cost(activity) == 300


def test_canonicalizer_never_rewrites_price_into_cost():
    constraint = (
        "result=True\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction' and activity_price(activity)>0: result=False"
    )
    out = canonicalize_hard_logic_py(constraint)
    assert "activity_price(activity)" in out
    assert "activity_cost(activity)" not in out


# ---------------------------------------------------------------------------
# Set-variable renames (free locals only).
# ---------------------------------------------------------------------------

def test_set_variable_renames():
    constraint = (
        "attraction_names_set = set()\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction':\n"
        "    attraction_names_set.add(activity_position(activity))\n"
        "result=({'X'}<=attraction_names_set)"
    )
    out = canonicalize_hard_logic_py(constraint)
    assert "attraction_names_set" not in out
    assert out.count("attraction_name_set") == 3


def test_hotel_set_renamed_to_accommodation():
    constraint = (
        "hotel_names_set = set()\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='accommodation':\n"
        "    hotel_names_set.add(activity_position(activity))\n"
        "result=({'H'}<=hotel_names_set)"
    )
    out = canonicalize_hard_logic_py(constraint)
    assert "hotel_names_set" not in out
    assert "accommodation_name_set" in out


def test_rename_guard_when_canonical_name_already_used():
    constraint = (
        "attraction_name_set=set()\n"
        "attraction_names_set = set()\n"
        "result=({'X'}<=attraction_names_set)"
    )
    out = canonicalize_hard_logic_py(constraint)
    # canonical name already bound: rename must be skipped, not capture it
    assert out == constraint


# ---------------------------------------------------------------------------
# Budget accumulator renames.
# ---------------------------------------------------------------------------

def test_attraction_budget_accumulator_rename():
    constraint = (
        "sightseeing_cost=0\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction':\n"
        "    sightseeing_cost+=activity_cost(activity)\n"
        "result=(sightseeing_cost<=800.0)"
    )
    out = canonicalize_hard_logic_py(constraint)
    assert "sightseeing_cost" not in out
    assert "attraction_cost<=800.0" in out.replace(" ", "")


def test_meal_budget_accumulator_rename_single_line_filter():
    constraint = (
        "meal_cost=0\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']: meal_cost+=activity_cost(activity)\n"
        "result=(meal_cost<=500.0)"
    )
    out = canonicalize_hard_logic_py(constraint)
    assert "meal_cost" not in out
    assert "restaurant_cost" in out


def test_intercity_accumulator_named_total_cost_is_renamed():
    constraint = (
        "total_cost=0\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity) in ['train', 'airplane']:\n"
        "    total_cost+=activity_cost(activity)\n"
        "result=(total_cost<=2300.0)"
    )
    out = canonicalize_hard_logic_py(constraint)
    assert "total_cost" not in out
    assert "inter_city_transportation_cost" in out


def test_unfiltered_total_cost_is_kept():
    constraint = (
        "total_cost=0\n"
        "for activity in allactivities(plan): total_cost+=activity_cost(activity)+innercity_transport_cost(activity_transports(activity))\n"
        "result=(total_cost<=3300.0)"
    )
    assert canonicalize_hard_logic_py(constraint) == constraint


def test_oracle_two_line_total_cost_not_misclassified_as_innercity():
    # The oracle splits the total accumulation over two statements; the
    # classifier must combine all accumulation sites of a variable.
    constraint = (
        "total_cost=0 \n"
        "for activity in allactivities(plan):\n"
        "    total_cost+=activity_cost(activity)\n"
        "    total_cost += innercity_transport_cost(activity_transports(activity))\n"
        "result=(total_cost<=11100)"
    )
    assert canonicalize_hard_logic_py(constraint) == constraint


def test_innercity_transport_accumulator_rename():
    constraint = (
        "transport_cost=0\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='transportation':\n"
        "    transport_cost+=innercity_transport_cost(activity_transports(activity))\n"
        "result=(transport_cost<=20)"
    )
    import re

    out = canonicalize_hard_logic_py(constraint)
    assert "inner_city_transportation_cost" in out
    # the free local is renamed; the DSL function innercity_transport_cost(...)
    # must remain untouched
    assert not re.search(r"(?<![\w])transport_cost\b", out)
    assert "innercity_transport_cost(activity_transports(activity))" in out


def test_canonicalizer_is_idempotent_and_semantics_preserving():
    from chinatravel.symbol_verification.hard_constraint import (
        evaluate_constraints_py,
    )

    plan = {
        "people_number": 2,
        "start_city": "Beijing",
        "target_city": "Shanghai",
        "itinerary": [
            {
                "day": 1,
                "activities": [
                    {
                        "position": "Jin Mao Tower",
                        "type": "attraction",
                        "cost": 240,
                        "price": 120,
                        "tickets": 2,
                        "transports": [],
                    }
                ],
            }
        ],
    }
    constraints = [
        (
            "sightseeing_cost=0\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity)=='attraction':\n"
            "    sightseeing_cost+=activity_cost(activity)\n"
            "result=(sightseeing_cost<=800.0)"
        ),
        (
            "attraction_names_set = set()\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity)=='attraction':\n"
            "    attraction_names_set.add(activity_position(activity))\n"
            "result=({'Jin Mao Tower'}<=attraction_names_set)"
        ),
    ]
    canon = canonicalize_hard_logic_list(constraints)
    assert canon == canonicalize_hard_logic_list(canon)  # idempotent
    assert list(evaluate_constraints_py(constraints, plan, verbose=False)) == list(
        evaluate_constraints_py(canon, plan, verbose=False)
    )


def test_canonicalize_query_helper():
    query = {"hard_logic_py": ["result=({'X'}<=hotel_names_set)"]}
    canonicalize_query_hard_logic(query)
    assert query["hard_logic_py"] == ["result=({'X'}<=accommodation_name_set)"]


# ---------------------------------------------------------------------------
# Widened extractor (uses a stub memory, no environment data needed).
# ---------------------------------------------------------------------------

def _extract(constraints, attractions=(), restaurants=(), hotels=()):
    from chinatravel.agent.UrbanTrip.tpc_agent_optimized_v6 import (
        UrbanTripOptimizedV6,
    )

    class _Stub:
        pass

    stub = _Stub()
    stub.memory = {
        "attractions": pd.DataFrame({"name": list(attractions)}),
        "restaurants": pd.DataFrame({"name": list(restaurants)}),
        "accommodations": pd.DataFrame({"name": list(hotels)}),
    }
    query = {
        "hard_logic_py": list(constraints),
        "days": 2,
        "people_number": 2,
    }
    main, _per = UrbanTripOptimizedV6.extract_user_constraints_by_DSL(stub, query)
    return main


def test_extractor_negative_position_mention():
    constraint = (
        "result=True\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction' and activity_position(activity)=='Bad Park': result=False"
    )
    res = _extract([constraint], attractions=["Bad Park"])
    assert res.get("must_not_see_attraction") == ["Bad Park"]
    assert "Bad Park" not in (res.get("must_see_attraction") or [])


def test_extractor_flag_variable_exclusion_and_requirement():
    constraint = (
        "visited=False\n"
        "stayed=False\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction' and activity_position(activity)=='Temple':\n"
        "    visited=True\n"
        "  if activity_type(activity)=='accommodation' and activity_position(activity)=='Nice Hotel':\n"
        "    stayed=True\n"
        "result=not(visited) or stayed"
    )
    res = _extract([constraint], attractions=["Temple"], hotels=["Nice Hotel"])
    assert res.get("must_not_see_attraction") == ["Temple"]
    assert res.get("must_live_hotel") == ["Nice Hotel"]


def test_extractor_only_free_via_price_idiom_sets_budget_zero():
    constraint = (
        "result=True\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction' and activity_price(activity)>0: result=False"
    )
    res = _extract([constraint])
    assert res.get("only_free_attractions") is True
    assert res.get("attraction_budget") == 0.0


def test_extractor_per_activity_hotel_cap_maps_to_budget():
    constraint = (
        "result=True\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='accommodation' and activity_cost(activity)>2100.0: result=False"
    )
    res = _extract([constraint])
    assert res.get("hotel_budget") == 2100.0


def test_extractor_transport_pair_completion():
    res = _extract(["result=(intercity_transport_set=={'train'})"])
    assert res.get("must_depart_transport") == ["train"]
    assert res.get("must_not_depart_transport") == ["airplane"]
    assert res.get("must_return_transport") == ["train"]
    assert res.get("must_not_return_transport") == ["airplane"]

    res = _extract([
        "intercity_transport_set = set()\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity) in ['train', 'airplane']:\n"
        "    intercity_transport_set.add(activity_type(activity))\n"
        "result=not({'airplane'}&intercity_transport_set)"
    ])
    assert res.get("must_not_depart_transport") == ["airplane"]
    assert res.get("must_depart_transport") == ["train"]


def test_extractor_singleton_subset_counts_as_match_any():
    constraint = (
        "attraction_type_set = set()\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction':\n"
        "    attraction_type_set.add(attraction_type(activity, target_city(plan)))\n"
        "result=({'park'}<=attraction_type_set)"
    )
    res = _extract([constraint])
    assert res.get("must_see_attraction_type") == ["park"]
    assert res.get("attraction_type_match_any") is True

    multi = constraint.replace("{'park'}", "{'park', 'Museum'}")
    res = _extract([multi])
    assert res.get("attraction_type_match_any") is False


def test_extractor_membership_and_named_set_idioms():
    res = _extract([
        "attraction_name_set = set()\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction':\n"
        "    attraction_name_set.add(activity_position(activity))\n"
        "result=('Jiaozi Park' in attraction_name_set)"
    ], attractions=["Jiaozi Park"])
    assert res.get("must_see_attraction") == ["Jiaozi Park"]

    res = _extract([
        "forbidden_hotels={'Hotel A', 'Hotel B'}\n"
        "result=True\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='accommodation':\n"
        "    if activity_position(activity) in forbidden_hotels: result=False"
    ], hotels=["Hotel A", "Hotel B"])
    assert sorted(res.get("must_not_live_hotel") or []) == ["Hotel A", "Hotel B"]


def test_extractor_type_predicate_mentions():
    res = _extract([
        "result=True\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n"
        "    if restaurant_type(activity, target_city(plan))=='Snacks': result=False"
    ])
    assert res.get("must_not_visit_restaurant_type") == ["Snacks"]

    res = _extract([
        "result=True\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='accommodation' and accommodation_type(activity, target_city(plan))!='Free parking': result=False"
    ])
    assert res.get("must_live_hotel_feature") == ["Free parking"]


def test_extractor_sentinel_budget_dropped():
    res = _extract([
        "total_cost=0\n"
        "for activity in allactivities(plan): total_cost+=activity_cost(activity)+innercity_transport_cost(activity_transports(activity))\n"
        "result=(total_cost<=999999999)"
    ])
    assert res.get("overall_budget") is None


def test_extractor_reroutes_misfiled_poi():
    res = _extract([
        "restaurant_name_set = set()\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n"
        "    restaurant_name_set.add(activity_position(activity))\n"
        "result=({'Scenic Garden'}<=restaurant_name_set)"
    ], attractions=["Scenic Garden"])
    assert res.get("must_see_attraction") == ["Scenic Garden"]
    assert "Scenic Garden" not in (res.get("must_visit_restaurant") or [])


def main():
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

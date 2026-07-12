"""Unit tests for the round-4 coverage / span-grounding verifier
(chinatravel.agent.nesy_agent.constraint_coverage).

Every positive case is the NL of a real gen-probe failure; every negative
case is a real gen-probe NL where the rule must NOT fire.

Run directly (no pytest needed):
    .venv/bin/python tests/test_constraint_coverage.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chinatravel.agent.nesy_agent.constraint_coverage import (
    apply_category_expansion,
    apply_room_count_override,
    apply_taxi_convention,
    city_has_cuisine,
    detect_category_disjunction,
    detect_local_cuisine,
    drop_ungrounded_cost_caps,
    enforce_coverage,
    explicit_room_count,
    has_round_trip_flights,
    inject_airplane_transport,
    inject_local_cuisine,
    inject_overall_budget,
    is_human_register,
    taxi_cars_from_constraints,
)


TAXI_BOILER = (
    "result=True\nfor activity in allactivities(plan):\n"
    "  if innercity_transport_type(activity_transports(activity))=='taxi' "
    "and taxi_cars(activity_transports(activity))!={n}: result=False"
)
TICKETS_BOILER = (
    "result=True\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['attraction', 'airplane', 'train'] "
    "and activity_tickets(activity)!=5: result=False"
)
ROOMS_3 = (
    "result=True\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity)=='accommodation' and "
    "room_count(activity)!=3: result=False"
)


# ---------------------------------------------------------------------------
# c1: taxi_cars human convention
# ---------------------------------------------------------------------------

def test_taxi_convention_rewrites_scaled_count_for_human():
    cons, fix = apply_taxi_convention([TAXI_BOILER.format(n=2)], human=True)
    assert fix is not None
    assert "taxi_cars(activity_transports(activity))!=1" in cons[0]


def test_taxi_convention_untouched_for_generated_split():
    cons, fix = apply_taxi_convention([TAXI_BOILER.format(n=2)], human=False)
    assert fix is None and "!=2" in cons[0]


def test_taxi_convention_does_not_touch_tickets():
    cons, fix = apply_taxi_convention([TICKETS_BOILER], human=True)
    assert fix is None and "activity_tickets(activity)!=5" in cons[0]


def test_taxi_cars_from_constraints():
    assert taxi_cars_from_constraints([TAXI_BOILER.format(n=1)]) == 1
    assert taxi_cars_from_constraints([TAXI_BOILER.format(n=2)]) == 2
    assert taxi_cars_from_constraints([TICKETS_BOILER]) is None
    assert taxi_cars_from_constraints([]) is None


def test_is_human_register():
    assert is_human_register({"uid": "h20241029143736524841"})
    assert is_human_register({"uid": "x1", "tag": "human"})
    assert not is_human_register({"uid": "20250321002504225956"})


# ---------------------------------------------------------------------------
# d1: explicit room count override
# ---------------------------------------------------------------------------

def test_room_count_a_twin_room_means_one():
    nl = (
        "Our budget is 3000, and the three of us need to stay in a twin "
        "room."
    )
    assert explicit_room_count(nl) == 1
    cons, fix = apply_room_count_override([ROOMS_3], nl)
    assert fix is not None and "room_count(activity)!=1" in cons[0]


def test_room_count_zh_one_room():
    assert explicit_room_count("三个同学需要住在一间双床房") == 1
    assert explicit_room_count("我们要开两间标间") == 2


def test_room_count_two_rooms_english():
    assert explicit_room_count("we would like two rooms please") == 2


def test_room_count_no_explicit_mention():
    # 'a family-friendly room' has no room-type word chain -> no override
    nl = "I would like to stay in a family-friendly room with my son."
    assert explicit_room_count(nl) is None
    cons, fix = apply_room_count_override([ROOMS_3], nl)
    assert fix is None and cons == [ROOMS_3]


def test_room_count_never_invents_constraint():
    cons, fix = apply_room_count_override(
        ["result=(day_count(plan)==2)"], "we need one room"
    )
    assert fix is None and cons == ["result=(day_count(plan)==2)"]


# ---------------------------------------------------------------------------
# b1: local-specialties -> signature cuisine
# ---------------------------------------------------------------------------

def test_local_cuisine_detection_probe_phrasings():
    cases = [
        ("... visit many famous local attractions and taste the local "
         "specialties.", "Suzhou", "Jiangsu-Zhejiang cuisine"),
        ("... experience as much local cuisine as possible.", "Wuhan",
         "Hubei cuisine"),
        ("... experience the local customs and culture, and taste the "
         "local cuisine.", "Chengdu", "Sichuan cuisine"),
        ("We all enjoy spicy food and would like to try some local "
         "specialties.", "Chengdu", "Sichuan cuisine"),
    ]
    for nl, city, want in cases:
        assert detect_local_cuisine(nl, city, "en") == want, (nl, city)


def test_local_cuisine_not_triggered_by_bare_food():
    # ...714886: oracle has NO cuisine constraint for this phrasing
    nl = "We enjoy fashion and food, and we want to visit some popular spots."
    assert detect_local_cuisine(nl, "Shanghai", "en") is None
    # hot-pot request without 'local' phrasing must not add Sichuan cuisine
    nl2 = "We want to try the famous hot pot in Chongqing and explore some spots."
    assert detect_local_cuisine(nl2, "Chongqing", "en") is None


def test_local_cuisine_injection_and_idempotence():
    q = {
        "nature_language": "We want to taste the local specialties.",
        "target_city": "Suzhou",
    }
    cons, fix = inject_local_cuisine([], q, "en")
    assert fix is not None and "'Jiangsu-Zhejiang cuisine'" in cons[-1]
    cons2, fix2 = inject_local_cuisine(cons, q, "en")
    assert fix2 is None and cons2 == cons


def test_local_cuisine_zh():
    assert detect_local_cuisine("想尝尝当地特色美食", "成都", "zh") == "川菜"
    assert detect_local_cuisine("想吃美食", "成都", "zh") is None
    assert city_has_cuisine("Chengdu", "川菜", "zh")


# ---------------------------------------------------------------------------
# b2: airfare -> airplane intercity transport
# ---------------------------------------------------------------------------

def test_airfare_injects_airplane_when_route_flies():
    q = {
        "nature_language": (
            "Excluding round-trip airfare, our budget is around 10,000 RMB."
        ),
        "start_city": "Guangzhou",
        "target_city": "Shanghai",
    }
    cons, fix = inject_airplane_transport([], q, "en")
    assert fix is not None
    assert "intercity_transport_set=={'airplane'}" in cons[-1]


def test_airfare_not_injected_without_flights():
    # Suzhou has no airport: airplane-only would be locally unsatisfiable
    assert not has_round_trip_flights("Beijing", "Suzhou", "en")
    q = {
        "nature_language": "The air tickets should be cheap.",
        "start_city": "Beijing",
        "target_city": "Suzhou",
    }
    cons, fix = inject_airplane_transport([], q, "en")
    assert fix is None and cons == []


def test_airfare_respects_existing_intercity_constraint():
    train = (
        "intercity_transport_set = set()\nfor activity in allactivities(plan):\n"
        "  if activity_type(activity) in ['train', 'airplane']:\n"
        "    intercity_transport_set.add(activity_type(activity))\n"
        "result=(intercity_transport_set=={'train'})"
    )
    q = {
        "nature_language": "buy plane tickets",
        "start_city": "Guangzhou",
        "target_city": "Shanghai",
    }
    cons, fix = inject_airplane_transport([train], q, "en")
    assert fix is None and cons == [train]


def test_airfare_not_triggered_without_ticket_words():
    q = {
        "nature_language": "We plan to travel by high-speed train round trip.",
        "start_city": "Nanjing",
        "target_city": "Chengdu",
    }
    cons, fix = inject_airplane_transport([], q, "en")
    assert fix is None


# ---------------------------------------------------------------------------
# d2: category disjunction expansion
# ---------------------------------------------------------------------------

def test_category_disjunction_probe_case():
    nl = (
        "We want to try the famous hot pot in Chongqing and explore some "
        "historical or scenic spots, preferably somewhere cool."
    )
    req = detect_category_disjunction(nl, "en")
    assert req == {"Cultural Tourism Area", "historical site",
                   "natural scenery"}


def test_category_disjunction_zh():
    req = detect_category_disjunction("再去看看历史文化或者风景名胜", "zh")
    assert req == {"文化旅游区", "历史古迹", "自然风光"}


def test_category_disjunction_not_on_conjunction():
    # ...282919: 'historical and cultural attractions, such as museums'
    # (no 'or' between two category words) -> museum handling only
    nl = (
        "We hope to experience more historical and cultural attractions, "
        "such as museums."
    )
    assert detect_category_disjunction(nl, "en") is None


def test_category_expansion_replaces_weakened_translation():
    weak = (
        "attraction_type_set = set()\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction':\n"
        "    attraction_type_set.add(attraction_type(activity, target_city(plan)))\n"
        "result=({'historical site', 'natural scenery'}&attraction_type_set)"
    )
    nl = "explore some historical or scenic spots"
    cons, fix = apply_category_expansion([weak], nl, "en")
    assert fix is not None and weak in fix["removed"]
    assert len(cons) == 1
    final = cons[0]
    assert "<=attraction_type_set" in final
    for t in ("Cultural Tourism Area", "historical site", "natural scenery"):
        assert "'%s'" % t in final


def test_category_expansion_keeps_unrelated_type_constraints():
    museum = (
        "attraction_type_set = set()\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction':\n"
        "    attraction_type_set.add(attraction_type(activity, target_city(plan)))\n"
        "result=({'Museum/Memorial Hall'}<=attraction_type_set)"
    )
    nl = "explore some historical or scenic spots"
    cons, fix = apply_category_expansion([museum], nl, "en")
    assert fix is not None and museum in cons


# ---------------------------------------------------------------------------
# budget span-grounding + coverage
# ---------------------------------------------------------------------------

def test_ungrounded_cost_cap_dropped():
    # NL with no money wording at all: the cap is invented -> dropped
    cap = (
        "total_cost=0\nfor activity in allactivities(plan): "
        "total_cost+=activity_cost(activity)"
        "+innercity_transport_cost(activity_transports(activity))\n"
        "result=(total_cost<=1000000)"
    )
    q = {"nature_language": "Are there any good restaurants nearby?",
         "people_number": 2, "days": 3}
    cons, fix = drop_ungrounded_cost_caps([cap], q)
    assert fix is not None and cons == []


def test_ungrounded_cap_with_money_talk_only_flagged():
    # probe ...617975256: 'a budget of three thousand' must NOT be dropped
    cap = (
        "total_cost=0\nfor activity in allactivities(plan): "
        "total_cost+=activity_cost(activity)"
        "+innercity_transport_cost(activity_transports(activity))\n"
        "result=(total_cost<=3000)"
    )
    q = {"nature_language": "with a budget of three thousand",
         "people_number": 3, "days": 2}
    cons, fix = drop_ungrounded_cost_caps([cap], q)
    # 'three thousand' is parsed -> grounded -> kept without even a flag
    assert fix is None and cons == [cap]
    # money wording present but number unparseable -> kept, flag only
    q2 = {"nature_language": "we have a modest budget", "people_number": 3,
          "days": 2}
    cons2, fix2 = drop_ungrounded_cost_caps([cap], q2)
    assert cons2 == [cap]
    assert fix2 is not None and fix2.get("flag_only")


def test_taxi_convention_rewrites_symbolic_formula():
    sym = (
        "result=True\nfor activity in allactivities(plan):\n"
        "  transports = activity_transports(activity)\n"
        "  if transports:\n"
        "    if innercity_transport_type(transports) == 'taxi':\n"
        "      expected_cars = (people_count(plan) + 3) // 4\n"
        "      if taxi_cars(transports) != expected_cars:\n"
        "        result = False"
    )
    cons, fix = apply_taxi_convention([sym], human=True)
    assert fix is not None
    assert "expected_cars = 1" in cons[0]
    assert "people_count(plan) + 3" not in cons[0]
    # planner-side parse resolves the symbolic form to 1
    assert taxi_cars_from_constraints(cons, people_number=5) == 1
    # and the unrewritten generated-split form resolves to the formula value
    assert taxi_cars_from_constraints([sym], people_number=5) == 2


def test_grounded_cost_cap_kept_with_scaling_and_units():
    cap = (
        "total_cost=0\nfor activity in allactivities(plan): "
        "total_cost+=activity_cost(activity)"
        "+innercity_transport_cost(activity_transports(activity))\n"
        "result=(total_cost<=6000)"
    )
    # per-person budget 3000 x 2 people = 6000; also '一万' style units
    q = {"nature_language": "a budget of 3,000 per person", "people_number": 2,
         "days": 5}
    cons, fix = drop_ungrounded_cost_caps([cap], q)
    assert fix is None and cons == [cap]


def test_poi_price_constraint_never_dropped_as_budget():
    named = (
        "result=False\nfor activity in allactivities(plan):\n"
        "  if activity_position(activity)=='The Palace Museum' and "
        "activity_price(activity)<=60: result=True"
    )
    q = {"nature_language": "visit the Palace Museum", "people_number": 1,
         "days": 1}
    cons, fix = drop_ungrounded_cost_caps([named], q)
    assert fix is None and cons == [named]


def test_overall_budget_injection():
    q = {
        "nature_language": "Two people for three days, with a budget of "
                           "around 3000.",
        "people_number": 2,
        "days": 3,
    }
    cons, fix = inject_overall_budget([], q)
    assert fix is not None and "total_cost<=3000" in cons[-1]


def test_overall_budget_covered_not_reinjected():
    cap = (
        "total_cost=0\nfor activity in allactivities(plan): "
        "total_cost+=activity_cost(activity)"
        "+innercity_transport_cost(activity_transports(activity))\n"
        "result=(total_cost<=3000)"
    )
    q = {
        "nature_language": "with a budget of around 3000",
        "people_number": 2,
        "days": 3,
    }
    cons, fix = inject_overall_budget([cap], q)
    assert fix is None and cons == [cap]


def test_qualified_budget_only_flagged():
    q = {
        "nature_language": "Our accommodation budget is 3300 yuan.",
        "people_number": 2,
        "days": 3,
    }
    cons, fix = inject_overall_budget([], q)
    assert cons == []
    assert fix is not None and fix.get("flag_only")


# ---------------------------------------------------------------------------
# end-to-end driver
# ---------------------------------------------------------------------------

def test_enforce_coverage_probe_524841_taxi():
    q = {
        "uid": "h20241029143736524841",
        "tag": "human",
        "nature_language": (
            "[Current location: Nanjing, Destination: Chengdu, Number of "
            "travelers: 5, Number of travel days: 4] Our family of five "
            "plans to visit Chengdu to see the pandas. We intend to travel "
            "by high-speed train round trip and also visit some famous "
            "nearby attractions. Please help us plan the itinerary, and "
            "keep the budget under 20,000 RMB."
        ),
        "people_number": 5,
        "days": 4,
        "start_city": "Nanjing",
        "target_city": "Chengdu",
        "hard_logic_py": [TAXI_BOILER.format(n=2)],
    }
    enforce_coverage(q, lang="en")
    assert any("taxi_cars(activity_transports(activity))!=1" in c
               for c in q["hard_logic_py"])
    rules = {f["rule"] for f in q.get("coverage_fixes", [])}
    assert "taxi_cars_human_convention" in rules
    # budget 20,000 uncovered -> canonical overall cap injected
    assert any("total_cost<=20000" in c for c in q["hard_logic_py"])


def test_enforce_coverage_idempotent():
    q = {
        "uid": "h1",
        "tag": "human",
        "nature_language": "We want to taste the local specialties.",
        "people_number": 2,
        "days": 2,
        "start_city": "Beijing",
        "target_city": "Suzhou",
        "hard_logic_py": [TAXI_BOILER.format(n=1)],
    }
    enforce_coverage(q, lang="en")
    first = list(q["hard_logic_py"])
    n_fixes = len(q.get("coverage_fixes", []))
    enforce_coverage(q, lang="en")
    assert q["hard_logic_py"] == first
    assert len(q.get("coverage_fixes", [])) == n_fixes


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

"""Unit tests for the V6 evening-window repair of REQUIRED attractions
(_postprocess_repair_required_evening_pois and helpers).

Motivating case: uid h20241029143911770965 (gen_probe) requires attraction
type 'university campus'; Hangzhou's only two campus POIs open 19:00-21/22:00,
but the planner checks into the hotel ~13:51 and ships a plan violating its
own hard constraint.  The repair must slot the required POI between the last
day activity and the movable hotel check-in.

Covered behaviors:
  * inserts the required type inside its open window (waits for opening),
    dwell >= 30 min, hotel check-in moved to the new arrival (< 24:00);
  * respects budget: an insert that would flip a passing hard budget line
    to fail is rejected (monotone hard-vector acceptance);
  * respects the POI open window / midnight: infeasible windows insert nothing;
  * does NOT fire when constraints are already satisfied;
  * does NOT fire for gaps unrelated to required attractions
    (soft-metric-only / budget-only failures);
  * honors must_see_attraction_type_match_any (any-of) semantics.

Run directly (no pytest needed):
    .venv/bin/python tests/test_evening_required_repair.py
Mirror check against the package twin:
    V6_IMPL=penguins .venv/bin/python tests/test_evening_required_repair.py
"""
import os
import sys
import time
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

if os.environ.get("V6_IMPL") == "penguins":
    import chinatravel.agent.tpc_agent_penguins.v6 as v6mod
    from chinatravel.agent.tpc_agent_penguins.v6 import UrbanTripOptimizedV6
else:
    import chinatravel.agent.UrbanTrip.tpc_agent_optimized_v6 as v6mod
    from chinatravel.agent.UrbanTrip.tpc_agent_optimized_v6 import UrbanTripOptimizedV6

from chinatravel.agent.UrbanTrip.utils import add_time_delta, time_to_minutes


CAMPUS = "university campus"

ATTRS = pd.DataFrame(
    [
        {"name": "West Lake", "type": "park", "price": 0,
         "opentime": "08:00", "endtime": "18:00", "lat": 30.250, "lon": 120.150},
        {"name": "Campus A", "type": CAMPUS, "price": 0,
         "opentime": "19:00", "endtime": "21:00", "lat": 30.255, "lon": 120.155},
        {"name": "Campus B", "type": CAMPUS, "price": 500,
         "opentime": "19:00", "endtime": "22:00", "lat": 30.300, "lon": 120.200},
        {"name": "Gallery G", "type": "art gallery", "price": 0,
         "opentime": "09:00", "endtime": "17:00", "lat": 30.245, "lon": 120.145},
    ]
)
HOTELS = pd.DataFrame([{"name": "Hotel H", "lat": 30.240, "lon": 120.140}])
RESTS = pd.DataFrame(columns=["name", "lat", "lon"])

QUERY = {
    "people_number": 2,
    "start_city": "Shanghai",
    "target_city": "Hangzhou",
    "days": 2,
    "hard_logic_py": [],
}


def make_agent(travel_minutes=20, taxi_cost=15, must_type=(CAMPUS,),
               match_any=False, attrs=ATTRS):
    """UrbanTripOptimizedV6 stub with only the state the repair path touches."""
    a = UrbanTripOptimizedV6.__new__(UrbanTripOptimizedV6)
    a.memory = {
        "attractions": attrs.copy(),
        "restaurants": RESTS.copy(),
        "accommodations": HOTELS.copy(),
    }
    a.must_see_attraction = None
    a.must_see_attraction_type = list(must_type) if must_type else None
    a.must_not_see_attraction = None
    a.must_not_see_attraction_type = None
    a.only_free_attractions = None
    a.must_see_attraction_type_match_any = match_any
    a.activities_arrive_time_dict = {}
    a.activities_leave_time_dict = {}
    a.activities_stay_time_dict = None
    a.too_many_backtrack = False
    a.innercity_transports_ranking = ["taxi"]
    a.transport_rules_by_distance = None
    a._candidate_static_cache = {}
    a.enable_evening_required_repair = True
    a.evening_repair_seconds = 5.0
    a.evening_repair_max_attempts = 20
    a.evening_repair_candidates = 8
    a.evening_repair_visit_minutes = 30
    a.time_before_search = time.time()
    a.TIME_CUT = 300
    a.debug = False
    a.transport_calls = []

    def collect(city, start, end, start_time, mode, _a=a):
        _a.transport_calls.append((start, end, start_time, mode))
        return [{
            "start": start, "end": end, "mode": "taxi",
            "start_time": start_time,
            "end_time": add_time_delta(start_time, travel_minutes),
            "cost": taxi_cost, "price": taxi_cost, "cars": 1, "distance": 3.0,
        }]

    a.collect_innercity_transport = collect
    a._commonsense_passes = lambda q, p: True
    return a


def install_fake_hard(agent, budget=None):
    """Fake DSL evaluator: [campus-type present] (+ optional [cost<=budget])."""
    campus_canon = agent._canon_type("attraction", CAMPUS)

    def results(query, plan):
        present = campus_canon in agent._plan_attraction_canon_types(
            plan.get("itinerary")
        )
        out = [present]
        if budget is not None:
            out.append(agent._plan_total_cost(plan) <= budget)
        return out

    agent._hard_logic_results = results
    agent._plan_passes_official_constraints = lambda q, p: (
        bool(p.get("itinerary")) and all(results(q, p))
    )
    return results


def base_plan(extra_attraction=None):
    """Day template that mirrors the failing uid: sightseeing, then a 13:51
    hotel check-in that runs to 24:00 (movable start)."""
    acts = [
        {"position": "West Lake", "type": "attraction", "price": 0, "cost": 0,
         "tickets": 2, "start_time": "10:00", "end_time": "13:30",
         "transports": []},
    ]
    if extra_attraction is not None:
        acts.append(dict(extra_attraction))
    acts.append(
        {"position": "Hotel H", "type": "accommodation", "price": 150,
         "cost": 300, "rooms": 2, "room_type": 1,
         "start_time": "13:51", "end_time": "24:00",
         "transports": [{"start": acts[-1]["position"], "end": "Hotel H",
                         "mode": "taxi", "start_time": "13:30",
                         "end_time": "13:51", "cost": 0, "price": 0,
                         "cars": 1, "distance": 2.0}]},
    )
    return {"itinerary": [{"day": 1, "activities": acts}]}


def run_repair(agent, plan):
    """Call the driver with repair_full_itinerary stubbed out (it needs the
    real city DB); acceptance still goes through the patched commonsense +
    hard-vector gates."""
    orig = v6mod.repair_full_itinerary
    v6mod.repair_full_itinerary = lambda a, q, it: True
    try:
        return agent._postprocess_repair_required_evening_pois(QUERY, deepcopy(plan))
    finally:
        v6mod.repair_full_itinerary = orig


# ---------------------------------------------------------------- tests ----

def test_inserts_required_type_in_evening_window():
    agent = make_agent()
    install_fake_hard(agent)
    plan = base_plan()
    out = run_repair(agent, plan)

    assert out.get("evening_repair_insertions") == 1, out
    acts = out["itinerary"][0]["activities"]
    assert len(acts) == 3
    inserted = acts[1]
    # free + nearest campus wins, visit waits for the 19:00 opening
    assert inserted["position"] == "Campus A"
    assert inserted["type"] == "attraction"
    assert inserted["start_time"] == "19:00"
    open_row = ATTRS[ATTRS["name"] == "Campus A"].iloc[0]
    assert time_to_minutes(inserted["start_time"]) >= time_to_minutes(open_row["opentime"])
    assert time_to_minutes(inserted["end_time"]) <= time_to_minutes(open_row["endtime"])
    dwell = time_to_minutes(inserted["end_time"]) - time_to_minutes(inserted["start_time"])
    assert dwell >= 30
    assert inserted["tickets"] == 2 and inserted["cost"] == 0
    # transports rebuilt on both sides of the insert
    assert inserted["transports"][0]["start"] == "West Lake"
    assert inserted["transports"][-1]["end"] == "Campus A"
    hotel = acts[2]
    assert hotel["type"] == "accommodation"
    assert hotel["transports"][0]["start"] == "Campus A"
    assert hotel["transports"][-1]["end"] == "Hotel H"
    # movable check-in: start shifted to the new arrival, before midnight
    assert hotel["start_time"] == hotel["transports"][-1]["end_time"]
    assert time_to_minutes(hotel["start_time"]) < time_to_minutes("24:00")
    assert hotel["end_time"] == "24:00"
    # the hard requirement is now discharged
    names, types = agent._missing_required_attraction_items(out["itinerary"])
    assert not names and not types
    assert agent._plan_passes_official_constraints(QUERY, out)


def test_budget_guard_rejects_costly_insert():
    # current plan: cost 300, budget 310 -> budget line passes now, but ANY
    # insert (2 taxi legs @15 = +30, or Campus B tickets +1000) flips it to
    # fail; the monotone acceptance must reject every candidate.
    agent = make_agent()
    install_fake_hard(agent, budget=310)
    plan = base_plan()
    out = run_repair(agent, plan)
    assert "evening_repair_insertions" not in out
    assert out == plan
    # with headroom the same setup inserts the free campus
    agent2 = make_agent()
    install_fake_hard(agent2, budget=1000)
    out2 = run_repair(agent2, plan)
    assert out2.get("evening_repair_insertions") == 1
    assert agent2._plan_total_cost(out2) <= 1000


def test_respects_open_window_minimum_dwell():
    # only campus closes 19 minutes after it opens late at night: the visit
    # cannot reach the 30-min dwell inside the window -> nothing inserted
    attrs = ATTRS[ATTRS["name"] != "Campus B"].copy()
    attrs.loc[attrs["name"] == "Campus A", "opentime"] = "23:40"
    attrs.loc[attrs["name"] == "Campus A", "endtime"] = "23:59"
    agent = make_agent(attrs=attrs)
    install_fake_hard(agent)
    plan = base_plan()
    out = run_repair(agent, plan)
    assert "evening_repair_insertions" not in out
    assert out == plan


def test_rejects_return_to_hotel_past_midnight():
    # visit fits (22:00-22:30) but the 100-min ride back reaches the hotel at
    # 24:10 -> insert rejected, plan unchanged
    attrs = ATTRS[ATTRS["name"] != "Campus B"].copy()
    attrs.loc[attrs["name"] == "Campus A", "opentime"] = "22:00"
    attrs.loc[attrs["name"] == "Campus A", "endtime"] = "23:59"
    agent = make_agent(travel_minutes=100)
    agent.memory["attractions"] = attrs
    install_fake_hard(agent)
    plan = base_plan()
    out = run_repair(agent, plan)
    assert "evening_repair_insertions" not in out
    assert out == plan


def test_does_not_fire_when_constraints_satisfied():
    agent = make_agent()
    install_fake_hard(agent)
    plan = base_plan(extra_attraction={
        "position": "Campus B", "type": "attraction", "price": 500,
        "cost": 1000, "tickets": 2, "start_time": "19:00",
        "end_time": "19:30", "transports": []})
    assert agent._plan_passes_official_constraints(QUERY, plan)
    out = run_repair(agent, plan)
    assert out == plan
    assert agent.transport_calls == []  # never even probed the env


def test_does_not_fire_for_unrelated_or_soft_gaps():
    # (a) hard logic fails for a NON-attraction reason (budget) while the
    # required type is already present -> repair must not touch the plan
    agent = make_agent()
    install_fake_hard(agent, budget=0)  # budget line always fails
    plan = base_plan(extra_attraction={
        "position": "Campus B", "type": "attraction", "price": 500,
        "cost": 1000, "tickets": 2, "start_time": "19:00",
        "end_time": "19:30", "transports": []})
    out = run_repair(agent, plan)
    assert out == plan
    assert agent.transport_calls == []
    # (b) no required attraction constraints at all (a pure soft-metric gap
    # cannot create missing names/types) -> repair must not fire either
    agent2 = make_agent(must_type=None)
    agent2._hard_logic_results = lambda q, p: [False]
    agent2._plan_passes_official_constraints = lambda q, p: False
    agent2._commonsense_passes = lambda q, p: True
    out2 = run_repair(agent2, base_plan())
    assert out2 == base_plan()
    assert agent2.transport_calls == []


def test_match_any_semantics():
    # any-of and one listed type already present -> satisfied, no fire
    agent = make_agent(must_type=(CAMPUS, "art gallery"), match_any=True)
    install_fake_hard(agent)
    plan = base_plan(extra_attraction={
        "position": "Gallery G", "type": "attraction", "price": 0,
        "cost": 0, "tickets": 2, "start_time": "15:00",
        "end_time": "16:00", "transports": []})
    names, types = agent._missing_required_attraction_items(plan["itinerary"])
    assert not names and not types
    # any-of with NONE present -> one insert discharges the whole requirement
    agent2 = make_agent(must_type=(CAMPUS, "art gallery"), match_any=True)
    install_fake_hard(agent2)
    out = run_repair(agent2, base_plan())
    assert out.get("evening_repair_insertions") == 1
    names2, types2 = agent2._missing_required_attraction_items(out["itinerary"])
    assert not names2 and not types2


def test_missing_must_see_name_is_repaired():
    agent = make_agent(must_type=None)
    agent.must_see_attraction = ["Campus B"]
    campus_b = "Campus B"

    def results(query, plan):
        return [campus_b in agent._plan_positions(plan.get("itinerary"))]

    agent._hard_logic_results = results
    agent._plan_passes_official_constraints = lambda q, p: all(results(q, p))
    out = run_repair(agent, base_plan())
    assert out.get("evening_repair_insertions") == 1
    inserted = out["itinerary"][0]["activities"][1]
    assert inserted["position"] == "Campus B"
    assert inserted["start_time"] == "19:00"
    assert inserted["cost"] == 1000  # 500 x 2 people


TESTS = [
    test_inserts_required_type_in_evening_window,
    test_budget_guard_rejects_costly_insert,
    test_respects_open_window_minimum_dwell,
    test_rejects_return_to_hotel_past_midnight,
    test_does_not_fire_when_constraints_satisfied,
    test_does_not_fire_for_unrelated_or_soft_gaps,
    test_match_any_semantics,
    test_missing_must_see_name_is_repaired,
]


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    impl = "tpc_agent_penguins.v6" if os.environ.get("V6_IMPL") == "penguins" else "UrbanTrip.tpc_agent_optimized_v6"
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed against {impl}")
    sys.exit(1 if failed else 0)

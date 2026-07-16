"""Phase-2 style adaptation tests (round 6, no LLM backend needed):

  1. window extraction: containment (phase-2) and covering (phase-1)
     dialects, type-guarded and guard-less spellings, escaped apostrophes,
     and the mixed-dialect cross-constraint contamination regression;
  2. _scheduled_poi_times pins two-ended windows onto [A, B] (satisfies both
     comparator dialects) and honors the containment tags;
  3. inline type bans / transport whitelists / intercity both-modes shapes;
  4. coverage rules: between-window injection, inner-city mode-line
     injection, numbered-requirements flag (all idempotent);
  5. crash guards: geodesic on junk coordinates, hotel branch without
     accommodation (1-day trips).

Run:  .venv/bin/python chinatravel/agent/tpc_agent_penguins/tests/test_phase2_style.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd  # noqa: E402

from chinatravel.agent.tpc_agent_penguins.v6 import UrbanTripOptimizedV6  # noqa: E402
from chinatravel.agent.tpc_agent_penguins import constraint_coverage as cc  # noqa: E402


def _bare_agent():
    agent = object.__new__(UrbanTripOptimizedV6)
    agent.memory = {
        "attractions": pd.DataFrame({"name": ["Shence Gate", "Bear Grandma's Garden"]}),
        "restaurants": pd.DataFrame({"name": ["Shengli River Food Street"]}),
        "accommodations": pd.DataFrame({"name": ["Qiuguo S Hotel (Beijing Capital Airport Second Branch)"]}),
    }
    return agent


CONTAIN_POSONLY = (
    "result=False\nfor activity in allactivities(plan):\n"
    "  if activity_position(activity)=='Shengli River Food Street':\n"
    "    if activity_start_time(activity)>='15:47' and activity_end_time(activity)<='17:32': result=True"
)
COVER_POSONLY = (
    "result=False\nfor activity in allactivities(plan):\n"
    "  if activity_position(activity)=='Shence Gate':\n"
    "    if activity_start_time(activity)<='15:34' and activity_end_time(activity)>='17:05': result=True"
)
CONTAIN_GUARD_MEAL = (
    "result=False\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner'] and "
    "activity_position(activity)==\"Guangzhou Junye Hotel\":\n"
    "    if activity_start_time(activity)>=\"07:50\" and activity_end_time(activity)<=\"08:45\": result=True"
)
COVER_GUARD_ACC = (
    "result=False\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity)=='accommodation' and "
    "activity_position(activity)=='Qiuguo S Hotel (Beijing Capital Airport Second Branch)':\n"
    "    if activity_start_time(activity)<='22:29' and activity_end_time(activity)>='24:00': result=True"
)
CONTAIN_APOSTROPHE = (
    "result=False\nfor activity in allactivities(plan):\n"
    "  if activity_position(activity)=='Bear Grandma\\'s Garden':\n"
    "    if activity_start_time(activity)>='17:57' and activity_end_time(activity)<='19:30': result=True"
)
INLINE_TYPE_BAN = (
    "result=True\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity)=='attraction' and "
    "attraction_type(activity, target_city(plan)) in ['Other', 'Red tourism sites']: result=False"
)
WHITELIST_CANON = (
    "inner_city_transportation_set=set()\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity)=='transportation': "
    "inner_city_transportation_set.add(activity_position(activity))\n"
    "result=(inner_city_transportation_set<={'metro', 'taxi'})"
)
INTERCITY_BOTH_EQ = (
    "intercity_transport_set=set()\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['train', 'airplane']: "
    "intercity_transport_set.add(intercity_transport_type(activity))\n"
    "result=(intercity_transport_set=={'airplane','train'})"
)
INTERCITY_BOTH_SUP = (
    "intercity_transport_set=set()\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['train', 'airplane']: "
    "intercity_transport_set.add(intercity_transport_type(activity))\n"
    "result=({'airplane', 'train'}<=intercity_transport_set)"
)


def test_window_extraction():
    agent = _bare_agent()
    res, _ = agent.extract_user_constraints_by_DSL({"hard_logic_py": [
        CONTAIN_POSONLY, COVER_POSONLY, CONTAIN_GUARD_MEAL, COVER_GUARD_ACC,
        CONTAIN_APOSTROPHE,
    ]})
    wins = res["activities_window_dict"]
    assert wins["Shengli River Food Street"] == {
        "window": ["15:47", "17:32"], "semantics": "within", "kind": None}
    assert wins["Shence Gate"]["semantics"] == "cover"
    assert wins["Guangzhou Junye Hotel"] == {
        "window": ["07:50", "08:45"], "semantics": "within", "kind": "meal"}
    assert wins["Qiuguo S Hotel (Beijing Capital Airport Second Branch)"]["kind"] == "accommodation"
    assert wins["Bear Grandma's Garden"]["window"] == ["17:57", "19:30"]

    arrive = res["activities_arrive_time_dict"]
    leave = res["activities_leave_time_dict"]
    # containment tags
    assert arrive["Shengli River Food Street"] == ["late", "15:47"]
    assert leave["Shengli River Food Street"] == ["early", "17:32"]
    # covering keeps the legacy phase-1 tags
    assert arrive["Shence Gate"] == ["early", "15:34"]
    assert leave["Shence Gate"] == ["late", "17:05"]
    # mixed-dialect regression: no cross-constraint contamination -- the
    # containment POI must NOT pick up the covering constraint's literals
    assert arrive["Shengli River Food Street"][1] != "15:34"
    # windowed POIs are promoted to must-sets by DB kind
    assert "Shence Gate" in res["must_see_attraction"]
    assert "Shengli River Food Street" in res["must_visit_restaurant"]
    print("test_window_extraction OK")


def test_nonstandard_shapes():
    agent = _bare_agent()
    res, _ = agent.extract_user_constraints_by_DSL({"hard_logic_py": [
        INLINE_TYPE_BAN, WHITELIST_CANON, INTERCITY_BOTH_EQ,
    ]})
    assert res["must_not_see_attraction_type"] == ["Other", "Red tourism sites"]
    assert sorted(res["must_innercity_transport"]) == ["metro", "taxi"]
    assert res["must_intercity_transport_all"] == ["airplane", "train"]
    assert res.get("must_depart_transport") is None  # split happens planner-side

    agent = _bare_agent()
    res, _ = agent.extract_user_constraints_by_DSL(
        {"hard_logic_py": [INTERCITY_BOTH_SUP]})
    assert res["must_intercity_transport_all"] == ["airplane", "train"]

    # single-mode superset: at least one leg with that mode
    agent = _bare_agent()
    res, _ = agent.extract_user_constraints_by_DSL({"hard_logic_py": [
        INTERCITY_BOTH_SUP.replace("{'airplane', 'train'}", "{'train'}")]})
    assert res["must_depart_transport"] == ["train"]
    assert res["must_return_transport"] == ["train"]
    print("test_nonstandard_shapes OK")


def test_scheduled_poi_times_window_pinning():
    agent = _bare_agent()
    agent.too_many_backtrack = False
    agent.activities_stay_time_dict = None
    # containment window
    agent.activities_window_dict = {
        "Shengli River Food Street": {
            "window": ["15:47", "17:32"], "semantics": "within", "kind": None}
    }
    agent.activities_arrive_time_dict = {
        "Shengli River Food Street": ["late", "15:47"]}
    agent.activities_leave_time_dict = {
        "Shengli River Food Street": ["early", "17:32"]}
    # early arrival waits for A
    times = agent._scheduled_poi_times(
        "Shengli River Food Street", "14:00", "09:00", "22:00", 60, "attraction")
    assert times is not None and times[0] == "15:47", times
    a, b = times
    assert a >= "15:47" and b <= "17:32"
    # late arrival inside the window still fits
    times = agent._scheduled_poi_times(
        "Shengli River Food Street", "16:30", "09:00", "22:00", 60, "attraction")
    assert times is not None and times[0] == "16:30" and times[1] <= "17:32"
    # arrival after B is infeasible
    times = agent._scheduled_poi_times(
        "Shengli River Food Street", "17:40", "09:00", "22:00", 60, "attraction")
    assert times is None

    # covering window (phase-1): LEGACY scheduling is preserved exactly --
    # start at arrival (early start is oracle-valid for start<=A) and extend
    # the visit to B. Pinning covering windows onto [A, B] was reverted after
    # it regressed phase-1 placement feasibility.
    agent.activities_window_dict = {
        "Shence Gate": {"window": ["15:34", "17:05"], "semantics": "cover",
                        "kind": None}}
    agent.activities_arrive_time_dict = {"Shence Gate": ["early", "15:34"]}
    agent.activities_leave_time_dict = {"Shence Gate": ["late", "17:05"]}
    times = agent._scheduled_poi_times(
        "Shence Gate", "14:00", "09:00", "22:00", 60, "attraction")
    assert times == ("14:00", "17:05"), times
    # arriving after A can no longer cover the window
    times = agent._scheduled_poi_times(
        "Shence Gate", "16:00", "09:00", "22:00", 60, "attraction")
    assert times is None
    print("test_scheduled_poi_times_window_pinning OK")


def test_coverage_window_containment_dialect():
    numbered_nl = (
        "Requirements:\n"
        "1. The trip must last 2 days.\n"
        "2. The plan must be for 2 travelers.\n"
        "3. Tickets must match 2 travelers.\n"
        "4. Visit Shence Gate between 15:34 and 17:05.\n"
    )
    cons, record = cc.enforce_window_containment_dialect(
        [COVER_POSONLY], {"nature_language": numbered_nl}, "en")
    assert record is not None
    assert "activity_start_time(activity)>='15:34'" in cons[0]
    assert "activity_end_time(activity)<='17:05'" in cons[0]
    # idempotent
    cons2, record2 = cc.enforce_window_containment_dialect(
        cons, {"nature_language": numbered_nl}, "en")
    assert record2 is None and cons2 == cons
    # prose register (phase-1) is never rewritten
    prose_nl = ("We are 4 people, and we wish to visit Shence Gate "
                "between 15:34 and 17:05.")
    cons3, record3 = cc.enforce_window_containment_dialect(
        [COVER_POSONLY], {"nature_language": prose_nl}, "en")
    assert record3 is None and cons3 == [COVER_POSONLY]
    print("test_coverage_window_containment_dialect OK")


def test_transport_chain_helpers():
    agent = _bare_agent()
    agent.must_not_innercity_transport = ["walk"]
    agent.must_innercity_transport = None
    assert agent._banned_innercity_modes() == {"walk"}
    agent.must_innercity_transport = ["metro", "taxi"]
    assert agent._banned_innercity_modes() == {"walk"}
    walk_chain = [{"mode": "walk"}]
    metro_chain = [{"mode": "walk"}, {"mode": "metro"}, {"mode": "walk"}]
    taxi_chain = [{"mode": "taxi"}]
    assert agent._plan_transport_chain_type(walk_chain) == "walk"
    assert agent._plan_transport_chain_type(metro_chain) == "metro"
    assert agent._plan_transport_chain_type(taxi_chain) == "taxi"
    assert agent._plan_transport_chain_type([]) is None
    print("test_transport_chain_helpers OK")


def test_calculate_distance_bad_coordinates():
    agent = _bare_agent()
    agent._distance_cache = {}

    class _Junk:
        def search(self, city, name):
            return "unknown format junk"

    agent.poi_search = _Junk()
    d = agent.calculate_distance({"target_city": "Shanghai"}, "A POI", "B POI")
    assert d is None
    print("test_calculate_distance_bad_coordinates OK")


def test_coverage_between_window_injection():
    nl = (
        "We are 2 people traveling from Nanjing to Guangzhou for 2 days.\n"
        "Requirements:\n"
        "1. The trip must last 2 days.\n"
        "2. The plan must be for 2 travelers.\n"
        "3. Tickets for attractions, intercity transport, and metro rides must match 2 travelers.\n"
        "4. Dine at Guangzhou Junye Hotel between 07:50 and 08:45.\n"
        "5. Stay at Victoria Hotel Zhujiang New Town Guangzhou between 21:39 and 24:00.\n"
    )
    query = {"nature_language": nl, "target_city": "Guangzhou"}
    # both names resolve against the public Guangzhou DB
    cons, record = cc.enforce_between_windows([], query, "en")
    assert record is not None and len(record["changes"]) == 2
    joined = "\n".join(cons)
    assert "activity_start_time(activity)>='07:50'" in joined
    assert "activity_end_time(activity)<='08:45'" in joined
    assert "activity_type(activity) in ['breakfast', 'lunch', 'dinner']" in joined
    assert "activity_type(activity)=='accommodation'" in joined
    # idempotent
    cons2, record2 = cc.enforce_between_windows(cons, query, "en")
    assert record2 is None and cons2 == cons
    # present (either dialect) -> untouched
    covered = [
        CONTAIN_GUARD_MEAL,
        COVER_GUARD_ACC.replace("22:29", "21:39").replace(
            "Qiuguo S Hotel (Beijing Capital Airport Second Branch)",
            "Victoria Hotel Zhujiang New Town Guangzhou"),
    ]
    cons3, record3 = cc.enforce_between_windows(
        covered, {"nature_language": nl, "target_city": "Guangzhou"}, "en")
    assert record3 is None
    print("test_coverage_between_window_injection OK")


def test_coverage_mode_line_injection():
    nl = (
        "Requirements:\n"
        "1. The trip must last 2 days.\n"
        "2. Do not use walking for transportation within the destination city.\n"
    )
    query = {"nature_language": nl, "target_city": "Guangzhou"}
    cons, record = cc.enforce_innercity_mode_lines([], query, "en")
    assert record is not None
    assert "result=not({'walk'}&inner_city_transportation_set)" in cons[-1]
    # idempotent (the injected constraint IS a mode constraint)
    cons2, record2 = cc.enforce_innercity_mode_lines(cons, query, "en")
    assert record2 is None

    nl_only = (
        "Requirements:\n"
        "1. Use only metro and taxi for transportation within the destination city.\n"
    )
    cons, record = cc.enforce_innercity_mode_lines(
        [], {"nature_language": nl_only, "target_city": "Guangzhou"}, "en")
    assert record is not None
    assert "result=not({'walk'}&inner_city_transportation_set)" in cons[-1]

    # an existing mode constraint (any dialect) suppresses injection
    cons, record = cc.enforce_innercity_mode_lines(
        [WHITELIST_CANON], {"nature_language": nl_only, "target_city": "Guangzhou"}, "en")
    assert record is None
    print("test_coverage_mode_line_injection OK")


def test_coverage_numbered_flag():
    nl = "Requirements:\n" + "\n".join(
        "%d. Requirement line %d." % (i, i) for i in range(1, 9))
    cons = ["result=(day_count(plan)==2)"] * 5
    _, record = cc.flag_numbered_requirements(cons, {"nature_language": nl}, "en")
    assert record is not None and record["flag_only"]
    assert record["numbered_lines"] == 8 and record["constraints"] == 5
    _, record = cc.flag_numbered_requirements(
        ["c"] * 8, {"nature_language": nl}, "en")
    assert record is None
    # prose register (no numbered lines) never flags
    _, record = cc.flag_numbered_requirements(
        [], {"nature_language": "We want a nice 2-day trip."}, "en")
    assert record is None
    print("test_coverage_numbered_flag OK")


def test_window_snap_candidates():
    agent = _bare_agent()
    agent.must_see_attraction = []
    agent.must_visit_restaurant = []
    itinerary = [{"activities": [
        {"type": "breakfast", "position": "Guangzhou Junye Hotel",
         "start_time": "06:00", "end_time": "06:30"},
        {"type": "accommodation", "position": "Qiuguo S Hotel",
         "start_time": "20:00", "end_time": "24:00"},
    ]}]
    spec = {"window": ["07:50", "08:45"], "semantics": "within", "kind": "meal"}
    matches = agent._window_snap_candidates(itinerary, "Guangzhou Junye Hotel", spec)
    assert matches == [(0, 0)]
    # already inside the window -> nothing to snap
    itinerary[0]["activities"][0]["start_time"] = "07:55"
    itinerary[0]["activities"][0]["end_time"] = "08:40"
    assert agent._window_snap_candidates(
        itinerary, "Guangzhou Junye Hotel", spec) is None
    # kind guard: an accommodation window never matches a meal activity
    spec_acc = {"window": ["21:39", "24:00"], "semantics": "within",
                "kind": "accommodation"}
    matches = agent._window_snap_candidates(itinerary, "Qiuguo S Hotel", spec_acc)
    assert matches == [(0, 1)]
    print("test_window_snap_candidates OK")


def main():
    test_window_extraction()
    test_nonstandard_shapes()
    test_scheduled_poi_times_window_pinning()
    test_transport_chain_helpers()
    test_calculate_distance_bad_coordinates()
    test_coverage_window_containment_dialect()
    test_coverage_between_window_injection()
    test_coverage_mode_line_injection()
    test_coverage_numbered_flag()
    test_window_snap_candidates()
    print("ALL PHASE2-STYLE TESTS PASSED")


if __name__ == "__main__":
    main()

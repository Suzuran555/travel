"""Unit tests for the round-7 phase-2 famil fixes.

Every fixture is the (minimally trimmed) shape of a real phase2_familiar
failure from the round-6 dress rehearsal:

  * vacuous transport-mode pseudo-DSL un-vacuation (canonicalizer)
    -- uids 00001/00002/00013/00017/00020/00041/00069/00088/00095/00096/00099
  * set-accumulator semantic canonicalization (hotel_feature_set) -- uid 00018
  * intercity membership-via-intersection dialect -- uid 00100
  * same-POI window collisions: (name, kind) keying + within-window
    intersection + accommodation extras -- uids m..00003, 00094, 00033
  * inline-accessor type-literal spans (NL grounding reach) -- uid 00033
  * verifier-style scoped restaurant cost -- uid 00006

Run directly (no pytest needed):
    .venv/bin/python tests/test_round7_fixes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from chinatravel.agent.UrbanTrip.dsl_canonicalizer import (
    canonicalize_hard_logic_py,
)
from chinatravel.agent.UrbanTrip.tpc_agent_optimized_v6 import (
    UrbanTripOptimizedV6,
)

PASSED = FAILED = 0


def check(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print("PASS", name)
    else:
        FAILED += 1
        print("FAIL", name)


class _Stub:
    pass


def _extract(constraints, attractions=(), restaurants=(), hotels=()):
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


# ---------------------------------------------------------------------------
# 1. vacuous transport-mode pseudo-DSL -> canonical mode-collection loop
# ---------------------------------------------------------------------------

VACUOUS_WALK_BAN = (
    "inner_city_transportation_set=set()\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='transportation': "
    "inner_city_transportation_set.add(activity_position(activity))\n"
    "result=not({'walk'}&inner_city_transportation_set)"
)

CANONICAL_MODE_LINE = (
    "  if activity_transports(activity)!=[]: "
    "inner_city_transportation_set.add("
    "innercity_transport_type(activity_transports(activity)))"
)


def test_mode_set_inline_rewritten():
    out = canonicalize_hard_logic_py(VACUOUS_WALK_BAN)
    check(
        "mode_set_inline_rewritten",
        CANONICAL_MODE_LINE in out
        and "activity_position" not in out
        and "result=not({'walk'}&inner_city_transportation_set)" in out,
    )


def test_mode_set_block_rewritten():
    block = (
        "inner_city_transportation_set=set()\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='transportation':\n"
        "    inner_city_transportation_set.add(activity_position(activity))\n"
        "result=not({'taxi', 'walk'}&inner_city_transportation_set)"
    )
    out = canonicalize_hard_logic_py(block)
    check(
        "mode_set_block_rewritten",
        "if activity_transports(activity)!=[]:" in out
        and "innercity_transport_type(activity_transports(activity))" in out
        and "activity_position" not in out,
    )


def test_mode_set_rewrite_not_vacuous():
    """The rewritten constraint must actually SEE a walk leg (the vacuous
    original never did): evaluate both against a tiny stub plan."""
    def allactivities(plan):
        return plan

    def activity_type(act):
        return act["type"]

    def activity_position(act):
        return act["position"]

    def activity_transports(act):
        return act.get("transports", [])

    def innercity_transport_type(transports):
        if len(transports) == 3:
            return transports[1]["mode"]
        return transports[0]["mode"] if transports else None

    env = {
        "allactivities": allactivities,
        "activity_type": activity_type,
        "activity_position": activity_position,
        "activity_transports": activity_transports,
        "innercity_transport_type": innercity_transport_type,
        "plan": [
            {"type": "attraction", "position": "A", "transports": []},
            {
                "type": "lunch",
                "position": "B",
                "transports": [{"mode": "walk"}],
            },
        ],
    }
    vac = dict(env)
    exec(VACUOUS_WALK_BAN, vac)
    rewritten = dict(env)
    exec(canonicalize_hard_logic_py(VACUOUS_WALK_BAN), rewritten)
    check(
        "mode_set_rewrite_not_vacuous",
        bool(vac["result"]) is True and bool(rewritten["result"]) is False,
    )


def test_mode_set_rewrite_passes_clean_plan():
    env_src = canonicalize_hard_logic_py(VACUOUS_WALK_BAN)
    scope = {
        "allactivities": lambda plan: plan,
        "activity_type": lambda a: a["type"],
        "activity_position": lambda a: a["position"],
        "activity_transports": lambda a: a.get("transports", []),
        "innercity_transport_type": lambda t: t[0]["mode"] if t else None,
        "plan": [
            {"type": "lunch", "position": "B", "transports": [{"mode": "metro"}]}
        ],
    }
    exec(env_src, scope)
    check("mode_set_rewrite_passes_clean_plan", bool(scope["result"]) is True)


def test_conjunct_transport_guard_left_alone():
    """The metro-tickets conjunct shape is NOT the mode-set dialect; the
    rewrite must not touch it (regression guard against over-firing)."""
    c = (
        "result=True\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='transportation' and "
        "innercity_transport_type(activity_transports(activity))=='metro' and "
        "metro_tickets(activity_transports(activity))!=4: result=False"
    )
    check(
        "conjunct_transport_guard_left_alone",
        canonicalize_hard_logic_py(c) == c,
    )


def test_extraction_still_reads_rewritten_ban():
    out = canonicalize_hard_logic_py(VACUOUS_WALK_BAN)
    res = _extract([out])
    check(
        "extraction_still_reads_rewritten_ban",
        res.get("must_not_innercity_transport") == ["walk"],
    )


# ---------------------------------------------------------------------------
# 2. hotel_feature_set: spelling rename + semantic accumulator rename
# ---------------------------------------------------------------------------

def test_hotel_feature_set_spelling_renamed():
    c = (
        "hotel_feature_set=set()\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='accommodation':\n"
        "    hotel_feature_set.add(accommodation_type(activity, target_city(plan)))\n"
        "result=('Free parking' in hotel_feature_set)"
    )
    out = canonicalize_hard_logic_py(c)
    res = _extract([out])
    check(
        "hotel_feature_set_spelling_renamed",
        "accommodation_type_set" in out
        and "hotel_feature_set" not in out
        and res.get("must_live_hotel_feature") == ["Free parking"],
    )


def test_semantic_accumulator_rename_unknown_spelling():
    """A spelling absent from SET_VARIABLE_RENAMES is still renamed by WHAT
    it accumulates (uid 00018's defense in depth)."""
    c = (
        "lodging_perks=set()\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='accommodation':\n"
        "    lodging_perks.add(accommodation_type(activity, target_city(plan)))\n"
        "result=not({'Designer hotel'}&lodging_perks)"
    )
    out = canonicalize_hard_logic_py(c)
    res = _extract([out])
    check(
        "semantic_accumulator_rename_unknown_spelling",
        "accommodation_type_set" in out
        and res.get("must_not_live_hotel_feature") == ["Designer hotel"],
    )


def test_semantic_rename_skips_mixed_accumulator():
    c = (
        "mixed=set()\n"
        "for activity in allactivities(plan):\n"
        "  mixed.add(accommodation_type(activity, target_city(plan)))\n"
        "  mixed.add(activity_position(activity))\n"
        "result=('X' in mixed)"
    )
    check(
        "semantic_rename_skips_mixed_accumulator",
        "mixed" in canonicalize_hard_logic_py(c),
    )


def test_semantic_rename_requires_set_init():
    c = (
        "for activity in allactivities(plan):\n"
        "  bag.add(attraction_type(activity, target_city(plan)))\n"
        "result=('Park' in bag)"
    )
    check(
        "semantic_rename_requires_set_init",
        "bag.add" in canonicalize_hard_logic_py(c),
    )


# ---------------------------------------------------------------------------
# 3. intercity membership-via-intersection dialect (uid 00100)
# ---------------------------------------------------------------------------

INTERSECT_TRAIN = (
    "intercity_transport_set = set()\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['train', 'airplane']:\n"
    "    intercity_transport_set.add(activity_type(activity))\n"
    "result=({'train'}&intercity_transport_set)"
)


def test_intersection_membership_extracts_train():
    res = _extract([INTERSECT_TRAIN])
    check(
        "intersection_membership_extracts_train",
        res.get("must_depart_transport") == ["train"]
        and res.get("must_return_transport") == ["train"],
    )


def test_intersection_membership_both_modes():
    c = INTERSECT_TRAIN.replace(
        "result=({'train'}&intercity_transport_set)",
        "result=({'train', 'airplane'}&intercity_transport_set)",
    )
    res = _extract([c])
    check(
        "intersection_membership_both_modes",
        res.get("must_intercity_transport_all") == ["airplane", "train"],
    )


def test_negated_intersection_still_a_ban():
    c = INTERSECT_TRAIN.replace(
        "result=({'train'}&intercity_transport_set)",
        "result=not({'airplane'}&intercity_transport_set)",
    )
    res = _extract([c])
    check(
        "negated_intersection_still_a_ban",
        res.get("must_not_depart_transport") == ["airplane"]
        and res.get("must_depart_transport") in (None, ["train"]),
    )


# ---------------------------------------------------------------------------
# 4. same-POI window collisions (uids m..00003, 00094, 00033)
# ---------------------------------------------------------------------------

HOTEL = "Overseas Chinese Town Nanyang Inn (Gankeng Ancient Town Shop)"

STAY_WINDOW = (
    "result=False\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='accommodation' and "
    "activity_position(activity)=='%s':\n"
    "    if activity_start_time(activity)>='20:39' and "
    "activity_end_time(activity)<='24:00': result=True"
) % HOTEL

MEAL_WINDOW = (
    "result=False\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner'] and "
    "activity_position(activity)=='%s':\n"
    "    if activity_start_time(activity)>='07:50' and "
    "activity_end_time(activity)<='08:40': result=True"
) % HOTEL


def test_two_kinds_same_name_keeps_both_windows():
    res = _extract([STAY_WINDOW, MEAL_WINDOW], hotels=[HOTEL])
    primary = (res.get("activities_window_dict") or {}).get(HOTEL)
    extras = res.get("activities_window_extra") or []
    extra_specs = [spec for name, spec in extras if name == HOTEL]
    check(
        "two_kinds_same_name_keeps_both_windows",
        primary is not None
        and primary.get("kind") == "meal"
        and primary.get("window") == ["07:50", "08:40"]
        and any(
            s.get("kind") == "accommodation"
            and s.get("window") == ["20:39", "24:00"]
            for s in extra_specs
        ),
    )


def test_two_kinds_order_independent():
    res = _extract([MEAL_WINDOW, STAY_WINDOW], hotels=[HOTEL])
    primary = (res.get("activities_window_dict") or {}).get(HOTEL)
    extras = [s for n, s in (res.get("activities_window_extra") or []) if n == HOTEL]
    check(
        "two_kinds_order_independent",
        primary is not None
        and primary.get("kind") == "meal"
        and any(s.get("kind") == "accommodation" for s in extras),
    )


JIAHONG = "Jiahong Hotel (Shanghai Pudong Airport, Chuansha Metro Station)"


def _meal_window(a, b):
    return (
        "result=False\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner'] and "
        "activity_position(activity)=='%s':\n"
        "    if activity_start_time(activity)>='%s' and "
        "activity_end_time(activity)<='%s': result=True"
    ) % (JIAHONG, a, b)


def test_overlapping_within_windows_intersected():
    res = _extract(
        [_meal_window("07:45", "08:40"), _meal_window("07:50", "08:35")],
        hotels=[JIAHONG],
    )
    spec = (res.get("activities_window_dict") or {}).get(JIAHONG)
    check(
        "overlapping_within_windows_intersected",
        spec is not None and spec.get("window") == ["07:50", "08:35"],
    )


def test_overlapping_windows_intersect_other_order():
    res = _extract(
        [_meal_window("07:50", "08:35"), _meal_window("07:45", "08:40")],
        hotels=[JIAHONG],
    )
    spec = (res.get("activities_window_dict") or {}).get(JIAHONG)
    check(
        "overlapping_windows_intersect_other_order",
        spec is not None and spec.get("window") == ["07:50", "08:35"],
    )


def test_single_window_unchanged():
    res = _extract([_meal_window("07:45", "08:40")], hotels=[JIAHONG])
    spec = (res.get("activities_window_dict") or {}).get(JIAHONG)
    check(
        "single_window_unchanged",
        spec is not None
        and spec.get("window") == ["07:45", "08:40"]
        and not res.get("activities_window_extra"),
    )


# ---------------------------------------------------------------------------
# 5. planner-side window plumbing
# ---------------------------------------------------------------------------

def test_all_window_spec_items_includes_extras():
    stub = _Stub()
    stub.activities_window_dict = {
        HOTEL: {"window": ["07:50", "08:40"], "semantics": "within", "kind": "meal"}
    }
    stub.activities_window_extra = [
        [
            HOTEL,
            {
                "window": ["20:39", "24:00"],
                "semantics": "within",
                "kind": "accommodation",
            },
        ]
    ]
    items = UrbanTripOptimizedV6._all_window_spec_items(stub)
    kinds = sorted(spec.get("kind") for _n, spec in items)
    check(
        "all_window_spec_items_includes_extras",
        len(items) == 2 and kinds == ["accommodation", "meal"],
    )


def test_all_window_spec_items_handles_missing():
    stub = _Stub()
    stub.activities_window_dict = None
    stub.activities_window_extra = None
    check(
        "all_window_spec_items_handles_missing",
        UrbanTripOptimizedV6._all_window_spec_items(stub) == [],
    )


def test_meal_type_for_window():
    stub = _Stub()
    stub._time_minutes = lambda s: UrbanTripOptimizedV6._time_minutes(stub, s)
    got = [
        UrbanTripOptimizedV6._meal_type_for_window(stub, t)
        for t in ("07:50", "12:30", "21:03")
    ]
    check("meal_type_for_window", got == ["breakfast", "lunch", "dinner"])


def test_plan_restaurant_cost_scoped():
    stub = _Stub()
    plan = {
        "itinerary": [
            {
                "activities": [
                    {"type": "breakfast", "cost": 40},
                    {"type": "attraction", "cost": 300},
                    {"type": "dinner", "cost": 656},
                    {"type": "accommodation", "cost": 900},
                ]
            }
        ]
    }
    check(
        "plan_restaurant_cost_scoped",
        UrbanTripOptimizedV6._plan_restaurant_cost(stub, plan) == 696.0,
    )


# ---------------------------------------------------------------------------
# 6. windowed-POI insertion vs intercity day boundaries (uid 00005: overnight
#    arrival train 17:42 -> 04:27; the insert must land AFTER the arrival leg)
# ---------------------------------------------------------------------------

def _insertion_stub():
    stub = _Stub()
    stub.must_see_attraction = ["Maple Bridge"]
    stub.must_visit_restaurant = []
    stub._time_minutes = lambda s: UrbanTripOptimizedV6._time_minutes(stub, s)
    stub._is_intercity_activity = (
        lambda a: UrbanTripOptimizedV6._is_intercity_activity(stub, a)
    )
    stub._activity_position = (
        lambda a: UrbanTripOptimizedV6._activity_position(stub, a)
    )
    stub._meal_type_for_window = (
        lambda t: UrbanTripOptimizedV6._meal_type_for_window(stub, t)
    )
    stub._pinned_transport_legs = lambda *a, **k: []
    stub._reconnect_day_transports = (
        lambda acts: UrbanTripOptimizedV6._reconnect_day_transports(stub, acts)
    )
    stub._activity_end_position = (
        lambda a: UrbanTripOptimizedV6._activity_end_position(stub, a)
    )
    stub._rebuild_intercity_access = (
        lambda q, acts: UrbanTripOptimizedV6._rebuild_intercity_access(
            stub, q, acts
        )
    )
    # env-free transport stubs: a single direct leg, one mode, arriving
    # 10 minutes after departure
    stub._postprocess_transport_modes = lambda q, s, e: ["metro"]

    def _collect(q, s, e, t, mode):
        h, m = map(int, str(t).split(":")[:2])
        m += 10
        h, m = h + m // 60, m % 60
        arr = "%02d:%02d" % (h, m)
        return (
            [{"start": s, "end": e, "mode": mode, "cost": 5,
              "start_time": t, "end_time": arr}],
            arr,
        )

    stub._postprocess_collect_transport = _collect
    return stub


def _build(stub, itinerary, window_a, window_b):
    row = {"name": "Maple Bridge", "price": 0}
    return UrbanTripOptimizedV6._build_windowed_poi_insertion(
        stub, {"people_number": 3}, itinerary, 0, row,
        "attraction", window_a, window_b,
    )


OVERNIGHT_DAY = [
    {
        "activities": [
            {"type": "train", "start": "A Station", "end": "B Station",
             "start_time": "17:42", "end_time": "04:27", "transports": []},
            {"type": "breakfast", "position": "Hotel H",
             "start_time": "06:00", "end_time": "06:30", "transports": []},
            {"type": "accommodation", "position": "Hotel H",
             "start_time": "06:30", "end_time": "24:00", "transports": []},
        ]
    }
]


def test_insert_after_overnight_arrival():
    stub = _insertion_stub()
    out = _build(stub, deepcopy(OVERNIGHT_DAY), "08:15", "10:05")
    acts = out[0]["activities"] if out else []
    order = [a.get("type") for a in acts]
    check(
        "insert_after_overnight_arrival",
        out is not None
        and order[0] == "train"
        and "attraction" in order
        and order.index("attraction") > 0
        and acts[order.index("attraction")]["start_time"] == "08:15",
    )


def test_insert_rejected_before_arrival():
    stub = _insertion_stub()
    out = _build(stub, deepcopy(OVERNIGHT_DAY), "03:00", "04:00")
    check("insert_rejected_before_arrival", out is None)


def test_insert_rejected_after_departure():
    stub = _insertion_stub()
    day = [
        {
            "activities": [
                {"type": "attraction", "position": "P",
                 "start_time": "09:00", "end_time": "10:00", "transports": []},
                {"type": "train", "start": "B Station", "end": "A Station",
                 "start_time": "18:00", "end_time": "22:00", "transports": []},
            ]
        }
    ]
    out = _build(stub, day, "19:00", "20:30")
    check("insert_rejected_after_departure", out is None)


def test_insert_moves_overlapping_checkin():
    stub = _insertion_stub()
    out = _build(stub, deepcopy(OVERNIGHT_DAY), "08:15", "10:05")
    checkin = None
    for act in (out[0]["activities"] if out else []):
        if act.get("type") == "accommodation":
            checkin = act
    check(
        "insert_moves_overlapping_checkin",
        checkin is not None and checkin["start_time"] == "10:05",
    )


from copy import deepcopy  # noqa: E402  (used by the insertion tests)


# ---------------------------------------------------------------------------
# 7. round-7b: late-arrival window pinning, intercity access rebuild,
#    droppable-filler selection (uids 00005 / 00030 / 00093)
# ---------------------------------------------------------------------------

def _legs_stub(modes_costs, leg_minutes=10):
    """Stub with parametrized transport modes: {mode: cost}."""
    stub = _insertion_stub()
    stub._WINDOW_MIN_VISIT_MIN = UrbanTripOptimizedV6._WINDOW_MIN_VISIT_MIN
    stub._postprocess_transport_modes = lambda q, s, e: list(modes_costs)

    def _collect(q, s, e, t, mode):
        h, m = map(int, str(t).split(":")[:2])
        m += leg_minutes
        h, m = h + m // 60, m % 60
        arr = "%02d:%02d" % (h, m)
        return (
            [{"start": s, "end": e, "mode": mode, "cost": modes_costs[mode],
              "start_time": t, "end_time": arr}],
            arr,
        )

    stub._postprocess_collect_transport = _collect
    return stub


def test_pinned_legs_accept_late_arrival_within_window():
    # prev ends 11:00, window [10:58, 12:08]: arrival 11:10 misses A but
    # leaves >10 min before B -> legs must be returned (uid 00093 swap)
    stub = _legs_stub({"metro": 5})
    prev = {"position": "Church", "end_time": "11:00"}
    legs = UrbanTripOptimizedV6._pinned_transport_legs(
        stub, {"people_number": 5}, prev, "Chen's", "10:58", "12:08"
    )
    check(
        "pinned_legs_accept_late_arrival_within_window",
        legs and legs[-1]["end_time"] == "11:10",
    )


def test_pinned_legs_reject_arrival_too_close_to_window_end():
    # arrival 12:05 leaves <10 min before B=12:08 -> no legs
    stub = _legs_stub({"metro": 5})
    prev = {"position": "Church", "end_time": "11:55"}
    legs = UrbanTripOptimizedV6._pinned_transport_legs(
        stub, {"people_number": 5}, prev, "Chen's", "10:58", "12:08"
    )
    check("pinned_legs_reject_arrival_too_close_to_window_end", legs is None)


def test_pinned_legs_prefer_cheapest_mode():
    stub = _legs_stub({"taxi": 22, "metro": 5, "walk": 0})
    prev = {"position": "Hotel", "end_time": "06:30"}
    legs = UrbanTripOptimizedV6._pinned_transport_legs(
        stub, {"people_number": 3}, prev, "Bridge", "08:15", "10:05"
    )
    check(
        "pinned_legs_prefer_cheapest_mode",
        legs and legs[0]["mode"] == "walk",
    )


def test_reconnect_leaves_intercity_alone():
    stub = _insertion_stub()
    acts = [
        {"type": "attraction", "position": "P1",
         "start_time": "10:00", "end_time": "11:00", "transports": []},
        {"type": "train", "start": "Station", "end": "Home",
         "start_time": "18:00", "end_time": "22:00",
         "transports": [{"start": "Hotel", "end": "Station", "mode": "metro"}]},
    ]
    UrbanTripOptimizedV6._reconnect_day_transports(stub, acts)
    check(
        "reconnect_leaves_intercity_alone",
        acts[1]["transports"] and acts[1]["transports"][0]["start"] == "Hotel",
    )


def test_rebuild_intercity_access_rewires_stale_chain():
    # departure train's access chain still starts at the hotel after an
    # insert put a restaurant before it (uid 00093 d1 airplane)
    stub = _legs_stub({"metro": 8})
    acts = [
        {"type": "lunch", "position": "Chen's",
         "start_time": "11:00", "end_time": "12:08", "transports": []},
        {"type": "train", "start": "Station", "end": "Home",
         "start_time": "23:22", "end_time": "00:29",
         "transports": [{"start": "Hotel", "end": "Station", "mode": "metro"}]},
    ]
    ok = UrbanTripOptimizedV6._rebuild_intercity_access(
        stub, {"people_number": 5}, acts
    )
    check(
        "rebuild_intercity_access_rewires_stale_chain",
        ok and acts[1]["transports"][0]["start"] == "Chen's",
    )


def test_rebuild_intercity_access_fails_when_too_late():
    stub = _legs_stub({"metro": 8}, leg_minutes=90)
    acts = [
        {"type": "dinner", "position": "Chen's",
         "start_time": "22:30", "end_time": "23:00", "transports": []},
        {"type": "train", "start": "Station", "end": "Home",
         "start_time": "23:22", "end_time": "00:29", "transports": []},
    ]
    ok = UrbanTripOptimizedV6._rebuild_intercity_access(
        stub, {"people_number": 5}, acts
    )
    check("rebuild_intercity_access_fails_when_too_late", not ok)


def test_droppable_fillers_exclude_protected_and_sort_by_leg_cost():
    stub = _insertion_stub()
    stub.must_see_attraction = ["Joyous Pavilion Park"]
    stub._all_window_spec_items = lambda: [("Windowed POI", {"window": ["08:00", "09:00"]})]
    itin = [
        {"activities": [
            {"type": "attraction", "position": "Joyous Pavilion Park",
             "transports": [{"cost": 22}]},
            {"type": "attraction", "position": "Rostrum",
             "transports": [{"cost": 38}]},
            {"type": "attraction", "position": "Windowed POI",
             "transports": [{"cost": 50}]},
            {"type": "attraction", "position": "Guardian",
             "transports": [{"cost": 15}]},
            {"type": "dinner", "position": "Cafe",
             "transports": [{"cost": 99}]},
        ]}
    ]
    out = UrbanTripOptimizedV6._droppable_filler_positions(stub, itin)
    check(
        "droppable_fillers_exclude_protected_and_sort_by_leg_cost",
        out == ["Rostrum", "Guardian"],
    )


def test_itinerary_without_position_drops_and_reconnects():
    stub = _legs_stub({"metro": 8})
    itin = [
        {"activities": [
            {"type": "attraction", "position": "A",
             "start_time": "09:00", "end_time": "10:00", "transports": []},
            {"type": "attraction", "position": "B",
             "start_time": "10:30", "end_time": "11:30",
             "transports": [{"start": "A", "end": "B", "mode": "metro"}]},
            {"type": "attraction", "position": "C",
             "start_time": "12:00", "end_time": "13:00",
             "transports": [{"start": "B", "end": "C", "mode": "metro"}]},
        ]}
    ]
    out = UrbanTripOptimizedV6._itinerary_without_position(
        stub, {"people_number": 2}, itin, "B"
    )
    acts = out[0]["activities"] if out else []
    check(
        "itinerary_without_position_drops_and_reconnects",
        out is not None
        and [a["position"] for a in acts] == ["A", "C"]
        and acts[1]["transports"] == [],
    )


def test_insertion_late_arrival_start_moves_to_arrival():
    # legs arriving after A: the pinned activity starts at the arrival time
    stub = _legs_stub({"metro": 5})
    stub._pinned_transport_legs = (
        lambda *a, **k: [{"start": "Hotel H", "end": "Maple Bridge",
                          "mode": "metro", "cost": 5,
                          "start_time": "08:20", "end_time": "08:30"}]
    )
    out = _build(stub, deepcopy(OVERNIGHT_DAY), "08:15", "10:05")
    inserted = None
    for act in (out[0]["activities"] if out else []):
        if act.get("type") == "attraction":
            inserted = act
    check(
        "insertion_late_arrival_start_moves_to_arrival",
        inserted is not None and inserted["start_time"] == "08:30"
        and inserted["end_time"] == "10:05",
    )


# ---------------------------------------------------------------------------

def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print()
    print("%d/%d tests passed" % (PASSED, PASSED + FAILED))
    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()

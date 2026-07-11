"""Unit tests for the round-3 mechanical verifiers in nl2sl_hybrid_en:

  * disjunction marker detection / branch counting over the NL
  * disjunction_gap (or-arity check over generated constraints)
  * merge_disjunction_constraint (drops lone-branch fragments, keeps base)
  * enforce_disjunction end-to-end with a fake LLM (no model calls)
  * strip_room_constraints / normalize_generated_constraints room guard
  * transport-mode normalizer must not rewrite mixed-domain (disjunction) blocks

Run directly (no pytest needed):
    .venv/bin/python tests/test_disjunction_room_strip.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chinatravel.agent.nesy_agent.nl2sl_hybrid_en import (
    detect_disjunction,
    disjunction_gap,
    enforce_disjunction,
    merge_disjunction_constraint,
    normalize_generated_constraints,
    normalize_transport_mode_constraint,
    strip_room_constraints,
    _or_arity,
)


# ---------------------------------------------------------------------------
# Disjunction marker detection.
# ---------------------------------------------------------------------------

NL_DISJ = (
    "We are a group of 5, departing from Nanjing for a 2-day trip to Beijing, "
    "and must meet at least one of the following conditions: 1. Do not wish to "
    "visit Xidan Commercial Street and Dingling Mausoleum; 2. Accommodation "
    "budget is 3300.0."
)


def test_detect_at_least_one_of():
    info = detect_disjunction(NL_DISJ)
    assert info is not None
    assert "at least one of" in info["marker"].lower()
    assert info["branch_count"] == 2


def test_detect_branch_labels_next_to_decimals():
    # '4500.0 2.' must count label 2 while ignoring the decimals
    nl = (
        "The trip must satisfy at least one of the following: 1. The dining "
        "budget is 4500.0 2. We do not want to use walking or taxi for travel "
        "within the city."
    )
    info = detect_disjunction(nl)
    assert info["branch_count"] == 2


def test_detect_phrasing_variants():
    variants = [
        "meeting any one of the following requirements: 1. A 2. B",
        "We need to meet any one of the following: 1. A 2. B",
        "One of the following conditions must be met: 1. A 2. B",
        "One of the following must be satisfied: 1. A 2. B",
        "must satisfy either: 1. A; 2. B.",
        "we need to meet either of the following conditions: 1. A 2. B",
        "satisfying either of the following: 1. A 2. B",
        "The following conditions must be met (any one of them): 1. A 2. B",
        "with one of the following requirements: 1. A. 2. B.",
        "Requirements: satisfy any one of the following: 1. A; 2. B.",
        "The itinerary must satisfy one of the following: 1. A. 2. B.",
        "and we need to meet one of the following requirements: 1. A; 2. B",
    ]
    for nl in variants:
        info = detect_disjunction(nl)
        assert info is not None, "marker missed: %s" % nl
        assert info["branch_count"] == 2, "branches missed: %s" % nl


def test_detect_chinese_markers():
    variants = [
        "我们一行5人，需要满足以下要求中的至少一项：1、不去西单 2、住宿预算3300",
        "需满足以下条件中的任意一条: 1. 预算3300 2. 不去西单",
        "行程需要至少满足以下之一：1. 预算3300 2. 不去西单",
    ]
    for nl in variants:
        info = detect_disjunction(nl)
        assert info is not None, "zh marker missed: %s" % nl
        assert info["branch_count"] == 2, "zh branches missed: %s" % nl


def test_no_marker_on_conjunction_list():
    # numbered requirement lists WITHOUT any-one-of wording are conjunctions
    nl = (
        "We are 2 people traveling to Chengdu for 3 days, with the following "
        "requirements: 1. the sightseeing budget is 500.0 2. no taxi rides."
    )
    assert detect_disjunction(nl) is None


def test_no_marker_on_inner_one_of_idiom():
    # inner 'one of the following restaurants' is an any-of-set requirement,
    # not a top-level disjunction
    nl = (
        "We are 2 people traveling to Suzhou for 2 days. We want to try one of "
        "the following types of restaurants: Hot pot."
    )
    assert detect_disjunction(nl) is None


def test_no_marker_on_plain_query():
    nl = "We are 2 people traveling from Shenzhen to Chengdu for 3 days, with a budget of 1300.0."
    assert detect_disjunction(nl) is None


# ---------------------------------------------------------------------------
# disjunction_gap / or-arity.
# ---------------------------------------------------------------------------

DISJ_OK = (
    "attraction_name_set=set()\nhotel_cost=0\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='attraction': attraction_name_set.add(activity_position(activity))\n"
    "  if activity_type(activity)=='accommodation': hotel_cost+=activity_cost(activity)\n"
    "branch_1=not({'Xidan Commercial Street', 'Dingling Mausoleum'}&attraction_name_set)\n"
    "branch_2=(hotel_cost<=3300.0)\n"
    "result=(branch_1 or branch_2)"
)
BASE = [
    "result=(day_count(plan)==2)",
    "result=(people_count(plan)==5)",
    "result=True\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=5: result=False\n"
    "  if innercity_transport_type(activity_transports(activity))=='metro' and metro_tickets(activity_transports(activity))!=5: result=False",
    "result=True\nfor activity in allactivities(plan):\n"
    "  if innercity_transport_type(activity_transports(activity))=='taxi' and taxi_cars(activity_transports(activity))!=2: result=False",
]
COLLAPSED = (
    "total_cost=0\nfor activity in allactivities(plan): "
    "total_cost+=activity_cost(activity)+innercity_transport_cost(activity_transports(activity))\n"
    "result=(total_cost<=3300.0)"
)


def test_gap_flagged_when_or_missing():
    gap = disjunction_gap(NL_DISJ, BASE + [COLLAPSED])
    assert gap is not None
    assert gap["required_or_arity"] == 1
    assert gap["found_or_arity"] == 0


def test_gap_absent_when_or_present():
    assert disjunction_gap(NL_DISJ, BASE + [DISJ_OK]) is None


def test_or_inside_string_literal_does_not_count():
    fake = "result=({'Amusement Park or Sports'}<=attraction_name_set)"
    assert _or_arity(fake) == 0
    assert disjunction_gap(NL_DISJ, BASE + [fake]) is not None


def test_gap_requires_two_branches():
    # truncated NL with a single enumerated branch: nothing to OR
    nl = "We need to meet at least one of the following: 1. Dining budget of 190"
    assert disjunction_gap(nl, ["result=(day_count(plan)==3)"]) is None


# ---------------------------------------------------------------------------
# merge_disjunction_constraint.
# ---------------------------------------------------------------------------

def test_merge_drops_collapsed_budget_branch():
    merged = merge_disjunction_constraint(BASE + [COLLAPSED], DISJ_OK)
    assert DISJ_OK in merged
    assert COLLAPSED not in merged  # shares 3300.0 with the disjunction
    for b in BASE:
        assert b in merged


def test_merge_drops_split_name_branch():
    split_branch = (
        "attraction_names_set = set()\nfor activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction':\n"
        "    attraction_names_set.add(activity_position(activity))\n"
        "result=not({'Xidan Commercial Street', 'Dingling Mausoleum'}&attraction_names_set)"
    )
    merged = merge_disjunction_constraint(BASE + [split_branch], DISJ_OK)
    assert split_branch not in merged
    assert DISJ_OK in merged


def test_merge_keeps_unrelated_constraint():
    unrelated = (
        "intercity_transport_set = set()\nfor activity in allactivities(plan):\n"
        "  if activity_type(activity) in ['train', 'airplane']:\n"
        "    intercity_transport_set.add(activity_type(activity))\n"
        "result=(intercity_transport_set=={'train'})"
    )
    merged = merge_disjunction_constraint(BASE + [unrelated], DISJ_OK)
    assert unrelated in merged


def test_merge_drops_literal_free_fragment_by_function_subset():
    # 'only free attractions' branch: no identifying literals, but its domain
    # functions are covered by the disjunction
    free_branch = (
        "result=True\nfor activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction' and activity_price(activity)>0: result=False"
    )
    disj = (
        "ok=True\nhotel_cost=0\nfor activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction' and activity_price(activity)>0: ok=False\n"
        "  if activity_type(activity)=='accommodation': hotel_cost+=activity_cost(activity)\n"
        "result=(ok or hotel_cost<=4000.0)"
    )
    merged = merge_disjunction_constraint(BASE + [free_branch], disj)
    assert free_branch not in merged
    assert disj in merged


# ---------------------------------------------------------------------------
# enforce_disjunction with a fake LLM (no model calls).
# ---------------------------------------------------------------------------

class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def __call__(self, messages, one_line=False, json_mode=True):
        self.calls += 1
        self.prompt = messages[0]["content"]
        return self.response


def _disj_query(constraints):
    return {
        "nature_language": NL_DISJ,
        "days": 2,
        "people_number": 5,
        "hard_logic_py": list(constraints),
    }


def test_enforce_disjunction_injects_and_merges():
    import json as _json

    llm = FakeLLM(_json.dumps([DISJ_OK]))
    query = enforce_disjunction(_disj_query(BASE + [COLLAPSED]), llm)
    assert llm.calls == 1
    assert "at least one of" in llm.prompt.lower()
    assert NL_DISJ in llm.prompt
    assert DISJ_OK in query["hard_logic_py"]
    assert COLLAPSED not in query["hard_logic_py"]
    assert query["disjunction_reflect"]["constraint"] == DISJ_OK


def test_enforce_disjunction_noop_when_or_present():
    llm = FakeLLM("[]")
    query = enforce_disjunction(_disj_query(BASE + [DISJ_OK]), llm)
    assert llm.calls == 0
    assert query["hard_logic_py"] == BASE + [DISJ_OK]


def test_enforce_disjunction_drops_fragment_without_model_call():
    # OR-constraint already emitted, but a collapsed branch coexists as a
    # spurious hard cap: it must be dropped mechanically, no LLM call
    llm = FakeLLM("[]")
    query = enforce_disjunction(_disj_query(BASE + [COLLAPSED, DISJ_OK]), llm)
    assert llm.calls == 0
    assert COLLAPSED not in query["hard_logic_py"]
    assert DISJ_OK in query["hard_logic_py"]
    for b in BASE:
        assert b in query["hard_logic_py"]
    assert query["disjunction_fragment_drop"] == [COLLAPSED]


def test_enforce_disjunction_rejects_or_free_reemission():
    import json as _json

    llm = FakeLLM(_json.dumps([COLLAPSED]))  # model fails again: no `or`
    query = enforce_disjunction(_disj_query(BASE + [COLLAPSED]), llm)
    assert llm.calls == 2  # both trails used
    assert query["disjunction_reflect"] == {"failed": True}
    assert query["hard_logic_py"] == BASE + [COLLAPSED]  # left untouched


def test_enforce_disjunction_noop_on_plain_query():
    llm = FakeLLM("[]")
    query = {
        "nature_language": "We are 2 people traveling to Chengdu for 3 days.",
        "days": 3,
        "people_number": 2,
        "hard_logic_py": ["result=(day_count(plan)==3)"],
    }
    out = enforce_disjunction(query, llm)
    assert llm.calls == 0
    assert "disjunction_gap" not in out


# ---------------------------------------------------------------------------
# Room-constraint strip.
# ---------------------------------------------------------------------------

ROOM_BLOCK = (
    "result=True\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity)=='accommodation' and room_count(activity)!=2: result=False\n"
    "  if activity_type(activity)=='accommodation' and room_type(activity)!=1: result=False"
)
NL_NO_ROOM = (
    "We are 2 people traveling from Shenzhen to Chengdu for 3 days, with the "
    "following requirement: the sightseeing budget is 500.0."
)
NL_ROOM = "We would like a hotel with a single bed room, price under 500 CNY."


def test_room_block_stripped_when_nl_silent():
    out = strip_room_constraints([ROOM_BLOCK, "result=(day_count(plan)==3)"], NL_NO_ROOM)
    assert out == ["result=(day_count(plan)==3)"]


def test_room_block_kept_when_nl_mentions_rooms():
    for nl in [NL_ROOM, "需要两间双人间", "希望是标间", "想要大床房"]:
        out = strip_room_constraints([ROOM_BLOCK], nl)
        assert out == [ROOM_BLOCK], nl


def test_mixed_room_block_pruned_not_dropped():
    mixed = (
        "result=True\nfor activity in allactivities(plan):\n"
        "  if activity_type(activity)=='accommodation' and room_count(activity)!=1: result=False\n"
        "  if activity_type(activity)=='accommodation' and activity_price(activity)>500: result=False"
    )
    out = strip_room_constraints([mixed], NL_NO_ROOM)
    assert len(out) == 1
    assert "room_count" not in out[0]
    assert "activity_price(activity)>500" in out[0]


def test_normalize_generated_constraints_applies_room_guard():
    out = normalize_generated_constraints(
        [ROOM_BLOCK, "result=(people_count(plan)==2)"],
        people_count=2,
        nature_language=NL_NO_ROOM,
    )
    assert out == ["result=(people_count(plan)==2)"]
    # without the NL the strip must NOT fire (fails open)
    out = normalize_generated_constraints([ROOM_BLOCK], people_count=2)
    assert out == [ROOM_BLOCK]


# ---------------------------------------------------------------------------
# Mode normalizer must leave mixed-domain (disjunction) blocks alone.
# ---------------------------------------------------------------------------

def test_mode_normalizer_skips_disjunction_block():
    disj = (
        "meal_cost=0\nmode_ok=True\nfor activity in allactivities(plan):\n"
        "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']: meal_cost+=activity_cost(activity)\n"
        "  if innercity_transport_type(activity_transports(activity))=='walk': mode_ok=False\n"
        "  if innercity_transport_type(activity_transports(activity))=='taxi': mode_ok=False\n"
        "result=(meal_cost<=4500.0 or mode_ok)"
    )
    assert normalize_transport_mode_constraint(disj) == disj


def test_mode_normalizer_still_rewrites_pure_mode_block():
    pure = (
        "result=True\nfor activity in allactivities(plan):\n"
        "  if innercity_transport_type(activity_transports(activity))=='walk': result=False"
    )
    out = normalize_transport_mode_constraint(pure)
    assert "activity_type(activity)=='transportation'" in out
    assert "result=not({'walk'}&inner_city_transportation_set)" in out


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

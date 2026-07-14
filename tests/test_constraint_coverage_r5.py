"""Unit tests for the round-5 coverage rules
(chinatravel.agent.nesy_agent.constraint_coverage).

Every positive case is the NL of a real full-1000 sweep failure; every
negative case is a real sweep NL where the rule must NOT fire.

Run directly (no pytest needed):
    .venv/bin/python tests/test_constraint_coverage_r5.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chinatravel.agent.nesy_agent.constraint_coverage import (
    _disjunction_region_start,
    build_directional_constraint,
    drop_disjunction_fragments,
    enforce_budget_scope,
    enforce_directional_transport,
    enforce_distance_taxi,
    enforce_hotel_distance,
    enforce_time_windows,
    explicit_room_count,
    fix_membership_polarity,
    fix_negation_tautology,
    inject_base_boilerplate,
    inject_free_attractions,
    parse_directional_transport,
    _resolve_leg_specs,
    scoped_budget_mentions,
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


TOTAL_CAP = (
    "total_cost=0\nfor activity in allactivities(plan): "
    "total_cost+=activity_cost(activity)"
    "+innercity_transport_cost(activity_transports(activity))\n"
    "result=(total_cost<=%s)"
)


# --- disjunction region ------------------------------------------------------

check(
    "region_requires_numbered_labels",
    _disjunction_region_start(
        "requiring one of the following: 1. Wish to visit only free "
        "attractions 2. Wish to take train to the destination."
    )
    is not None,
)
check(
    "region_marker_must_satisfy_either",
    _disjunction_region_start(
        "must satisfy either: 1. Travel by train both ways, or "
        "2. Cross-city transport budget of 1400.0."
    )
    is not None,
)
check(
    "region_not_opened_by_inline_any_of",
    _disjunction_region_start(
        "we hope to stay at one of the following hotels: Four Points by "
        "Sheraton Shanghai Kangqiao or Atour S hotel, Shanghai Wanyuan Road."
    )
    is None,
)

# --- budget scope ------------------------------------------------------------

NL_INNER = (
    "We are 5 people departing from Nanjing to travel in Beijing for 2 days, "
    "with the following requirements: We do not wish to visit West Yellow "
    "Temple. The budget for travel within the city is 800.0."
)
m = scoped_budget_mentions(NL_INNER)
check("mention_innercity", m and m[0][0] == "innercity" and m[0][1] == 800.0)

NL_MEAL5 = (
    "A group of 5 people traveling from Suzhou to Chengdu for 3 days, with a "
    "meal budget of 1000.0 and an accommodation budget of 1900.0."
)
m = scoped_budget_mentions(NL_MEAL5)
check(
    "mention_no_daycount_false_amount",
    sorted((s, a) for s, a, _ in m) == [("accommodation", 1900.0), ("dining", 1000.0)],
)

q = {"nature_language": NL_INNER, "people_number": 5, "days": 2}
cons, rec = enforce_budget_scope([TOTAL_CAP % "800.0"], q)
check(
    "rescope_total_to_innercity",
    rec
    and not any("total_cost" in c for c in cons)
    and any("inner_city_transportation_cost<=800" in c for c in cons),
)

NL_DINING = (
    "Our group of 4 is traveling from Nanjing to Chongqing for 4 days, with "
    "the following requirements: we want to visit Amusement Park, with a "
    "dining budget of 2200.0."
)
q = {"nature_language": NL_DINING, "people_number": 4, "days": 4}
cons, rec = enforce_budget_scope([TOTAL_CAP % "2200.0"], q)
check(
    "rescope_total_to_dining",
    rec and any("restaurant_cost+=activity_cost(activity)" in c and "2200" in c for c in cons),
)

# correctly scoped translation is left alone
GOOD_DINING = (
    "restaurant_cost=0\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']: "
    "restaurant_cost+=activity_cost(activity)\nresult=(restaurant_cost<=2200)"
)
cons, rec = enforce_budget_scope([GOOD_DINING], q)
check("scoped_implementation_untouched", rec is None and cons == [GOOD_DINING])

# misfiltered innercity sum is replaced by the unfiltered oracle idiom
NL_INNER30 = (
    "1 person, departing from Beijing, traveling to Shenzhen for 3 days. "
    "Requirements: wish to visit a park. Budget for intra-city "
    "transportation is 30.0."
)
BAD_INNER = (
    "inner_city_transportation_cost=0\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity)!='accommodation':\n"
    "    inner_city_transportation_cost+="
    "innercity_transport_cost(activity_transports(activity))\n"
    "result=(inner_city_transportation_cost<=30.0)"
)
q = {"nature_language": NL_INNER30, "people_number": 1, "days": 3}
cons, rec = enforce_budget_scope([BAD_INNER], q)
check(
    "misfiltered_innercity_replaced",
    rec
    and BAD_INNER not in cons
    and any(
        "inner_city_transportation_cost<=30" in c and "!=" not in c for c in cons
    ),
)

# invented total cap (number absent from NL) is dropped
NL_NOBUDGET = (
    "2 people are traveling from Wuhan to Chengdu for 2 days and want to "
    "visit Jiaozi Park with a comfortable pace, and the cost matters."
)
q = {"nature_language": NL_NOBUDGET, "people_number": 2, "days": 2}
cons, rec = enforce_budget_scope([TOTAL_CAP % "3000"], q)
check("invented_total_cap_dropped", rec and cons == [])

# grounded overall budget is kept
NL_TOTAL = (
    "One person traveling from Shenzhen to Shanghai for 2 days, with the "
    "following requirements: only visit free attractions, total travel "
    "budget of 3000.0."
)
q = {"nature_language": NL_TOTAL, "people_number": 1, "days": 2}
cons, rec = enforce_budget_scope([TOTAL_CAP % "3000.0"], q)
check("grounded_total_cap_kept", any("total_cost" in c for c in cons))

# a scoped budget whose accumulator is merely NAMED total_cost but carries
# the right type filter is an implementation, not a mis-scoped total cap
NL_INTER800 = (
    "One person, traveling from Suzhou to Wuhan for 3 days. Requirement: "
    "Intercity transportation budget is 800.0."
)
MISNAMED = (
    "total_cost=0\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['train', 'airplane']:\n"
    "    total_cost+=activity_cost(activity)\nresult=(total_cost<=800.0)"
)
q = {"nature_language": NL_INTER800, "people_number": 1, "days": 3}
cons, rec = enforce_budget_scope([MISNAMED], q)
check("misnamed_filtered_accumulator_kept", rec is None and cons == [MISNAMED])

# scoped mention inside a disjunction region must not be enforced
NL_DISJ = (
    "We are 4 people traveling from Chongqing to Nanjing for 2 days. One of "
    "the following conditions must be met: 1. The budget for intra-city "
    "transportation is 60.0, or 2. We prefer to take a train to the "
    "destination and take a plane back."
)
q = {"nature_language": NL_DISJ, "people_number": 4, "days": 2}
cons, rec = enforce_budget_scope(["result=(people_count(plan)==4)"], q)
check("disjunction_branch_budget_not_injected", rec is None)

# --- disjunction fragment leak ----------------------------------------------

DISJ_OR = (
    "cost=0\nintercity_transport_set=set()\nfor activity in "
    "allactivities(plan):\n  cost+=innercity_transport_cost("
    "activity_transports(activity))\nbranch_1=(cost<=60.0)\n"
    "branch_2=(intercity_transport_set=={'train', 'airplane'})\n"
    "result=(branch_1 or branch_2)"
)
cons, rec = drop_disjunction_fragments([DISJ_OR, TOTAL_CAP % "60"])
check(
    "branch_cap_leak_dropped",
    rec and cons == [DISJ_OR],
)
cons, rec = drop_disjunction_fragments([DISJ_OR, TOTAL_CAP % "5000"])
check("unrelated_cap_not_dropped", rec is None)

# --- directional transport ----------------------------------------------------

NL_DIR_NEG = (
    "We are 4 people traveling from Shenzhen to Shanghai for 3 days. We want "
    "to visit Old Wharf. We do not want to fly to the destination, and we do "
    "not want to take a train back."
)
specs = parse_directional_transport(NL_DIR_NEG)
resolved = _resolve_leg_specs(specs)
check(
    "directional_neg_parse",
    resolved == {"go": ("!=", "airplane"), "back": ("!=", "train")},
)

NL_DIR_POS = (
    "we prefer to take a train to the destination and return by airplane."
)
resolved = _resolve_leg_specs(parse_directional_transport(NL_DIR_POS))
check(
    "directional_pos_parse",
    resolved == {"go": ("==", "train"), "back": ("==", "airplane")},
)

NL_NOR = (
    "We do not wish to take an airplane to the destination, nor do we wish "
    "to take an airplane for the return journey."
)
resolved = _resolve_leg_specs(parse_directional_transport(NL_NOR))
check(
    "directional_nor_do_we_negated",
    resolved == {"go": ("!=", "airplane"), "back": ("!=", "airplane")},
)

NL_ORBACK = "Do not want to take an airplane to the destination or back."
resolved = _resolve_leg_specs(parse_directional_transport(NL_ORBACK))
check(
    "directional_or_back_suffix",
    resolved == {"go": ("!=", "airplane"), "back": ("!=", "airplane")},
)

NL_BOTHWAYS = "Requirements: visit only free attractions, travel by airplane both ways."
resolved = _resolve_leg_specs(parse_directional_transport(NL_BOTHWAYS))
check(
    "directional_both_ways",
    resolved == {"go": ("==", "airplane"), "back": ("==", "airplane")},
)

check(
    "directional_silent_on_global_mode",
    parse_directional_transport(
        "we want to travel by train and stay in a twin room") == [],
)

GLOBAL_SET = (
    "intercity_transport_set = set()\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['train', 'airplane']:\n"
    "    intercity_transport_set.add(activity_type(activity))\n"
    "result=(intercity_transport_set=={'train'})"
)
q = {"nature_language": NL_DIR_POS, "target_city": "Shanghai"}
cons, rec = enforce_directional_transport([GLOBAL_SET], q)
check(
    "directional_replaces_global_set",
    rec
    and GLOBAL_SET not in cons
    and any("allactivities(plan)[0]['type'] == \"train\"" in c for c in cons)
    and any("allactivities(plan)[-1]['type'] == \"airplane\"" in c for c in cons),
)

TICKETS = (
    "result=True\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['attraction', 'airplane', 'train'] "
    "and activity_tickets(activity)!=4: result=False"
)
cons, rec = enforce_directional_transport([GLOBAL_SET, TICKETS], q)
check("directional_keeps_tickets_boilerplate", TICKETS in cons)

# inline branch replacement inside a disjunction
NL_DISJ_DIR = (
    "We are 4 people, departing from Shenzhen, traveling to Shanghai for 3 "
    "days. The trip must satisfy one of the following conditions:  \n"
    "1. Travel to the destination by train and return by plane.  \n"
    "2. Accommodation should be within 1.57 km of Shanghai Shipyard "
    "Riverside Green Space."
)
BRANCHED = (
    "intercity_transport_set = set()\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['train', 'airplane']:\n"
    "    intercity_transport_set.add(activity_type(activity))\n"
    "branch_1 = (intercity_transport_set == {'train', 'airplane'})\n"
    "branch_2 = True\n"
    "result=(branch_1 or branch_2)"
)
q = {"nature_language": NL_DISJ_DIR, "target_city": "Shanghai"}
cons, rec = enforce_directional_transport([BRANCHED], q)
check(
    "directional_inline_branch_rewrite",
    rec
    and rec["inlined"]
    and "allactivities(plan)[0]['type'] == \"train\"" in cons[0]
    and "result=(branch_1 or branch_2)" in cons[0]
    and not any("allactivities" in c and "branch" not in c for c in cons[1:]),
)

check(
    "directional_constraint_shape",
    build_directional_constraint({"go": ("!=", "train")})
    == "result=False\nintercity_transport_go=''\nintercity_transport_back=''\n"
    "if allactivities(plan)[0]['type'] != \"train\" and "
    "intercity_transport_origin(allactivities(plan)[0])==start_city(plan):\n"
    "  result=True",
)

# --- free attractions ---------------------------------------------------------

NL_FREE = (
    "We are 3 people traveling from Shanghai to Shenzhen for 4 days, with "
    "the following requirements:\n- Do not want to visit museums\n- Only "
    "want to visit free attractions"
)
cons, rec = inject_free_attractions([], {"nature_language": NL_FREE})
check(
    "free_attractions_injected",
    rec and any("attraction_cost<=0" in c for c in cons),
)
cons2, rec2 = inject_free_attractions(list(cons), {"nature_language": NL_FREE})
check("free_attractions_idempotent", rec2 is None)
NL_FREE_DISJ = (
    "requiring one of the following: 1. Wish to visit only free attractions "
    "2. Wish to take train to the destination and return by airplane."
)
cons, rec = inject_free_attractions([], {"nature_language": NL_FREE_DISJ})
check("free_attractions_skip_in_disjunction", rec is None)
cons, rec = inject_free_attractions(
    [], {"nature_language": "we want to visit the Free Trade Zone"})
check("free_attractions_needs_marker", rec is None)

# --- POI time windows ----------------------------------------------------------

NL_ARRIVE = (
    "I am traveling alone from Shenzhen to Shanghai for 2 days. Requirements: "
    "I must arrive at Shanghai Club (Hong Kong Metropolis Branch) no later "
    "than 11:00."
)
WRONG_ARRIVE = (
    "result=False\nfor activity in allactivities(plan):\n"
    "  if activity_position(activity)=='Shanghai Club (Hong Kong Metropolis "
    "Branch)':\n    if activity_end_time(activity)<='11:00': result=True"
)
q = {"nature_language": NL_ARRIVE, "target_city": "Shanghai"}
cons, rec = enforce_time_windows([WRONG_ARRIVE], q)
check(
    "arrive_rewritten_to_start_time",
    rec and "activity_start_time(activity)<='11:00'" in cons[0],
)

NL_LEAVE = (
    "We are 4 people traveling from Shenzhen to Shanghai for 3 days. "
    "Requirements: We hope to leave West Nanjing Road no earlier than 09:50."
)
WRONG_LEAVE = (
    "result=False\nfor activity in allactivities(plan):\n"
    "  if activity_position(activity)=='West Nanjing Road':\n"
    "    if activity_start_time(activity)>='09:50': result=True"
)
q = {"nature_language": NL_LEAVE, "target_city": "Shanghai"}
cons, rec = enforce_time_windows([WRONG_LEAVE], q)
check(
    "leave_rewritten_to_end_time",
    rec and "activity_end_time(activity)>='09:50'" in cons[0],
)

NL_DEPART_FROM = (
    "- Depart no earlier than 17:40 from Hawaii Global Seafood Artistry "
    "(High-Tech Branch)"
)
WRONG_UNIVERSAL = (
    "result=True\nfor activity in allactivities(plan):\n"
    "  if activity_position(activity)=='Hawaii Global Seafood Artistry "
    "(High-Tech Branch)':\n    if activity_start_time(activity)<'17:40': "
    "result=False"
)
q = {"nature_language": NL_DEPART_FROM, "target_city": "Chengdu"}
cons, rec = enforce_time_windows([WRONG_UNIVERSAL], q)
check(
    "vacuous_universal_rewritten_to_existential",
    rec
    and cons[0].startswith("result=False")
    and "activity_end_time(activity)>='17:40'" in cons[0],
)

# escaped apostrophe in the carrier constraint must round-trip cleanly
NL_APOS = (
    "with the following requirements: arrive at Su Xiaoxiao's Tomb by the "
    "Qiantang River no later than 12:20."
)
CARRIER = (
    "result=False\nfor activity in allactivities(plan):\n"
    "  if activity_position(activity)=='Su Xiaoxiao\\'s Tomb by the Qiantang "
    "River':\n    if activity_end_time(activity)<='12:20': result=True"
)
q = {"nature_language": NL_APOS, "target_city": "Hangzhou"}
cons, rec = enforce_time_windows([CARRIER], q)
check(
    "escaped_quote_name_roundtrip",
    rec
    and '"Su Xiaoxiao\'s Tomb by the Qiantang River"' in cons[0]
    and "\\'" not in cons[0],
)

# a 'between A and B' two-ended window must not be touched
BETWEEN = (
    "result=False\nfor activity in allactivities(plan):\n"
    "  if activity_position(activity)=='C Cafe':\n"
    "    if activity_start_time(activity)<='10:00' and "
    "activity_end_time(activity)>='11:30': result=True"
)
q = {
    "nature_language": "want to visit C Cafe between 10:00 and 11:30, and "
    "we hope to leave Qibao Ancient Town no earlier than 10:00.",
    "target_city": "Shanghai",
}
cons, rec = enforce_time_windows([BETWEEN], q)
check("between_window_untouched", BETWEEN in cons)

# --- distance-conditional taxi -------------------------------------------------

NL_TAXI = (
    "We are 2 people traveling from Hangzhou to Shanghai for 3 days, with "
    "the following requirements:\n- Do not want to get around the city by "
    "walking.\n- If the distance between two locations exceeds 4.8 km, take "
    "a taxi."
)
BAD_TAXI = (
    "result=True\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity)=='transportation':\n"
    "    transports=activity_transports(activity)\n"
    "    if innercity_transport_type(transports)!='taxi':\n"
    "      dist=innercity_transport_distance(transports)\n"
    "      if dist>4.8: result=False"
)
q = {"nature_language": NL_TAXI}
cons, rec = enforce_distance_taxi([BAD_TAXI], q)
check(
    "distance_taxi_canonicalized",
    rec
    and BAD_TAXI not in cons
    and any(
        "innercity_transport_type(activity_transports(activity)) != 'taxi' "
        "and innercity_transport_distance(activity_transports(activity))>4.8"
        in c
        for c in cons
    ),
)
cons2, rec2 = enforce_distance_taxi(list(cons), q)
check("distance_taxi_idempotent", rec2 is None)
cons, rec = enforce_distance_taxi(
    [], {"nature_language": "the hotel is 4.8 km from the airport"})
check("distance_taxi_needs_marker", rec is None)

# --- hotel distance -------------------------------------------------------------

NL_HOTEL = (
    "We are 4 people traveling from Shenzhen to Shanghai for 3 days, with "
    "the following requirements: Accommodation should be within 2.03 km of "
    "Nanpu Bridge."
)
BAD_HOTEL = (
    "result=True\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) == 'accommodation':\n"
    "    distance = poi_distance(target_city(plan), "
    "activity_position(activity), 'Nanpu Bridge')\n"
    "    if distance > 2.03:\n      result = False"
)
q = {"nature_language": NL_HOTEL, "target_city": "Shanghai"}
cons, rec = enforce_hotel_distance([BAD_HOTEL], q)
check(
    "hotel_distance_canonicalized",
    rec
    and BAD_HOTEL not in cons
    and any(
        "poi_distance(target_city(plan), 'Nanpu Bridge', "
        "accommodation_position)<=2.03" in c
        for c in cons
    ),
)
NL_HOTEL_DISJ = (
    "requiring one of the following: 1. total travel budget is 3700.0, or "
    "2. accommodation should be within 0.8 km of Pingjiang Road Historic "
    "District."
)
cons, rec = enforce_hotel_distance(
    [], {"nature_language": NL_HOTEL_DISJ, "target_city": "Suzhou"})
check("hotel_distance_skip_in_disjunction", rec is None)

# --- tautology + polarity --------------------------------------------------------

TAUT = (
    "accommodation_name_set = set()\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity) == 'accommodation':\n"
    "    accommodation_name_set.add(activity_position(activity))\n"
    "result=(not({'Howard Johnson Leonora Plaza Shanghai'} & "
    "accommodation_name_set) == False)"
)
cons, rec = fix_negation_tautology([TAUT])
check(
    "tautology_not_eq_false_fixed",
    rec
    and cons[0].endswith(
        "result=({'Howard Johnson Leonora Plaza Shanghai'} & "
        "accommodation_name_set)"
    ),
)
OK_NOT = (
    "result=(not (set() == attraction_type_set & {'Art museum'}))"
)
cons, rec = fix_negation_tautology([OK_NOT])
check("plain_not_expression_untouched", rec is None)

NL_POLARITY = (
    "We are 1 person, traveling from Nanjing to Shanghai for 3 days. The "
    "requirements are as follows: the budget for meals is 4700.0, and we "
    "hope to stay at one of the following hotels: Four Points by Sheraton "
    "Shanghai Kangqiao or Atour S hotel, Shanghai Wanyuan Road."
)
NEGATED = (
    "accommodation_name_set=set()\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity)=='accommodation':\n"
    "    accommodation_name_set.add(activity_position(activity))\n"
    "result=(not({'Four Points by Sheraton Shanghai Kangqiao', 'Atour S "
    "hotel, Shanghai Wanyuan Road'}&accommodation_name_set))"
)
cons, rec = fix_membership_polarity([NEGATED], {"nature_language": NL_POLARITY})
check(
    "positive_stay_polarity_flipped",
    rec and "result=({'Four Points" in cons[0] and "not(" not in cons[0],
)
NL_AVOID = (
    "with the following requirements:\nDo not want to stay at the following "
    "hotels: CM+ Service Apartment and Suzhou Central Hotel"
)
NEG_OK = (
    "accommodation_name_set=set()\nfor activity in allactivities(plan):\n"
    "  if activity_type(activity)=='accommodation':\n"
    "    accommodation_name_set.add(activity_position(activity))\n"
    "result=(not({'CM+ Service Apartment', 'Suzhou Central Hotel'}"
    "&accommodation_name_set))"
)
cons, rec = fix_membership_polarity([NEG_OK], {"nature_language": NL_AVOID})
check("negative_stay_polarity_kept", rec is None)

# the zero-cap free-attraction constraint must survive the full driver
# (drop_ungrounded_cost_caps must not treat cap 0 as an invented budget)
from chinatravel.agent.nesy_agent.constraint_coverage import enforce_coverage

q = {
    "nature_language": "One person traveling from Hangzhou to Shenzhen for "
    "3 days. Requirement: only visit free attractions.",
    "people_number": 1,
    "days": 3,
    "target_city": "Shenzhen",
    "start_city": "Hangzhou",
    "hard_logic_py": ["result=(people_count(plan)==1)"],
}
enforce_coverage(q, lang="en")
check(
    "free_attraction_survives_full_driver",
    any("attraction_cost<=0" in c for c in q["hard_logic_py"]),
)

# --- base boilerplate ------------------------------------------------------------

q = {"days": 2, "people_number": 2, "uid": "20250323142722224243"}
cons, rec = inject_base_boilerplate([], q)
check(
    "boilerplate_injected_on_empty",
    rec
    and any("day_count(plan)==2" in c for c in cons)
    and any("people_count(plan)==2" in c for c in cons)
    and any("activity_tickets(activity)!=2" in c for c in cons)
    and any("taxi_cars(activity_transports(activity))!=1" in c for c in cons),
)
cons2, rec2 = inject_base_boilerplate(list(cons), q)
check("boilerplate_idempotent", rec2 is None)

# --- round-4 interaction: 'a single bed room' is a room TYPE, not a count ---------

check(
    "single_bed_room_not_a_count",
    explicit_room_count("We hope to stay in a single bed room.") is None,
)
check(
    "twin_room_still_counts",
    explicit_room_count("the three of us stay in a twin room") == 1,
)

print("\n%d/%d tests passed" % (PASSED, PASSED + FAILED))
if FAILED:
    sys.exit(1)

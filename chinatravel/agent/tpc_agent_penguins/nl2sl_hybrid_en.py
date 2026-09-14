import os
import re
import sys
import traceback
from json_repair import repair_json

project_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)

import json
from tqdm import tqdm
from copy import deepcopy
from chinatravel.symbol_verification.concept_func import func_dict
from .sv_compat import normalize_hard_logic_constraint
from .prompts_en import NL2SL_INSTRUCTION
from .constraint_coverage import enforce_coverage
from chinatravel.agent.nesy_agent.ast_checker_en import HardLogicPyChecker
from chinatravel.data.load_datasets import save_json_file, load_json_file


func_docs = """
(1) day_count(plan)
Docs: Get the number of days in the plan.
Return: int
(2) people_count(plan)
Docs: Get the number of people in the plan.
Return: int
(3) start_city(plan)
Docs: Get the start city of the plan.
Return: str
(4) target_city(plan)
Docs: Get the target city of the plan.
Return: str
(5) allactivities(plan)
Docs: Get all the activities in the plan.
Return: list of activities
(6) allactivities_count(plan)
Docs: Get the number of activities in the plan.
Return: int
(7) dayactivities(plan, day)
Docs: Get all the activities in the specific day [1, 2, 3, ...].
Return: list of activities
(8) activity_cost(activity)
Docs: Get the cost of specific activity without transport cost.
Return: float
(9) activity_position(activity)
Docs: Get the position name of specific activity.
Return: str
(10) activity_price(activity)
Docs: Get the price of specific activity. The price is price per person.
Return: float
(11) activity_type(activity)
Docs: Get the type of specific activity. ['breakfast', 'lunch', 'dinner', 'attraction', 'accommodation', 'train', 'airplane']
Return: str
(12) activity_tickets(activity)
Docs: Get the number of tickets needed for specific activity. ['attraction', 'train', 'airplane']
Return: int
(13) activity_transports(activity)
Docs: Get the transport information of specific activity.
Return: list of dict
(14) activity_start_time(activity)
Docs: Get the start time of specific activity.
Return: str
(15) activity_end_time(activity)
Docs: Get the end time of specific activity.
Return: str
(16) activity_time(activity)
Docs: Get the duration of specific activity.
Return: int (minutes)
(17) innercity_transport_cost(transports)
Docs: Get the total cost of innercity transport.
Return: float
(18) poi_recommend_time(city, poi):
Docs: Get the recommend time of specific poi in the city. Only support attractions now.
Return: int (minutes)
(19) poi_distance(city, poi1, poi2):
Docs: Get the distance between two pois in the city.
Return: float (km)
(20) innercity_transport_price(transports)
Docs: Get the price of innercity transport. The price is price per person.
Return: float
(21) innercity_transport_distance(transports)
Docs: Get the distance of innercity transport.
Return: float (km)
(22) metro_tickets(transports)
Docs: Get the number of metro tickets if the type of transport is metro.
Return: int
(23) taxi_cars(transports)
Docs: Get the number of taxi cars if the type of transport is taxi. The number of taxi cars is `(people_count(plan) + 3) // 4`.
Return: int
(24) room_count(activity)
Docs: Get the number of rooms of accommodation activity.
Return: int
(25) room_type(activity)
Docs: Get the type of room of accommodation activity.
1 for single room, 2 for double room. Must be 1 or 2. Never use "double room" or "twin room" or other words but 1 or 2.
Return: int
(26) restaurant_type(activity, target_city)
Docs: Get the type of restaurant's cuisine in the target city. The return value must be in ['Yunnan cuisine', 'Tibetan cuisine', 'Northeastern Chinese cuisine', 'Barbecue', 'Asian cuisine', 'Cantonese cuisine', 'Northwestern Chinese cuisine', 'Fujian cuisine', 'Hakka cuisine', 'Fast food and casual dining', 'Sichuan cuisine', 'Taiwanese cuisine', 'Other', 'Halal cuisine', 'Snacks', 'Western cuisine', 'Vegetarian cuisine', 'Japanese cuisine', 'Jiangsu-Zhejiang cuisine', 'Hubei cuisine', 'Southeast Asian cuisine', 'Hunan cuisine', 'Beijing cuisine', 'Korean cuisine', 'Seafood', 'Middle Eastern cuisine', 'fusion cuisine', 'Teahouse', 'Bar/Pub', 'Creative Cuisine', 'buffet', 'coffee shop', 'Shanghai cuisine', 'Huizhou cuisine', 'Latin American cuisine', 'Shandong Cuisine', 'Xinjiang cuisine', 'Farmhouse cuisine', 'Hainan cuisine', 'Hot pot', 'Bakery and Desserts', 'Other Chinese Cuisine'].
Return: str
(27) attraction_type(activity, target_city)
Docs: Get the type of attraction in the target city. The return value must be in ['Museum/Memorial Hall', 'Art museum', 'Red tourism sites', 'natural scenery', 'Cultural Landscape', 'University campus', 'historical site', 'Amusement Park/Sports Entertainment', 'Garden', 'Other', 'Cultural Tourism Area', 'park', 'commercial district'].
Return: str
(28) accommodation_type(activity, target_city)
Docs: Get the feature of accommodation in the target city to judge whether it's feature meets the user's requirement. The return value must be in ["Kids' Club", 'Air purifier', 'Mountain View Room', 'Private Hot Spring Room', 'Courtyard house', 'hot spring', 'Lakeside Residence', 'e-sports hotel', 'Hot spring bathing', 'Executive Lounge', 'Charging station', 'Designer hotel', 'homestay', 'Lake View Room', 'Stunning Night Views', 'Luggage Storage', 'Chinese-style courtyard', 'Billiards Room', 'Private Pool', 'Fishing', 'Charming sea view', 'Garden Architecture', 'Old Western-style house', "Children's Pool", 'Historic Residence', 'Mahjong and Card Game Room', 'Smart Room Control', "Couple's Room", 'small and beautiful', 'Tea Room', 'Family-themed room', 'Multifunction Hall', 'Laundry room', 'inn', 'Self-operated family room', 'Parking lot', 'Recommended by the Boss', 'River view room', 'Sunbathing area', 'Self-operated entertainment room', 'Kitchen', 'Air conditioning', 'Instagrammable pool', 'Villa', 'Free parking', 'Laundry service', 'Great view from the window', 'Serviced Apartment', 'Conference Hall', 'Family Room', '24-hour front desk', 'Business Center', 'Early Park Entry', 'Farm stay', 'Smart toilet', 'Gourmet Hotel', 'Spa', 'Photogenic', 'Ocean View Room', 'Swimming Pool', 'Media Room', 'Butler Service', 'Airport shuttle service', 'Sauna', 'Robot Service', "Children's Playground", 'Fitness Room', 'Washing machine', 'Self-operated Comfort Sleep Room', 'Pet-friendly', 'e-sports room', 'Excellent location', 'Suite'].
Return: str
(29) innercity_transport_type(transports)
Docs: Get the type of innercity transport. The return value must be in ['metro', 'taxi', 'walk'].
Return: str
(30) innercity_transport_start_time(transports)
Docs: Get the start time of innercity transport.
Return: str
(31) innercity_transport_end_time(transports)
Docs: Get the end time of innercity transport.
Return: str
(32) intercity_transport_type(activity)
Docs: Get the type of intercity transport. The return value must be in ['train', 'airplane'].
Return: str
(33) innercity_transport_time(transports)
Docs: Get the duration of innercity transport.
Return: int (minutes)
(34) intercity_transport_origin(activity)
Docs: Get the origin city of intercity transport.
Return: str
(35) intercity_transport_destination(activity)
Docs: Get the destination city of intercity transport.
Return: str
"""

sl_trans_prompt = (
    """
We offer some functions below, try to translate the constraints in nature language into python code and output them in json list format.
variables:
(1) plan: a dict of the generated plan with information of the specific plan.

functions:"""
    + func_docs
    + """
You need to response in the following format:
[
    "python code block 1",
    "python code block 2",
    ...
]

Not all the constraints need to be translated into python code. Ignore them if they can not be translated into legal python code.
!!! Only `plan` variable can be used directly in the python code. Others must be defined in the python code use the functions we offer above. !!! Pay attention to the return TYPE of functions!!!
For most case, for exist constraints, you can set `result=False` at the beginning of the code, and then set `result=True` if the condition is satisfied. For all constraints, you can set `result=True` at the beginning of the code, and then set `result=False` if the condition is not satisfied.

### HARD RULES (code violating these crashes or is rejected)
1. The executor exposes ONLY the builtin `set`. len, bool, any, all, sum, str, int, float, map, sorted, abs, max, min are NOT defined and raise NameError. Express non-emptiness as result=(A&B), emptiness/negation as result=not(A&B), subset as A<=B, and counts with an explicit counter variable incremented inside the loop.
2. Each string in the output list is executed independently in a fresh namespace: it must be fully self-contained and assign `result`. Never reference a variable defined in another string. 'at least one of / either ... or' requirements must become ONE constraint whose sub-conditions are computed in the same code block and combined with `or`.
3. Copy every POI name VERBATIM from the request as a single string: the exact substring including parentheses, '·', branch suffixes and spacing. Never split one name on internal separators, never translate, shorten or normalize it. Only split a list of POIs on explicit delimiters (commas / 'and') that separate obviously distinct venues.
4. ALWAYS emit the base constraints exactly as in the example: days, people, the tickets constraint (attraction/airplane/train tickets and metro tickets == people number). Emit the taxi_cars constraint when the request is free prose, or when a requirement line mentions taxis (e.g. 'Whenever taking a taxi, use 1 taxi' -> taxi_cars(activity)!=1 loop); in the NUMBERED-REQUIREMENTS register with no taxi line, omit taxi_cars. NEVER emit room_count/room_type or any accommodation constraint unless the request explicitly mentions rooms, beds, bed type or a hotel requirement: no `room_count(activity)!=N` or `room_type(activity)!=N` check in any form, standalone or inside another loop. 'Each accommodation stay must reserve N rooms' IS an explicit room requirement -> the room_count(activity)!=N loop.
5. Before answering, self-check: every requirement clause of the request maps to exactly one constraint; no constraint lacks a source in the request (base constraints excepted); no forbidden builtin appears. NUMBERED-REQUIREMENTS register ('Requirements:' followed by '1. ... 2. ...'): EVERY numbered line maps to exactly ONE constraint, in order ('The trip must last N days' -> day_count, 'plan for N travelers' -> people_count, 'Tickets ... must match N travelers' -> the tickets constraint); never merge, split or silently skip a numbered line -- including transport-mode lines like 'Use only metro and taxi within the destination city' (blacklist of the missing modes) and 'Do not use walking for transportation within the destination city' (blacklist of 'walk').
6. Colloquial idioms are HARD requirements: 'taste/try the local specialties/cuisine' requires the target city's signature cuisine in restaurant_type_set (Beijing->'Beijing cuisine', Shanghai->'Shanghai cuisine', Nanjing/Suzhou/Hangzhou->'Jiangsu-Zhejiang cuisine', Guangzhou/Shenzhen->'Cantonese cuisine', Chengdu/Chongqing->'Sichuan cuisine', Wuhan->'Hubei cuisine'). Any mention of airfare / air tickets (机票) means intercity transport must be airplane. An explicit room-count phrase ('a twin room', 'one room', 'two rooms', '一间') overrides the default room count: 'the three of us stay in a twin room' means ONE room (room_count==1, room_type==2), not three.

### CANONICAL PATTERNS (copy these shapes exactly)
- must visit/eat/stay at X: build the name set over the right activity types, then result=({'X'}<=name_set)
- 'one of / any of X, Y': result=({'X','Y'}&name_set)  (ONE constraint; intersection &, not <=)
- 'do not want / avoid X': result=not({'X'}&name_set)
- intra-city travel mode preference (no taxi / no walking / prefer or only metro ...): ALWAYS write the forbidden modes as this blacklist ('only metro' forbids walk and taxi):
"inner_city_transportation_set=set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='transportation': inner_city_transportation_set.add(activity_position(activity))\nresult=not({'walk', 'taxi'}&inner_city_transportation_set)"
NEVER use innercity_transport_type or activity_transports for mode preferences, and never turn a blacklist into a whitelist.
- 'Visit/Dine at/Stay at X between A and B' (or 'from A to B'): the activity must fall INSIDE the window, i.e. start no earlier than A AND end no later than B, with the activity-type guard picked by the verb/venue: 'Visit' (attraction) -> activity_type(activity)=='attraction'; 'Dine at' (restaurant or hotel meal) -> activity_type(activity) in ['breakfast', 'lunch', 'dinner']; 'Stay at' (hotel check-in) -> activity_type(activity)=='accommodation':
"result=False\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction' and activity_position(activity)=='X':\n    if activity_start_time(activity)>='A' and activity_end_time(activity)<='B': result=True"
Do NOT write activity_start_time<='A' and activity_end_time>='B' (that inverted form demands the visit COVER the window and fails within-window plans).
- budget caps are SCOPED - map the budget noun to its own aggregation, NEVER to the total_cost pattern: meal/dining/food budget -> restaurant_cost accumulating activity_cost over ['breakfast','lunch','dinner']; accommodation/hotel budget -> accommodation_cost over 'accommodation'; sightseeing/attraction budget -> attraction_cost over 'attraction'; inter-city/cross-city transportation budget -> inter_city_transportation_cost accumulating activity_cost over ['airplane','train']; intra-city / within-the-city / local transportation budget -> inner_city_transportation_cost accumulating innercity_transport_cost(activity_transports(activity)) over ALL activities with NO activity_type filter. Each ends with result=(accumulator<=CAP). ONLY an explicit 'total/overall (travel) budget' uses the total_cost pattern in the example; translating a scoped budget as total_cost makes the query unsatisfiable.
- 'only (want to) visit free attractions': "attraction_cost=0\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction': attraction_cost+=activity_cost(activity)\nresult=attraction_cost<=0"
- DIRECTIONAL intercity modes - 'take MODE to the destination' / 'return by MODE' (also the negated 'do not want to ...') constrain ONLY the first/last activity, never the global mode set:
"result=False\nintercity_transport_go=''\nintercity_transport_back=''\nif allactivities(plan)[0]['type'] == \\"train\\" and intercity_transport_origin(allactivities(plan)[0])==start_city(plan) and allactivities(plan)[-1]['type'] == \\"airplane\\" and intercity_transport_origin(allactivities(plan)[-1])==target_city(plan):\n  result=True"
(use != for negated wishes; drop the leg that is not mentioned). When direction words ('to the destination', 'return', 'back') are present, a global intercity_transport_set constraint is WRONG and often contradictory.
- 'arrive at X no later than T' -> existential on the START time: "result=False\nfor activity in allactivities(plan):\n  if activity_position(activity)=='X':\n    if activity_start_time(activity)<='T':\n      result=True". 'leave/depart (from) X no earlier than T' -> the same shape with activity_end_time(activity)>='T'. Never swap start/end and never write the vacuous universal (result=True ...) form.
- 'if the distance between two locations exceeds D km, take a taxi': "result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity)) != 'taxi' and innercity_transport_distance(activity_transports(activity))>D:\n    result=False\n    break"
- 'accommodation within D km of X': "result=False\naccommodation_position=''\nfor activity in allactivities(plan):\n  if activity_type(activity)=='accommodation': accommodation_position=activity_position(activity)\nresult=(poi_distance(target_city(plan), 'X', accommodation_position)<=D)"

### DISJUNCTION ('must meet at least/any one of the following', 'either of the following')
Such a request lists numbered branches but the plan only has to satisfy ONE of them. Translate it as exactly ONE self-contained constraint that computes EVERY branch condition in the same code block and combines them with `or`.
NL: '... must meet at least one of the following: 1. Do not wish to visit Xidan Commercial Street and Dingling Mausoleum; 2. Accommodation budget is 3300.0.'
CORRECT (one constraint):
"attraction_name_set=set()\nhotel_cost=0\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction': attraction_name_set.add(activity_position(activity))\n  if activity_type(activity)=='accommodation': hotel_cost+=activity_cost(activity)\nbranch_1=not({'Xidan Commercial Street', 'Dingling Mausoleum'}&attraction_name_set)\nbranch_2=(hotel_cost<=3300.0)\nresult=(branch_1 or branch_2)"
WRONG - collapsed (forbidden): emitting only one branch, e.g. just "result=(hotel_cost<=3300.0)", turns 'at least one of' into an unconditional hard requirement.
WRONG - split (forbidden): emitting branch_1 and branch_2 as two separate list items means BOTH must hold (AND semantics).

### Attention!!!
If you find some pesucode in the nature language constraints is not defined in the functions we offer above, you must translate them into python block code with the functions we offer above. Usually, for attractions and restaurants, if the required one exists, the requirement is satisfied. However, for accommodation, people usually stay in the same hotel for the whole trip, so we need check all the accommodation activities in the plan.
###

if you find some error in nature language constraints, you need to fix them in the code block. if {'natural landscape'} <= spot_type, you need to change it to 'natural scenery' in the code block as we offer above.

Example:
nature_language:
days==2
people_number==3
cost<=3000
tickets==3
{'Beijing cuisine'}<=food_type
intercity_transport=={'train'}
{'natural scenery', 'Museum/Memorial Hall'}<=spot_type
{'Smart Room Control'}<=hotel_feature
hotel_price<=500
{'Beijing Quanjude (Qianmen Branch)'} <= restaurant_names
food_price<=100
transport_type<={'metro', 'taxi'}
{'The Palace Museum'}<=attraction_names
taxi_cars==1
answer:
[
"result=(day_count(plan)==2)",
"result=(people_count(plan)==3)",
"total_cost=0\nfor activity in allactivities(plan): total_cost+=activity_cost(activity)+innercity_transport_cost(activity_transports(activity))\nresult=(total_cost<=3000)",
"result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=2: result=False\n  if innercity_transport_type(activity_transports(activity))=='metro' and metro_tickets(activity_transports(activity))!=2: result=False",
"result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity))=='taxi' and taxi_cars(activity_transports(activity))!=1: result=False",
"result=True\nfor activity in allactivities(plan):\n  if activity_type(activity)=='accommodation' and accommodation_type(activity, target_city(plan))!='Smart Room Control': result=False\n  if activity_type(activity)=='accommodation' and activity_price(activity)>500: result=False",
"restaurant_type_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n    restaurant_type_set.add(restaurant_type(activity, target_city(plan)))\nresult=({'Beijing cuisine'}<=restaurant_type_set)",
"attraction_type_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction':\n    attraction_type_set.add(attraction_type(activity, target_city(plan)))\nresult=({'natural scenery', 'Museum/Memorial Hall'}<=attraction_type_set)",
"intercity_transport_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['train', 'airplane']:\n    intercity_transport_set.add(activity_type(activity))\nresult=(intercity_transport_set=={'train'})",
"restaurant_names_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n    restaurant_names_set.add(activity_position(activity))\nresult=({'Beijing Quanjude (Qianmen Branch)'}<=restaurant_names_set)",
"result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['breakfast', 'lunch', 'dinner'] and activity_price(activity)>100: result=False",
"inner_city_transportation_set=set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='transportation': inner_city_transportation_set.add(activity_position(activity))\nresult=not({'walk'}&inner_city_transportation_set)",
"attraction_names_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction':\n    attraction_names_set.add(activity_position(activity))\nresult=({'The Palace Museum'}<=attraction_names_set)",
]
"""
)

reflect_prompt = (
    """
we offer some functions below, try to reflect on the python code block and fix them and output in the same format.
[
"python code block 1",
"python code block 2",
...
]
We offer functions below:"""
    + func_docs
    + """
Try to fix the error in the code block and output them in json list format.
The attractions_type, restaurants_type, and accommodations_type must be in the list we offer above. You must trans the original type to !!!a similar one!!! we offer if the original type is not in the list we offer above.

For return value of activity_position(activity), it will be checked by whether the position is in the database. You need to trans it to a similar one if it is rufused with you own knowledge. Also hotel_names should be checked by activity_position(activity), not accommodation_type(activity, target_city(plan)) and so for other names.
Usually, for attractions and restaurants, if the required one exists, the requirement is satisfied. However, for accommodation, people usually stay in the same hotel for the whole trip, so we need check all the accommodation activities in the plan. Either change the function or value to make the code block correct.

Fix rules:
- The executor exposes ONLY the builtin `set`. len, bool, any, all, sum, str, int, float, map, sorted are NOT defined and raise NameError. Rewrite non-emptiness as result=(A&B), emptiness/negation as result=not(A&B), subset as A<=B, counts with a counter variable incremented in the loop.
- Each code block runs independently in a fresh namespace: it must be self-contained and assign `result`; never reference a variable defined in another block. Merge 'at least one of / either' alternatives into ONE block combined with `or`.
- Never add room_count/room_type or accommodation constraints unless the request explicitly mentions rooms or beds. Keep POI names verbatim as they appear in the request.
- NEVER rewrite a block into any()/all()/sum()/comprehension style: that is exactly what raises NameError here. Rewrite errored blocks in PLAIN LOOP style only. Example fix:
  errored: "result=any(activity_type(a)=='attraction' for a in allactivities(plan))"  (NameError: name 'any' is not defined)
  correct: "result=False\\nfor activity in allactivities(plan):\\n  if activity_type(activity)=='attraction': result=True"
- Keep each block's polarity unchanged while fixing: a requirement ('must', 'Dine at', 'Visit') stays init result=False / set True on hit; a prohibition ('Do not', 'avoid') stays init result=True / set False on violation. Never flip one into the other while repairing syntax.

You must output the whole code block. Including those constraints that are correct.
The original code block is:
"""
)


# ---------------------------------------------------------------------------
# Deterministic post-processing normalizers (phase-2 mechanical backstops).
# Generic, pattern-keyed rewrites of LLM-emitted constraints; never keyed on
# uids. Two families:
#   1. transport-mode idiom: intra-city mode preferences (no taxi / only
#      metro ...) must use the benchmark's canonical (vacuous) idiom over
#      activity_type=='transportation', never innercity_transport_type.
#   2. count boilerplate: the base tickets / metro_tickets / taxi_cars
#      constraints must compare against people_count (taxi cars:
#      (people+3)//4); the LLM sometimes miscomputes the taxi-car count or
#      rewrites the tickets constraint as a sum.
# ---------------------------------------------------------------------------

_MODE_RE = r"['\"](walk|taxi|metro)['\"]"
_COUNT_TOKENS = ("_tickets", "_cars", "tickets(", "cars(")
_MODE_ORDER = ("walk", "taxi", "metro")

# A transport-mode rewrite must never fire on a block that also computes a
# non-mode condition (e.g. one branch of a disjunction): rewriting would
# silently delete the other branch.
_NON_MODE_HINT_RE = re.compile(
    r"['\"](?:breakfast|lunch|dinner|attraction|accommodation)['\"]"
    r"|activity_cost\(|activity_price\(|activity_position\("
    r"|room_(?:count|type)\(|restaurant_type\(|attraction_type\("
    r"|accommodation_type\(|activity_(?:start_|end_)?time\("
)

_CANON_MODE_HEADER = (
    "inner_city_transportation_set=set()\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='transportation': "
    "inner_city_transportation_set.add(activity_position(activity))\n"
)


def _modes_in(text):
    return set(re.findall(_MODE_RE, text))


def _canonical_mode_constraint(modes, whitelist):
    ordered = [m for m in _MODE_ORDER if m in modes]
    literal = "{" + ", ".join("'%s'" % m for m in ordered) + "}"
    if whitelist:
        return _CANON_MODE_HEADER + "result=(inner_city_transportation_set<=%s)" % literal
    return _CANON_MODE_HEADER + "result=not(%s&inner_city_transportation_set)" % literal


def normalize_transport_mode_constraint(constraint):
    """Rewrite intra-city transport-MODE constraints to the canonical idiom.

    Blacklist NL ('do not want to walk / take taxi') ->
        result=not({modes}&inner_city_transportation_set)
    Whitelist NL ('only/prefer metro') ->
        result=(inner_city_transportation_set<={modes})
    Anything that is not clearly a mode constraint passes through unchanged:
    count boilerplate (taxi_cars / metro_tickets / activity_tickets),
    intercity constraints (train / airplane), and constraints already in the
    canonical activity_type=='transportation' idiom.
    """
    c = constraint
    if any(tok in c for tok in _COUNT_TOKENS):
        return constraint
    if "intercity_transport" in c or re.search(r"['\"](train|airplane)['\"]", c):
        return constraint
    if re.search(r"activity_type\(activity\)\s*==\s*['\"]transportation['\"]", c):
        return constraint
    if "innercity_transport_type" not in c:
        return constraint
    if not re.search(_MODE_RE, c):
        return constraint
    if _NON_MODE_HINT_RE.search(c):
        # mixed-domain block (e.g. a disjunction with a mode branch): leave it
        return constraint

    # whitelist: subset comparison against a mode collection (set or list)
    m = re.search(r"<=\s*(\{[^{}]*\}|\[[^\[\]]*\])", c)
    if m:
        wl = _modes_in(m.group(1))
        if wl:
            return _canonical_mode_constraint(wl, whitelist=True)
    # whitelist: equality of a collected set with a mode-set literal
    m = re.search(r"==\s*(\{[^{}]*\})", c)
    if m:
        wl = _modes_in(m.group(1))
        if wl:
            return _canonical_mode_constraint(wl, whitelist=True)
    # blacklist: negated intersection  not({'walk','taxi'}&transport_set)
    if re.search(r"\bnot\s*\(", c) and "&" in c:
        bl = _modes_in(c)
        if bl:
            return _canonical_mode_constraint(bl, whitelist=False)
    # existence intersection without negation ('prefer metro') -> whitelist
    if "&" in c and "not" not in c:
        wl = _modes_in(c)
        if wl:
            return _canonical_mode_constraint(wl, whitelist=True)

    # line-wise scan of ==/!=/in conditions steering result=True/False
    lines = c.split("\n")
    wl, bl = set(), set()
    for i, line in enumerate(lines):
        lm = _modes_in(line)
        if not lm:
            continue
        compact = line.replace(" ", "")
        # locate the result assignment governed by this condition: same line
        # or the following lines up to the next mode-bearing condition
        tail = compact
        j = i
        while "result=" not in tail and j + 1 < len(lines):
            j += 1
            nxt = lines[j]
            if _modes_in(nxt):
                break
            tail = nxt.replace(" ", "")
        sets_false = "result=False" in tail
        sets_true = "result=True" in tail
        negated = ("!=" in compact) or ("notin" in compact)
        if sets_false and not sets_true:
            (wl if negated else bl).update(lm)
        elif sets_true and not sets_false:
            (bl if negated else wl).update(lm)
    if bl and not wl:
        return _canonical_mode_constraint(bl, whitelist=False)
    if wl and not bl:
        return _canonical_mode_constraint(wl, whitelist=True)
    return constraint


_CANON_TICKETS_TMPL = (
    "result=True\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['attraction', 'airplane', 'train'] "
    "and activity_tickets(activity)!={n}: result=False\n"
    "  if innercity_transport_type(activity_transports(activity))=='metro' "
    "and metro_tickets(activity_transports(activity))!={n}: result=False"
)

# call with one nesting level of parentheses, e.g. f(activity_transports(activity))
_CALL = r"\((?:[^()]|\([^()]*\))*\)"


def normalize_count_boilerplate(constraint, people_count):
    """Fix the people-derived counts in the base ticket/taxi boilerplate.

    The benchmark boilerplate always checks `!= N` where N is people_count
    for activity/metro tickets and (people_count+3)//4 for taxi cars. The
    LLM occasionally miscomputes the taxi-car count, or rewrites the tickets
    constraint as an accumulated sum; both are deterministically repairable
    from the query's people_number.
    """
    if not people_count:
        return constraint
    c = constraint
    # sum-form tickets constraint -> canonical per-activity boilerplate
    if (
        "activity_tickets" in c
        and "+=" in c
        and "activity_cost" not in c
        and "activity_price" not in c
        and re.search(r"result\s*=\s*\(?\s*\w+\s*==\s*\d+", c)
    ):
        return _CANON_TICKETS_TMPL.format(n=people_count)
    taxi_cars_n = (people_count + 3) // 4
    c = re.sub(
        r"(taxi_cars\s*%s\s*!=\s*)\d+" % _CALL,
        lambda m: m.group(1) + str(taxi_cars_n),
        c,
    )
    c = re.sub(
        r"(metro_tickets\s*%s\s*!=\s*)\d+" % _CALL,
        lambda m: m.group(1) + str(people_count),
        c,
    )
    c = re.sub(
        r"(activity_tickets\s*%s\s*!=\s*)\d+" % _CALL,
        lambda m: m.group(1) + str(people_count),
        c,
    )
    return c


# ---------------------------------------------------------------------------
# Room-constraint strip: the LLM keeps inventing room_count/room_type checks
# (rooms==N, room_count(activity)!=N, ...) for requests that never mention
# rooms or beds. Mechanically remove them unless the NL licenses them.
# ---------------------------------------------------------------------------

_ROOM_FUNC_RE = re.compile(r"\broom_(?:count|type)\s*\(")
_ROOM_NL_RE = re.compile(
    r"room|bed|单人间|双人间|标间|床|房间|客房|房型|开[一两二三四五六七八九十\d]+间",
    re.IGNORECASE,
)

# functions whose presence makes a pruned block still worth keeping
_DOMAIN_FUNC_RE = re.compile(
    r"\b(?:activity_(?:cost|price|position|tickets|time|start_time|end_time)"
    r"|restaurant_type|attraction_type|accommodation_type"
    r"|innercity_transport_(?:cost|price|distance|time|type)"
    r"|intercity_transport_\w+|metro_tickets|taxi_cars"
    r"|day_count|people_count|poi_(?:recommend_time|distance))\s*\("
)


def _is_meaningful_block(code):
    """True if `code` still checks something real and compiles."""
    if "result" not in code or not _DOMAIN_FUNC_RE.search(code):
        return False
    try:
        compile(code, "<constraint>", "exec")
    except SyntaxError:  # includes IndentationError from dangling headers
        return False
    return True


def strip_room_constraints(constraints, nature_language):
    """Drop room_count/room_type checks the request never asked for.

    Guard: if the NL mentions rooms/beds (en or zh) everything passes through
    untouched. Otherwise room_count/room_type lines are pruned; a block whose
    remainder no longer checks anything meaningful is dropped entirely.
    """
    if not nature_language or _ROOM_NL_RE.search(nature_language):
        return list(constraints)
    out = []
    for c in constraints:
        if not (isinstance(c, str) and _ROOM_FUNC_RE.search(c)):
            out.append(c)
            continue
        pruned = "\n".join(
            ln for ln in c.split("\n") if not _ROOM_FUNC_RE.search(ln)
        )
        if _is_meaningful_block(pruned):
            out.append(pruned)
        # else: constraint only checked rooms -> drop it completely
    return out


def normalize_generated_constraints(constraints, people_count=None, nature_language=None):
    """Apply all deterministic normalizers to an LLM-emitted constraint list."""
    # Style canonicalizer shared with the V6 cache-load path: rewrites the
    # open-weight DSL dialect (set-variable spellings, budget accumulator
    # names) into the oracle dialect without changing evaluation semantics.
    from .dsl_canonicalizer import (
        canonicalize_hard_logic_py,
        _repair_transport_type_guards,
    )

    out = []
    for c in constraints:
        if not isinstance(c, str):
            out.append(c)
            continue
        c2 = normalize_transport_mode_constraint(c)
        if c2 == c:
            c2 = normalize_count_boilerplate(c, people_count)
        # multi-branch (or-combined) blocks: the style canonicalizer classifies
        # accumulators by a single activity-type filter and would misname
        # branch accumulators (e.g. hotel_cost -> attraction_cost); skip it,
        # but still repair vacuous 'transportation' type guards -- a guarded
        # branch otherwise sums nothing and passes vacuously (round 5).
        if _or_arity(c2) == 0:
            c2 = canonicalize_hard_logic_py(c2)
        else:
            c2 = _repair_transport_type_guards(c2)
        out.append(c2)
    out = strip_room_constraints(out, nature_language)
    # normalization can collapse variants into duplicates
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Disjunction verifier: NL requests of the form 'must meet at least/any one of
# the following: 1. ... 2. ...' must yield ONE constraint OR-ing all branches.
# The LLM tends to either collapse to a single unconditional branch (turning a
# soft alternative into a hard, often infeasible cap) or split the branches
# into separate constraints (AND semantics). Detect the pattern in the NL,
# check the emitted constraints for a matching `or`, and if absent inject a
# targeted reflection turn asking for the single disjunction constraint.
# ---------------------------------------------------------------------------

_DISJ_MARKER_PATTERNS = (
    # English, top-level 'at least/any one of' phrasings
    r"at least one of",
    r"any one of (?:the following|them)",
    r"\(any one(?:\s+of(?:\s+them)?)?\s*\)?\s*:",
    r"\(any one(?:\s+of(?:\s+them)?)?\)",
    r"any of the following",
    r"either of the following",
    r"(?:satisf\w+|meet(?:ing)?|met|fulfill?(?:ing)?|requir\w+|need(?:s|ing|ed)?|that)\s+either\b",
    r"(?:satisf\w+|meet(?:ing)?|met|fulfill?(?:ing)?|requir\w+|need(?:s|ing|ed)?|with)\s+"
    r"(?:at least\s+|any\s+)?one of the following",
    r"one of the following(?:\s+\w+){0,2}\s+must be (?:met|satisfied)",
    # Chinese (held-out language unknown; cover common phrasings)
    r"满足以下(?:要求|条件)?(?:中的)?(?:至少|任意|任一)?一(?:个|项|条)",
    r"(?:至少|任意|任一)满足(?:以下|下列|其中)之?一",
    r"满足(?:以下|下列)(?:要求|条件)?(?:中的)?任(?:意|一)",
    r"(?:以下|下列)(?:要求|条件)(?:至少)?满足(?:其中)?(?:任意|任一)?一(?:个|项|条)?",
    r"任选其一|满足其一|其中之一即可",
)
_DISJ_MARKER_RE = re.compile(
    "|".join("(?:%s)" % p for p in _DISJ_MARKER_PATTERNS), re.IGNORECASE
)
# enumerated branch labels: '1. ', '(2) ', '3、'; never decimals like 4500.0
_BRANCH_LABEL_RE = re.compile(r"(?<![\d.])([1-9])\s*(?:[.)]\s|、)")
_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")


def _or_arity(code):
    """Number of python `or` operators in code (string literals ignored)."""
    if not isinstance(code, str):
        return 0
    return len(re.findall(r"\bor\b", _STRING_LITERAL_RE.sub("", code)))


def detect_disjunction(nature_language):
    """Return {'marker', 'branch_count'} if the NL contains a top-level
    disjunction marker, else None. branch_count is the length of the
    consecutive 1..k run of enumerated branch labels found in the NL."""
    if not nature_language:
        return None
    m = _DISJ_MARKER_RE.search(nature_language)
    if not m:
        return None
    labels = {int(x) for x in _BRANCH_LABEL_RE.findall(nature_language)}
    k = 0
    while (k + 1) in labels:
        k += 1
    if k < 2 and "either" in m.group(0).lower():
        # inline disjunction without enumeration: 'require either A or B'
        if re.search(r"\bor\b", nature_language[m.end():], re.IGNORECASE):
            k = 2
    return {"marker": m.group(0), "branch_count": k}


def disjunction_gap(nature_language, constraints):
    """Return gap info when the NL demands an OR that the constraints lack.

    A disjunction over k enumerated branches needs ONE constraint containing
    at least k-1 `or` operators. Requires k >= 2 so that inline 'one of X, Y'
    set-intersection idioms (no numbered branches) never trigger.
    """
    info = detect_disjunction(nature_language)
    if not info or info["branch_count"] < 2:
        return None
    need = info["branch_count"] - 1
    found = max((_or_arity(c) for c in constraints), default=0)
    if found >= need:
        return None
    info["required_or_arity"] = need
    info["found_or_arity"] = found
    return info


# --- merging the re-emitted disjunction into the constraint list -----------

_SIG_STRING_RE = re.compile(r"'([^']{2,})'|\"([^\"]{2,})\"")
_SIG_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")
# activity-type filter strings appear as scaffolding in almost every
# constraint and carry no requirement identity
_SIG_STOPWORDS = {
    "breakfast", "lunch", "dinner", "attraction", "accommodation",
    "transportation",
}


def _constraint_signature(code):
    """Requirement-identifying literals: POI names, budget numbers, modes."""
    sig = {a or b for a, b in _SIG_STRING_RE.findall(code)}
    sig -= _SIG_STOPWORDS
    for n in _SIG_NUMBER_RE.findall(_STRING_LITERAL_RE.sub("", code)):
        if float(n) >= 10:
            sig.add(n)
    return sig


def _is_base_constraint(code):
    compact = code.replace(" ", "")
    if "day_count(plan)==" in compact or "people_count(plan)==" in compact:
        return True
    if "activity_tickets" in code and "metro_tickets" in code:
        return True
    if "taxi_cars" in code:
        return True
    return False


def merge_disjunction_constraint(constraints, disjunction):
    """Add `disjunction`, dropping earlier lone-branch translations of it.

    A previously emitted non-base, or-free constraint is considered a branch
    fragment (collapsed or split branch) when it shares requirement literals
    with the disjunction, or - if it has no identifying literals at all -
    when every domain function it uses also appears in the disjunction.
    """
    dsig = _constraint_signature(disjunction)
    dfuncs = set(_DOMAIN_FUNC_RE.findall(disjunction))
    out = []
    for c in constraints:
        if isinstance(c, str) and not _is_base_constraint(c) and _or_arity(c) == 0:
            csig = _constraint_signature(c)
            if csig & dsig:
                continue
            cfuncs = set(_DOMAIN_FUNC_RE.findall(c))
            if not csig and cfuncs and cfuncs <= dfuncs:
                continue
        out.append(c)
    out.append(disjunction)
    return list(dict.fromkeys(out))


disjunction_reflect_header = """
The request contains an OR-requirement (marker: "{marker}", {k} numbered branches): the plan only has to satisfy AT LEAST ONE of the numbered branches, not all of them.
Rule: such a requirement must be translated into exactly ONE self-contained python constraint that computes EVERY branch condition inside the same code block and combines them with `or`.
WRONG - collapsed (forbidden): emitting only one branch as its own constraint turns 'at least one of' into an unconditional hard requirement.
WRONG - split (forbidden): emitting the branches as separate constraints means ALL of them must hold (AND semantics).
Worked example:
NL: '... must meet at least one of the following: 1. Do not wish to visit Xidan Commercial Street and Dingling Mausoleum; 2. Accommodation budget is 3300.0.'
CORRECT single constraint:
"attraction_name_set=set()\\nhotel_cost=0\\nfor activity in allactivities(plan):\\n  if activity_type(activity)=='attraction': attraction_name_set.add(activity_position(activity))\\n  if activity_type(activity)=='accommodation': hotel_cost+=activity_cost(activity)\\nbranch_1=not({{'Xidan Commercial Street', 'Dingling Mausoleum'}}&attraction_name_set)\\nbranch_2=(hotel_cost<=3300.0)\\nresult=(branch_1 or branch_2)"
Executor rules: only the builtin `set` exists (no len/any/all/sum); the block must be fully self-contained and assign `result`; copy POI names verbatim from the request.
The available functions are:"""

disjunction_reflect_tail = (
    "\nRe-emit ONLY the corrected disjunction constraint, as a json list "
    "containing exactly one string.\nThe request is:\n"
)


# ---------------------------------------------------------------------------
# Instruction-language switch (phase-2 H1 probe). PENGUINS_PROMPT_LANG=zh
# swaps the INSTRUCTION/explanation text of every prompt in this module for
# natural-Chinese renderings of the same hardened rules / few-shots /
# checklists (prompts_zh_instr.py). The DSL itself is untouched: python code
# shapes, function docs, value vocabularies, few-shot examples, POI handling
# and output format stay English, and the deterministic normalizers /
# verifiers below run identically. Default ("en" or unset) changes nothing.
# ---------------------------------------------------------------------------
if os.environ.get("PENGUINS_PROMPT_LANG", "en").strip().lower() == "zh":
    from . import prompts_zh_instr as _zh_instr

    NL2SL_INSTRUCTION = _zh_instr.NL2SL_INSTRUCTION
    sl_trans_prompt = _zh_instr.build_sl_trans_prompt(func_docs)
    reflect_prompt = _zh_instr.build_reflect_prompt(func_docs)
    disjunction_reflect_header = _zh_instr.disjunction_reflect_header
    disjunction_reflect_tail = _zh_instr.disjunction_reflect_tail


def reflect_disjunction(query, backbone_llm, gap, header=None, tail=None):
    """One targeted reflection turn; returns the best candidate string or None.

    header/tail default to this module's (possibly language-switched) prompt
    text; the zh query path passes its own Chinese renderings."""
    if header is None:
        header = disjunction_reflect_header
    if tail is None:
        tail = disjunction_reflect_tail
    content = (
        header.format(marker=gap["marker"], k=gap["branch_count"])
        + func_docs
        + tail
        + query["nature_language"]
        + "\nReturn a single JSON object of the form {\"constraints\": [\"<constraint>\", ...]} and nothing else.\nanswer:\n"
    )
    messages = [{"role": "user", "content": content}]
    res = backbone_llm(messages, one_line=False, json_mode="list")
    res = get_first_list_in_str(res)
    try:
        items = json.loads(res)
    except Exception:
        return None
    items = [str(i) for i in items if isinstance(i, str) and i.strip()]
    if not items:
        return None
    return max(items, key=_or_arity)


def enforce_disjunction(query, backbone_llm, max_trails=2, header=None, tail=None):
    """Mechanical verifier: if the NL demands a disjunction the constraints
    lack, inject targeted reflection turns until one OR-combined constraint
    validates, then merge it (dropping lone-branch fragments).

    When a sufficient OR-constraint is already present, no model call is
    made, but lone-branch fragments that duplicate its branches (a collapsed
    branch emitted alongside the disjunction acts as a spurious hard cap)
    are still dropped mechanically."""
    nl = query.get("nature_language", "")
    info = detect_disjunction(nl)
    if not info or info["branch_count"] < 2:
        return query
    need = info["branch_count"] - 1
    cons = list(query.get("hard_logic_py", []))
    satisfying = [c for c in cons if _or_arity(c) >= need]
    if satisfying:
        best = max(satisfying, key=_or_arity)
        merged = merge_disjunction_constraint(
            [c for c in cons if c != best], best
        )
        dropped = [c for c in cons if c not in merged]
        if dropped:
            query["hard_logic_py"] = merged
            query["disjunction_fragment_drop"] = dropped
        return query
    gap = dict(
        info,
        required_or_arity=need,
        found_or_arity=max((_or_arity(c) for c in cons), default=0),
    )
    query["disjunction_gap"] = gap
    for attempt in range(max_trails):
        candidate = reflect_disjunction(query, backbone_llm, gap, header, tail)
        if not candidate:
            continue
        normalized = normalize_generated_constraints(
            [candidate], query.get("people_number"), nl
        )
        if not normalized:
            continue
        candidate = normalized[0]
        if _or_arity(candidate) < gap["required_or_arity"]:
            continue
        probe = {"hard_logic_py": [candidate], "days": query.get("days", -1)}
        if check(probe)[0]:
            continue
        query["hard_logic_py"] = merge_disjunction_constraint(
            query["hard_logic_py"], candidate
        )
        query["disjunction_reflect"] = {"attempt": attempt, "constraint": candidate}
        return query
    query["disjunction_reflect"] = {"failed": True}
    return query


def make_checker(target_city):
    """HardLogicPyChecker with the canonical intra-city-mode idiom whitelisted.

    The oracle idiom for transport-mode preferences filters on
    activity_type(activity)=='transportation' (vacuously true on real plans,
    which contain no such activity type). Without this whitelist the AST
    value-checker flags 'transportation' as invalid and the reflect loop
    burns retries mangling or dropping the canonical pattern.
    """
    checker = HardLogicPyChecker(target_city)
    tracker = checker.trackers.get("activity_type")
    if tracker is not None:
        tracker.valid_values.add("transportation")
    return checker


def load_example_plans(example_plans_dir=os.path.join(
        project_path, "chinatravel", "agent", "nesy_agent", "plan_for_check_en")):
    plan_for_test = {}
    plan_files = os.listdir(example_plans_dir)
    plan_files = [plan_file for plan_file in plan_files if plan_file.endswith(".json")]
    for file in plan_files:
        with open(os.path.join(example_plans_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            plan_for_test[int(file.split("day")[1].split(".")[0])] = data
    return plan_for_test


EXAMPLE_PLANS = load_example_plans()


def get_first_list_in_str(json_str):
    """Extract the constraint list from an LLM reply, robustly.

    Server-dependent trap: a strictly grammar-constrained ``json_object``
    mode (e.g. SGLang) FORCES the reply to be a JSON object, so the array we
    asked for arrives wrapped ({"constraints": [...]}); lenient servers
    (e.g. DashScope) return the bare array. The old first-balanced-bracket
    scan grabbed whatever '[' came first -- on wrapped/verbose replies that
    is often a tiny python list INSIDE a constraint string, silently
    dropping most constraints. Now: parse the whole reply when possible
    (unwrapping dict values), otherwise consider every balanced bracket
    span, and keep the LONGEST candidate that parses to a list of strings.
    """
    s = repair_json(json_str, ensure_ascii=False)
    best = None

    def consider(value):
        nonlocal best
        if (
            isinstance(value, list)
            and value
            and all(isinstance(x, str) for x in value)
        ):
            if best is None or len(value) > len(best):
                best = value

    try:
        whole = json.loads(s)
    except Exception:
        whole = None
    if isinstance(whole, list):
        consider(whole)
    elif isinstance(whole, dict):
        for val in whole.values():
            consider(val)
        # object-of-strings shape: {"1": "result=...", "2": "..."}
        vals = list(whole.values())
        if len(vals) >= 2 and all(isinstance(v, str) for v in vals):
            consider(vals)

    # fallback: every balanced bracket span (bad spans fail json.loads and
    # are skipped; the longest list-of-strings wins)
    depth = 0
    start = None
    for i, c in enumerate(s):
        if c == "[":
            if depth == 0:
                start = i
            depth += 1
        elif c == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        consider(json.loads(s[start : i + 1]))
                    except Exception:
                        pass
                    start = None

    if best is not None:
        return json.dumps(best, ensure_ascii=False)
    return "[]"


def nl2sl_step1(query, backbone_llm):

    nature_language = query["nature_language"]
    messages = [{"role": "user", "content": NL2SL_INSTRUCTION.format(nature_language)}]
    # print(messages[0]["content"])
    query_ = backbone_llm(messages, one_line=False, json_mode=True)

    try:
        l_ptr = query_.find("{")
        r_ptr = query_.rfind("}")
        if l_ptr != -1 and r_ptr != -1:
            query_ = query_[l_ptr : r_ptr + 1]

        query_ = json.loads(query_)
        for key in query_:
            query[key] = query_[key]
    except Exception as e:
        query["hard_logic"] = []
        return query


    return query


def nl2sl_step2(query, backbone_llm):
    try:
        query["hard_logic"] = [str(hl) for hl in query["hard_logic"]]
        hard_logic = "\n".join(query["hard_logic"])
    except Exception as e:
        query["hard_logic"] = []
        query["hard_logic_py"] = []
        return query
    messages = [
        {
            "role": "user",
            "content": sl_trans_prompt
            + hard_logic
            + "\n The query is: \n"
            + query["nature_language"]
            + "\nReturn a single JSON object of the form {\"constraints\": [\"<constraint>\", ...]} and nothing else.\nanswer:\n",
        }
    ]
    # print(messages[0]["content"])
    hard_logic_py = backbone_llm(messages, one_line=False, json_mode="list")
    # l_ptr = hard_logic_py.find("[")
    # r_ptr = hard_logic_py.rfind("]")
    # if l_ptr != -1 and r_ptr != -1:
    #     hard_logic_py = hard_logic_py[l_ptr : r_ptr + 1]
    hard_logic_py = get_first_list_in_str(hard_logic_py)
    # print(hard_logic_py)
    try:
        query["hard_logic_py"] = json.loads(hard_logic_py)
    except Exception as e:
        query["error_hard_logic_py"] = hard_logic_py
        query["hard_logic_py"] = []
    query["hard_logic_py"] = [str(item) for item in query["hard_logic_py"]]
    query["hard_logic_py"] = list(set(query["hard_logic_py"]))
    query["hard_logic_py"] = normalize_generated_constraints(
        query["hard_logic_py"],
        query.get("people_number"),
        query.get("nature_language"),
    )
    return query


def check(query):
    run_error_list = []
    run_error_idx = []
    hard_logic_py = query["hard_logic_py"]

    if query["days"] not in EXAMPLE_PLANS:
        print("Error: days should be in [1, 2, 3, 4, 5, 6]")
        return [], []

    example_plan = EXAMPLE_PLANS[query["days"]]
    for idx, constraint in enumerate(hard_logic_py):
        vars_dict = deepcopy(func_dict)
        vars_dict["plan"] = example_plan
        try:
            # Evaluate the constraint in a safe manner
            exec(
                normalize_hard_logic_constraint(constraint),
                {
                    "__builtins__": {
                        "set": set,
                    }
                },
                vars_dict,
            )
        except Exception as e:
            if str(e) not in [
                "Failed to create Point instance from string: unknown format.",
            ]:
                run_error_list.append(str(e))
                run_error_idx.append(idx)
    return run_error_list, run_error_idx


def reflect_info(query, checker: HardLogicPyChecker):
    hard_logic_py = query["hard_logic_py"]

    run_error_list, run_error_idx = check(query)
    if len(run_error_list):
        return run_error_list, run_error_idx, [], []
    value_error_list = [checker.check(constraint)[0] for constraint in hard_logic_py]
    value_error_idx = [idx for idx, item in enumerate(value_error_list) if len(item)]
    value_error_list = [item for sublist in value_error_list for item in sublist]
    return run_error_list, run_error_idx, value_error_list, value_error_idx


def reflect(query, backbone_llm, run_error_list, value_error_list):

    content = (
        reflect_prompt
        + str(query["hard_logic_py"])
        + "The error is: "
        + "\n".join(run_error_list)
        + "\n".join(value_error_list)
        + "\nThe query is: \n"
        + query["nature_language"]
        + "\nReturn a single JSON object of the form {\"constraints\": [\"<constraint>\", ...]} and nothing else.\nanswer:\n"
    )
    # print(content)
    messages = [{"role": "user", "content": content}]
    res = backbone_llm(messages, one_line=False, json_mode="list")
    # l_ptr = res.find("[")
    # r_ptr = res.rfind("]")
    # if l_ptr != -1 and r_ptr != -1:
    #     res = res[l_ptr : r_ptr + 1]
    res = get_first_list_in_str(res)
    # print(res)
    try:
        query["hard_logic_py"] = json.loads(res)
    except Exception as e:
        query["error_hard_logic_py"] = res
        query["hard_logic_py"] = []
    query["hard_logic_py"] = [str(item) for item in query["hard_logic_py"]]
    query["hard_logic_py"] = normalize_generated_constraints(
        query["hard_logic_py"],
        query.get("people_number"),
        query.get("nature_language"),
    )
    # print(query["hard_logic_py"])
    return query, len(run_error_list + value_error_list) == 0


def nl2sl_step3(query, backbone_llm, checker, max_trails=5):

    cnt = 0
    query["reflect_info"] = []
    query["hard_logic_py_ood"] = []
    value_error_idx = []
    run_error_idx = []
    while cnt < max_trails:
        run_error_list, run_error_idx, value_error_list, value_error_idx = reflect_info(
            query, checker
        )
        query["reflect_info"].append(
            {
                "cnt": cnt,
                "run_error_list": run_error_list,
                "value_error_list": value_error_list,
                "hard_logic_py": query["hard_logic_py"],
            }
        )
        flag = len(run_error_list + value_error_list) == 0
        if flag:
            break
        query, _ = reflect(query, backbone_llm, run_error_list, value_error_list)
        query["hard_logic_py"] = list(set(query["hard_logic_py"]))

        # if "OOD!!!" in query["hard_logic_py"]:
        #     run_error_list, run_error_idx, value_error_list, value_error_idx = (
        #         reflect_info(query, checker)
        #     )
        #     query["ood"] = True
        #     ood_idx = list(set(run_error_idx + value_error_idx))
        #     for idx in ood_idx:
        #         query["hard_logic_py_ood"].append(query["hard_logic_py"][idx])
        #     for ood_logic in query["hard_logic_py_ood"]:
        #         query["hard_logic_py"].remove(ood_logic)
        #     return query

        cnt += 1
    query["reflect_cnt"] = cnt
    run_error_list, run_error_idx, value_error_list, value_error_idx = reflect_info(
        query, checker
    )
    query["reflect_info"].append(
        {
            "cnt": cnt,
            "run_error_list": run_error_list,
            "value_error_list": value_error_list,
            "hard_logic_py": query["hard_logic_py"],
        }
    )
    error_indices = set(run_error_idx + value_error_idx)
    # Best-round salvage (A800 campaign): when the reflect loop cannot
    # converge (e.g. the model keeps rewriting into any/all/sum style that the
    # official {'set'}-only sandbox rejects), dropping every errored index of
    # the FINAL round can collapse the set to near-nothing while an earlier
    # round had far more valid constraints. Pick the round with the most
    # error-free constraints instead, then drop only its errored ones.
    best_round, best_valid = None, -1
    for info in query.get("reflect_info", []):
        cons_r = info.get("hard_logic_py") or []
        errs_r = len(info.get("run_error_list") or []) + len(
            info.get("value_error_list") or [])
        valid_r = len(cons_r) - errs_r
        if valid_r > best_valid:
            best_round, best_valid = info, valid_r
    final_valid = len(query["hard_logic_py"]) - len(error_indices)
    if best_round is not None and best_valid > final_valid:
        cons_r = list(best_round["hard_logic_py"])
        _, err_idx_r, _, verr_idx_r = reflect_info(
            {**query, "hard_logic_py": cons_r}, checker)
        query["hard_logic_py"] = [
            v for i, v in enumerate(cons_r)
            if i not in set(err_idx_r) | set(verr_idx_r)
        ]
        query["reflect_salvaged_round"] = best_round.get("cnt")
    else:
        query["hard_logic_py"] = [
            val
            for idx, val in enumerate(query["hard_logic_py"])
            if idx not in error_indices
        ]
    # mechanical disjunction verifier: runs on the final surviving list so a
    # missing OR-constraint is re-requested even when the reflect loop above
    # converged without errors
    query = enforce_disjunction(query, backbone_llm)
    # round-4 coverage / span-grounding verifier: deterministic NL-triggered
    # fixes for dropped (local-cuisine, airfare->airplane, budget), invented
    # (taxi-car scaling, ungrounded cost caps) and wrong (explicit room
    # count, category-disjunction expansion) constraints
    query = enforce_coverage(query, lang="en")
    # ood_idx = list(set(run_error_idx + value_error_idx))
    # if len(ood_idx):
    #     query["ood"] = True
    #     for idx in ood_idx:
    #         query["hard_logic_py_ood"].append(query["hard_logic_py"][idx])
    #     for ood_logic in query["hard_logic_py_ood"]:
    #         query["hard_logic_py"].remove(ood_logic)
    return query


def nl2sl(query, backbone_llm, checker, cache_dir="cache_hybrid"):
    file_path = os.path.join(
        project_path,
        cache_dir,
        "translation_{}_reflect".format(backbone_llm.name),
        "{}.json".format(query["uid"]),
    )
    if os.path.exists(file_path):
        query = load_json_file(file_path)
        return query
    city_list = [
        "Beijing",
        "Shanghai",
        "Nanjing",
        "Suzhou",
        "Hangzhou",
        "Shenzhen",
        "Chengdu",
        "Wuhan",
        "Guangzhou",
        "Chongqing",
    ]
    if query["target_city"] not in city_list or query["start_city"] not in city_list:
        query["hard_logic"] = []
        query["hard_logic_py"] = []
        query["ood"] = True
        return query
    query = nl2sl_step1(query, backbone_llm)
    query = nl2sl_step2(query, backbone_llm)
    query = nl2sl_step3(query, backbone_llm, checker)

    save_json_file(query, file_path)
    return query


def nl2sl_reflect(query, backbone_llm):
    city_list = [
        "Beijing",
        "Shanghai",
        "Nanjing",
        "Suzhou",
        "Hangzhou",
        "Shenzhen",
        "Chengdu",
        "Wuhan",
        "Guangzhou",
        "Chongqing",
    ]
    if "target_city" in query and "start_city" in query:
        if query["target_city"] not in city_list or query["start_city"] not in city_list:
            query["hard_logic"] = []
            query["hard_logic_py"] = []
            query["ood"] = True
            return query
    query = nl2sl_step1(query, backbone_llm)
    query = nl2sl_step2(query, backbone_llm)
    try:
        checker = make_checker(query["target_city"])
        query = nl2sl_step3(query, backbone_llm, checker)
        query["hard_logic_py_iter_3"] = query["hard_logic_py"]
    except Exception as e:
        query["reflect_error"] = traceback.format_exc()
    return query



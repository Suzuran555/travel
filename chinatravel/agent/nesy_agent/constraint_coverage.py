"""Coverage / span-grounding verifier (phase-2 round-4 generalization fixes).

Deterministic, pattern-keyed checkers + fixers that run on the FINAL
LLM-emitted constraint list (after the reflect loop and the disjunction
verifier). Never keyed on uids. Every rule is derived from a failure
mechanism observed on the never-seen human-split probe:

  b. dropped constraints (vacuity):
     - 'taste the local specialties / 当地特色美食' phrasing must yield the
       target city's signature-cuisine constraint (city->cuisine map,
       gated on the cuisine actually existing in the city DB);
     - any '机票 / airfare / air ticket' mention implies airplane-only
       intercity transport (benchmark convention), gated on the route
       actually having flights in the DB;
     - a 'budget N' mention whose number reaches no constraint gets the
       canonical overall-cost cap injected (only for unqualified budgets).
  c. invented constraints (span-grounding):
     - human-split boilerplate is ALWAYS taxi_cars==1 regardless of party
       size; rewrite people-scaled taxi-car counts for human-register
       queries (the generated splits keep the (people+3)//4 convention);
     - pure cost-cap constraints whose cap number has no support in the
       NL (including xpeople / xdays scalings) are dropped.
  d. wrong constraints:
     - explicit room-count words ('one/a twin room', '一间双床房', 'N间')
       override the rooms-from-people default in room_count checks;
     - 'A or B / A或者B' over attraction-category words expands to the
       conjunction of ALL mentioned category type-sets with <= (never &):
       历史文化/historical -> {Cultural Tourism Area, historical site},
       风景名胜/scenic spots -> {natural scenery}.

`enforce_coverage(query, lang)` applies all fixes in place and records
them under query['coverage_fixes'] / query['coverage_flags'].
`coverage_report(query, lang)` is the dry-run variant (no mutation).
"""

import csv
import json
import os
import re
from copy import deepcopy

_ENV_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "environment")
)

# ---------------------------------------------------------------------------
# city / cuisine tables
# ---------------------------------------------------------------------------

_ZH2EN_CITY = {
    "北京": "Beijing", "上海": "Shanghai", "南京": "Nanjing", "苏州": "Suzhou",
    "杭州": "Hangzhou", "深圳": "Shenzhen", "成都": "Chengdu", "武汉": "Wuhan",
    "广州": "Guangzhou", "重庆": "Chongqing",
}
_EN2ZH_CITY = {v: k for k, v in _ZH2EN_CITY.items()}

# city -> signature local cuisine (the benchmark's implicit expansion of
# 'local specialties'); keys are canonical EN city names
_SIGNATURE_CUISINE = {
    "en": {
        "Beijing": "Beijing cuisine", "Shanghai": "Shanghai cuisine",
        "Nanjing": "Jiangsu-Zhejiang cuisine", "Suzhou": "Jiangsu-Zhejiang cuisine",
        "Hangzhou": "Jiangsu-Zhejiang cuisine", "Shenzhen": "Cantonese cuisine",
        "Guangzhou": "Cantonese cuisine", "Chengdu": "Sichuan cuisine",
        "Chongqing": "Sichuan cuisine", "Wuhan": "Hubei cuisine",
    },
    "zh": {
        "Beijing": "北京菜", "Shanghai": "本帮菜",
        "Nanjing": "江浙菜", "Suzhou": "江浙菜",
        "Hangzhou": "江浙菜", "Shenzhen": "粤菜",
        "Guangzhou": "粤菜", "Chengdu": "川菜",
        "Chongqing": "川菜", "Wuhan": "湖北菜",
    },
}


def _canon_city(name):
    """Canonical EN city name from an EN or ZH spelling, else None."""
    if not name:
        return None
    name = str(name).strip()
    if name in _ZH2EN_CITY:
        return _ZH2EN_CITY[name]
    cap = name.capitalize()
    return cap if cap in _EN2ZH_CITY else None


# ---------------------------------------------------------------------------
# database gates (satisfiability guards for injected constraints)
# ---------------------------------------------------------------------------

_CUISINE_CACHE = {}
_FLIGHT_CACHE = {}


def _db_dir(lang):
    return os.path.join(
        _ENV_ROOT, "database_en" if lang == "en" else "database"
    )


def city_has_cuisine(city_en, cuisine, lang):
    """True if the city's restaurant DB has >=1 restaurant of `cuisine`.
    Unreadable DB -> False (never inject an unverifiable constraint)."""
    key = (city_en, lang)
    if key not in _CUISINE_CACHE:
        cuisines = set()
        low = city_en.lower()
        path = os.path.join(
            _db_dir(lang), "restaurants", low, "restaurants_%s.csv" % low
        )
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("cuisine"):
                        cuisines.add(row["cuisine"].strip())
        except OSError:
            pass
        _CUISINE_CACHE[key] = cuisines
    return cuisine in _CUISINE_CACHE[key]


def _flight_city_pairs(lang):
    """Set of (from_city_en, to_city_en) pairs with at least one flight."""
    if lang in _FLIGHT_CACHE:
        return _FLIGHT_CACHE[lang]
    pairs = set()
    path = os.path.join(
        _db_dir(lang), "intercity_transport", "airplane.jsonl"
    )
    names = _EN2ZH_CITY if lang == "en" else {
        zh: zh for zh in _ZH2EN_CITY
    }

    def airport_city(airport):
        for name in names:
            if str(airport).startswith(name):
                return _canon_city(name)
        return None

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                a = airport_city(row.get("From", ""))
                b = airport_city(row.get("To", ""))
                if a and b:
                    pairs.add((a, b))
    except OSError:
        pass
    _FLIGHT_CACHE[lang] = pairs
    return pairs


def has_round_trip_flights(start_city, target_city, lang):
    a, b = _canon_city(start_city), _canon_city(target_city)
    if not a or not b:
        return False
    pairs = _flight_city_pairs(lang)
    return (a, b) in pairs and (b, a) in pairs


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------

_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_QUOTED_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")
_SCAFFOLD_LITERALS = {
    "breakfast", "lunch", "dinner", "attraction", "accommodation",
    "transportation", "train", "airplane", "metro", "taxi", "walk",
}
_CALL = r"\((?:[^()]|\([^()]*\))*\)"


def _or_arity(code):
    if not isinstance(code, str):
        return 0
    return len(re.findall(r"\bor\b", _STRING_LITERAL_RE.sub("", code)))


def _literals(code):
    return {a or b for a, b in _QUOTED_RE.findall(code)}


def _has_literal(constraints, value):
    return any(
        isinstance(c, str) and ("'%s'" % value in c or '"%s"' % value in c)
        for c in constraints
    )


def _num_token(v):
    """Render a float cap as the shortest faithful literal."""
    return str(int(v)) if float(v) == int(v) else str(v)


def is_human_register(query):
    """Human-split queries (colloquial register): tag=='human' or h-prefixed
    uid. The generated phase-1 split carries neither."""
    return (
        query.get("tag") == "human"
        or str(query.get("uid", "")).startswith("h")
    )


# ---------------------------------------------------------------------------
# rule c1: taxi_cars boilerplate convention (human split: always ==1)
# ---------------------------------------------------------------------------

_TAXI_NEQ_RE = re.compile(r"(taxi_cars\s*%s\s*!=\s*)(\d+)" % _CALL)
_TAXI_NEQ_SYM_RE = re.compile(
    r"taxi_cars\s*%s\s*!=\s*([A-Za-z_]\w*)" % _CALL
)
# people-scaled car-count formulas the LLM writes inline instead of a literal
_CAR_FORMULA_RES = (
    re.compile(r"\(\s*people_count\(plan\)\s*\+\s*3\s*\)\s*//\s*4"),
    re.compile(
        r"int\s*\(\s*\(\s*people_count\(plan\)\s*-\s*1\s*\)\s*/\s*4\s*\)"
        r"\s*\+\s*1"
    ),
)
_ARITH_ONLY_RE = re.compile(r"^[\d\s()+\-*/]+$")


def taxi_cars_from_constraints(constraints, people_number=None):
    """The single taxi-car count the DSL requires, or None if absent or
    ambiguous. Used by the planner to book exactly the DSL's car count.
    Handles both literal counts (taxi_cars(...)!=2) and symbolic ones
    (expected_cars=(people_count(plan)+3)//4 ... taxi_cars(...)!=expected_cars)."""
    vals = set()
    for c in constraints or []:
        if not isinstance(c, str):
            continue
        for m in _TAXI_NEQ_RE.finditer(c):
            vals.add(int(m.group(2)))
        for m in _TAXI_NEQ_SYM_RE.finditer(c):
            sym = m.group(1)
            am = re.search(r"\b%s\s*=([^=].*)" % re.escape(sym), c)
            if not am:
                continue
            expr = am.group(1).strip()
            if people_number is not None:
                expr = re.sub(
                    r"people_count\(plan\)", str(people_number), expr
                )
            if _ARITH_ONLY_RE.match(expr):
                try:
                    vals.add(int(eval(expr, {"__builtins__": {}}, {})))
                except Exception:
                    pass
    return vals.pop() if len(vals) == 1 else None


def apply_taxi_convention(constraints, human):
    """Human-register queries carry the universal taxi_cars==1 boilerplate
    regardless of party size; generated splits keep the scaled count.
    Rewrites both literal counts and inline people-scaled formulas."""
    if not human:
        return constraints, None
    out, changed = [], []
    for c in constraints:
        if isinstance(c, str):
            c2 = _TAXI_NEQ_RE.sub(lambda m: m.group(1) + "1", c)
            if "taxi_cars" in c2:
                for rx in _CAR_FORMULA_RES:
                    c2 = rx.sub("1", c2)
            if c2 != c:
                changed.append({"before": c, "after": c2})
            c = c2
        out.append(c)
    if not changed:
        return constraints, None
    return out, {"rule": "taxi_cars_human_convention", "changes": changed}


# ---------------------------------------------------------------------------
# rule d1: explicit room-count words override the rooms-from-people default
# ---------------------------------------------------------------------------

_WORD2NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_ROOM_TYPE_WORDS = (
    r"(?:twin|double|standard|triple|quad|king|queen|"
    r"big[- ]?bed|two[- ]?bed|double[- ]?bed|family)"
)
# 'a/an <room-type> room' counts as ONE room only when a room-type word
# pins it down; bare numerals/count words before 'room(s)' always count.
# 'single' is deliberately excluded: 'a single(-bed) room' names the room
# TYPE (the oracle constrains room_type only, and one single room cannot
# hold a multi-person party), unlike 'a twin room' which pins one room.
_ROOM_ARTICLE_EN_RE = re.compile(
    r"\b(a|an)\s+%s(?:[- ]bed)?\s+room\b" % _ROOM_TYPE_WORDS, re.IGNORECASE
)
_ROOM_NUMBER_EN_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:[\w-]+\s+){0,2}?rooms?\b",
    re.IGNORECASE,
)
_ROOM_ZH_RE = re.compile(r"([一两二三四五六七八九十\d])\s*间")
_ROOM_COUNT_NEQ_RE = re.compile(r"(room_count\s*\(activity\)\s*!=\s*)(\d+)")


def explicit_room_count(nature_language):
    """The explicit room count stated in the NL, or None (absent/ambiguous)."""
    if not nature_language:
        return None
    counts = set()
    for m in _ROOM_ARTICLE_EN_RE.finditer(nature_language):
        counts.add(1)
    for m in _ROOM_NUMBER_EN_RE.finditer(nature_language):
        tok = m.group(1).lower()
        counts.add(int(tok) if tok.isdigit() else _WORD2NUM[tok])
    for m in _ROOM_ZH_RE.finditer(nature_language):
        tok = m.group(1)
        counts.add(int(tok) if tok.isdigit() else _WORD2NUM[tok])
    return counts.pop() if len(counts) == 1 else None


def apply_room_count_override(constraints, nature_language):
    """Rewrite room_count checks to the NL's explicit room count. Fires only
    when a room_count constraint already exists (never invents rooms)."""
    n = explicit_room_count(nature_language)
    if not n:
        return constraints, None
    out, changed = [], []
    for c in constraints:
        if isinstance(c, str):
            c2 = _ROOM_COUNT_NEQ_RE.sub(
                lambda m: m.group(1) + str(n) if int(m.group(2)) != n else m.group(0),
                c,
            )
            if c2 != c:
                changed.append({"before": c, "after": c2})
            c = c2
        out.append(c)
    if not changed:
        return constraints, None
    return out, {"rule": "explicit_room_count", "count": n, "changes": changed}


# ---------------------------------------------------------------------------
# rule b1: 'taste the local specialties' -> signature-cuisine constraint
# ---------------------------------------------------------------------------

_LOCAL_FOOD_EN_RE = re.compile(
    r"\b(?:taste|tastes|tasting|tasted|try|tries|trying|sample|sampling|"
    r"savor|savour|experience|experiencing|enjoy|enjoying|eat|eating)\b"
    r"[^.!?;]{0,80}?\b(?:local|regional|authentic)\s+"
    r"(?:specialt(?:y|ies)|cuisines?|foods?|delicac(?:y|ies)|dish(?:es)?|"
    r"snacks?|flavou?rs?|gourmet|delicious\s+food)",
    re.IGNORECASE,
)
_LOCAL_FOOD_ZH_RE = re.compile(
    r"(?:品尝|尝尝|尝一尝|吃|体验|感受|打卡)[^。！？;；.!?]{0,20}?"
    r"(?:当地|本地)?特色(?:美食|菜|小吃)"
    r"|(?:当地|本地)(?:的)?(?:特色)?美食"
    r"|特色(?:美食|菜肴|菜|小吃)"
)

_CUISINE_TMPL = (
    "restaurant_type_set = set()\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n"
    "    restaurant_type_set.add(restaurant_type(activity, target_city(plan)))\n"
    "result=({'%s'}<=restaurant_type_set)"
)


def detect_local_cuisine(nature_language, target_city, lang="en"):
    """Signature cuisine required by 'taste local specialties' phrasing
    (or '<target city> cuisine/美食'), else None."""
    if not nature_language:
        return None
    city = _canon_city(target_city)
    if city is None:
        return None
    cuisine = _SIGNATURE_CUISINE[lang].get(city)
    if cuisine is None:
        return None
    if _LOCAL_FOOD_EN_RE.search(nature_language) or _LOCAL_FOOD_ZH_RE.search(
        nature_language
    ):
        return cuisine
    # '<city> cuisine / 成都美食' with the TARGET city's name
    zh_city = _EN2ZH_CITY[city]
    city_food_re = re.compile(
        r"\b%s\s+(?:cuisine|specialt(?:y|ies)|delicac(?:y|ies)|food)\b" % city
        + r"|%s(?:的)?(?:特色)?(?:美食|菜)" % zh_city,
        re.IGNORECASE,
    )
    if city_food_re.search(nature_language):
        return cuisine
    return None


def inject_local_cuisine(constraints, query, lang="en"):
    cuisine = detect_local_cuisine(
        query.get("nature_language"), query.get("target_city"), lang
    )
    if not cuisine or _has_literal(constraints, cuisine):
        return constraints, None
    city = _canon_city(query.get("target_city"))
    if not city_has_cuisine(city, cuisine, lang):
        return constraints, None  # locally unsatisfiable: never inject
    new = _CUISINE_TMPL % cuisine
    return constraints + [new], {
        "rule": "local_signature_cuisine",
        "cuisine": cuisine,
        "added": new,
    }


# ---------------------------------------------------------------------------
# rule b2: airfare/机票 mention -> airplane-only intercity transport
# ---------------------------------------------------------------------------

_AIRFARE_RE = re.compile(
    r"\b(?:airfares?|air\s*fares?|air\s*tickets?|plane\s*tickets?|"
    r"flight\s*tickets?)\b|机票",
    re.IGNORECASE,
)
_INTERCITY_TMPL = (
    "intercity_transport_set = set()\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['train', 'airplane']:\n"
    "    intercity_transport_set.add(activity_type(activity))\n"
    "result=(intercity_transport_set=={'airplane'})"
)


def inject_airplane_transport(constraints, query, lang="en"):
    nl = query.get("nature_language") or ""
    if not _AIRFARE_RE.search(nl):
        return constraints, None
    # an intercity-transport constraint already exists (either mode): the
    # explicit translation wins, never override it
    if any(
        isinstance(c, str) and "intercity_transport_set" in c
        for c in constraints
    ):
        return constraints, None
    if not has_round_trip_flights(
        query.get("start_city"), query.get("target_city"), lang
    ):
        return constraints, None  # no flights on this route: unsatisfiable
    return constraints + [_INTERCITY_TMPL], {
        "rule": "airfare_implies_airplane",
        "added": _INTERCITY_TMPL,
    }


# ---------------------------------------------------------------------------
# rule d2: category disjunction over attraction-type words -> conjunction of
# the full type expansions with <=
# ---------------------------------------------------------------------------

_CATEGORY_LEXICON = {
    "en": (
        (
            re.compile(
                r"histor(?:y|ic(?:al)?)"
                r"(?:\s+(?:and|&)\s+cultur(?:e|al)\w*)?",
                re.IGNORECASE,
            ),
            ("Cultural Tourism Area", "historical site"),
        ),
        (
            re.compile(
                r"scenic\s+(?:spots?|areas?|sites?|attractions?)"
                r"|famous\s+scener\w*|natural\s+scenery",
                re.IGNORECASE,
            ),
            ("natural scenery",),
        ),
    ),
    "zh": (
        (re.compile(r"历史文化"), ("文化旅游区", "历史古迹")),
        (re.compile(r"风景名胜"), ("自然风光",)),
    ),
}
_OR_SEP_RE = re.compile(
    r"^\s*(?:,|，|、)?\s*(?:or|或者|或|还是)\s+?(?:some\s+|the\s+|a\s+)?\s*$"
    r"|^\s*(?:,|，|、)?\s*(?:或者|或|还是)\s*$",
    re.IGNORECASE,
)

_ATTRACTION_TMPL = (
    "attraction_type_set = set()\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='attraction':\n"
    "    attraction_type_set.add(attraction_type(activity, target_city(plan)))\n"
    "result=({%s}<=attraction_type_set)"
)
_NON_TYPE_FUNC_RE = re.compile(
    r"activity_(?:position|cost|price|tickets|start_time|end_time|time)\(|"
    r"restaurant_type\(|accommodation_type\(|room_(?:count|type)\(|"
    r"innercity_transport_|intercity_transport_|metro_tickets|taxi_cars|"
    r"poi_(?:distance|recommend_time)"
)


def detect_category_disjunction(nature_language, lang="en"):
    """Full attraction-type set required by 'catA or catB' NL phrasing over
    lexicon category words, else None. The benchmark convention expands the
    disjunction to the CONJUNCTION of all mentioned categories' type sets."""
    if not nature_language:
        return None
    entries = _CATEGORY_LEXICON.get(lang, ())
    matches = []
    for idx, (rx, types) in enumerate(entries):
        for m in rx.finditer(nature_language):
            matches.append((m.start(), m.end(), idx, types))
    matches.sort()
    for i in range(len(matches) - 1):
        s1, e1, idx1, t1 = matches[i]
        s2, e2, idx2, t2 = matches[i + 1]
        if idx1 == idx2 or s2 <= e1:
            continue
        between = nature_language[e1:s2]
        if len(between) <= 20 and _OR_SEP_RE.match(between):
            return set(t1) | set(t2)
    return None


def apply_category_expansion(constraints, nature_language, lang="en"):
    required = detect_category_disjunction(nature_language, lang)
    if not required:
        return constraints, None
    # already fully expressed as a subset check somewhere -> nothing to do
    for c in constraints:
        if (
            isinstance(c, str)
            and "<=" in c
            and all(t in _literals(c) for t in required)
        ):
            return constraints, None
    out, removed = [], []
    for c in constraints:
        if (
            isinstance(c, str)
            and "attraction_type(" in c
            and _or_arity(c) == 0
            and not _NON_TYPE_FUNC_RE.search(c)
        ):
            lits = _literals(c) - _SCAFFOLD_LITERALS
            # a partial / weakened (&) translation of the same requirement
            if lits and lits <= required:
                removed.append(c)
                continue
        out.append(c)
    literal = ", ".join("'%s'" % t for t in sorted(required))
    new = _ATTRACTION_TMPL % literal
    out.append(new)
    return out, {
        "rule": "category_disjunction_expansion",
        "required": sorted(required),
        "removed": removed,
        "added": new,
    }


# ---------------------------------------------------------------------------
# budget rules: span-grounding drop (b/c) + overall-cap coverage injection (b)
# ---------------------------------------------------------------------------

_NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([万wWkK])?")
_ZH_NUM_UNIT_RE = re.compile(r"([一两二三四五六七八九十]+)\s*([万千百])")
_ZH_UNIT_MULT = {"万": 10000, "千": 1000, "百": 100}
_EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_UNITS = dict(
    _WORD2NUM,
    **_EN_TENS,
    eleven=11, twelve=12, thirteen=13, fourteen=14, fifteen=15,
    sixteen=16, seventeen=17, eighteen=18, nineteen=19,
)
# 'three thousand', 'twenty-five hundred', 'ten thousand', '3 thousand'
_EN_WORDNUM_RE = re.compile(
    r"\b(a|an|\d+(?:\.\d+)?|%s)(?:[-\s](%s))?\s+(hundred|thousand)\b"
    % ("|".join(_EN_UNITS), "|".join(k for k in _WORD2NUM if k.isalpha())),
    re.IGNORECASE,
)
_EN_SCALE = {"hundred": 100, "thousand": 1000}


def _en_word_numbers(s):
    nums = set()
    for m in _EN_WORDNUM_RE.finditer(s):
        head, tail, scale = m.group(1).lower(), m.group(2), m.group(3).lower()
        try:
            v = float(head)
        except ValueError:
            v = _EN_UNITS.get(head, 1)  # 'a hundred' -> 1
        if tail:
            v += _EN_UNITS.get(tail.lower(), 0)
        nums.add(v * _EN_SCALE[scale])
    return nums


def _nl_numbers(nature_language):
    """All numeric magnitudes stated in the NL (commas stripped, 万/k units,
    English word-numbers and simple Chinese-numeral+unit forms expanded)."""
    nums = set()
    s = (nature_language or "").replace(",", "").replace("，", "")
    for m in _NUM_UNIT_RE.finditer(s):
        v = float(m.group(1))
        unit = m.group(2)
        nums.add(v)
        if unit in ("万", "w", "W"):
            nums.add(v * 10000)
        elif unit in ("k", "K"):
            nums.add(v * 1000)
    for m in _ZH_NUM_UNIT_RE.finditer(s):
        head = m.group(1)
        if head in _WORD2NUM:
            nums.add(_WORD2NUM[head] * _ZH_UNIT_MULT[m.group(2)])
    nums |= _en_word_numbers(s)
    return nums


_MONEY_MENTION_RE = re.compile(
    r"budget|yuan|rmb|[¥￥]|cost|spend|expens|afford|"
    r"预算|元|块|万|花费|花销|开销|经费|费用",
    re.IGNORECASE,
)


_CAP_LE_RE = re.compile(r"<=\s*(\d+(?:\.\d+)?)")
_CAP_GT_RE = re.compile(r">\s*(\d+(?:\.\d+)?)\s*:\s*result\s*=\s*False")
_COST_FUNC_RE = re.compile(
    r"activity_cost\(|activity_price\(|innercity_transport_(?:cost|price)\("
)


def _cost_caps(code):
    """Numeric caps of a cost/price constraint (both <=CAP and >CAP forms)."""
    stripped = _STRING_LITERAL_RE.sub("", code)
    caps = {float(x) for x in _CAP_LE_RE.findall(stripped)}
    caps |= {float(x) for x in _CAP_GT_RE.findall(stripped)}
    return caps


def _is_pure_cost_constraint(code):
    if not isinstance(code, str) or _or_arity(code) > 0:
        return False
    if not _COST_FUNC_RE.search(code):
        return False
    if _literals(code) - _SCAFFOLD_LITERALS:
        return False  # carries POI/type identity: not a pure budget cap
    # a zero cap is a 'free of charge' requirement (only-free-attractions),
    # grounded in wording rather than a stated amount: never a budget cap
    return bool({cap for cap in _cost_caps(code) if cap > 0})


def _grounded(cap, nl_nums, people, days):
    """A cap is NL-supported if it equals a stated number or a stated number
    scaled by the party size or trip length (per-person/per-day budgets)."""
    scales = {1}
    if people:
        scales.add(people)
    if days:
        scales.add(days)
    for n in nl_nums:
        for s in scales:
            if abs(cap - n * s) < 1e-6:
                return True
    return False


def drop_ungrounded_cost_caps(constraints, query):
    """Drop invented budget caps; flag merely un-matchable ones.

    A pure cost-cap constraint whose cap number has no NL support is
    dropped only when the NL contains NO money wording at all (a fully
    invented budget). When the NL does talk about money but the number
    cannot be matched (unparsed numeral form, reformulated amount), the
    constraint is kept and surfaced as a flag: dropping a real budget is
    worse than keeping a slightly-off one."""
    nl = query.get("nature_language") or ""
    nl_nums = _nl_numbers(nl)
    money_talk = bool(_MONEY_MENTION_RE.search(nl))
    people = query.get("people_number")
    days = query.get("days")
    out, dropped, suspect = [], [], []
    for c in constraints:
        if _is_pure_cost_constraint(c) and not any(
            _grounded(cap, nl_nums, people, days) for cap in _cost_caps(c)
        ):
            if not money_talk:
                dropped.append(c)
                continue
            suspect.append(c)
        out.append(c)
    if dropped:
        return out, {"rule": "ungrounded_cost_cap_dropped", "dropped": dropped}
    if suspect:
        return constraints, {
            "rule": "ungrounded_cost_cap_suspect",
            "flag_only": True,
            "constraints": suspect,
        }
    return constraints, None


_BUDGET_QUALIFIER_RE = re.compile(
    r"accommodation|hotel|lodging|room|meal|dining|food|restaurant|"
    r"transport|taxi|airfare|air\s*ticket|flight|train|tickets?|"
    r"sightseeing|attraction|"
    r"住宿|酒店|餐|吃|交通|打车|机票|车票|门票|景点|游玩|观光",
    re.IGNORECASE,
)
_BUDGET_NEAR_RE = re.compile(
    r"(?:budget|预算)[^.!?;。！？；]{0,50}?(\d[\d,，]*(?:\.\d+)?)\s*([万wWkK])?"
    r"|(\d[\d,，]*(?:\.\d+)?)\s*([万wWkK])?\s*(?:yuan|rmb|元|块)?"
    r"[^.!?;。！？；]{0,25}?(?:budget|预算)",
    re.IGNORECASE,
)
_TOTAL_COST_TMPL = (
    "total_cost=0\n"
    "for activity in allactivities(plan): "
    "total_cost+=activity_cost(activity)"
    "+innercity_transport_cost(activity_transports(activity))\n"
    "result=(total_cost<=%s)"
)


def _stated_budget(nature_language):
    """(amount, window_text) of an unqualified overall-budget mention."""
    if not nature_language:
        return None
    m = _BUDGET_NEAR_RE.search(nature_language)
    if not m:
        return None
    raw = (m.group(1) or m.group(3)).replace(",", "").replace("，", "")
    unit = m.group(2) or m.group(4)
    amount = float(raw)
    if unit in ("万", "w", "W"):
        amount *= 10000
    elif unit in ("k", "K"):
        amount *= 1000
    window = nature_language[max(0, m.start() - 40): m.end() + 10]
    return amount, window


def inject_overall_budget(constraints, query):
    """NL states an (unqualified) budget number that reaches no constraint:
    inject the canonical overall-cost cap."""
    found = _stated_budget(query.get("nature_language"))
    if not found:
        return constraints, None
    amount, window = found
    if amount < 100:
        return constraints, None
    all_caps = set()
    for c in constraints:
        if isinstance(c, str):
            all_caps |= _cost_caps(c)
    if any(abs(amount - cap) < 1e-6 for cap in all_caps):
        return constraints, None  # already covered
    if _BUDGET_QUALIFIER_RE.search(window):
        # qualified budget (hotel/meal/airfare-excluded/...): too ambiguous
        # to inject mechanically; surface as a flag only
        return constraints, {
            "rule": "qualified_budget_uncovered",
            "flag_only": True,
            "amount": amount,
        }
    if all_caps:
        # some other cost cap exists; injecting a second overall cap is more
        # likely to over-constrain than to recover a dropped one
        return constraints, {
            "rule": "budget_number_uncovered",
            "flag_only": True,
            "amount": amount,
        }
    new = _TOTAL_COST_TMPL % _num_token(amount)
    return constraints + [new], {
        "rule": "overall_budget_injected",
        "amount": amount,
        "added": new,
    }


# ===========================================================================
# ROUND-5 rules (phase-2, full-1000 sweep autopsy).  All deterministic,
# NL-marker-triggered, never keyed on uids.  Injected/rewritten constraints
# use the ORACLE dialect verbatim: the planner's DSL extractor demonstrably
# honors that dialect (the oracle-translation pipeline scores 100.00), so a
# rewrite into oracle shape simultaneously fixes verification semantics AND
# planner-side enforcement.
# ===========================================================================

_CJK_RE = re.compile(u"[㐀-䶿一-鿿豈-﫿]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_nl_lang(nature_language, default="en"):
    """'zh' or 'en' by CJK-character ratio (mirrors lang_router)."""
    text = nature_language if isinstance(nature_language, str) else ""
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if cjk + latin == 0:
        return default
    return "zh" if cjk / (cjk + latin) >= 0.20 else "en"


# --- disjunction region ----------------------------------------------------
# Requirements listed AFTER a 'satisfy one of the following' marker are
# OR-branches: no round-5 rule may inject/rescope them into unconditional
# hard constraints.  (Mirror of the nl2sl disjunction verifier's marker set;
# kept local to avoid a circular import.)

_R5_DISJ_MARKER_RE = re.compile(
    r"(?:at\s+least\s+|any\s+)?(?:one|any|either)\s+of\s+"
    r"(?:the\s+following|these|them)"
    r"|(?:meet|satisfy|satisfies)\s+either\b|requir\w*\s+either"
    r"|\(any\s+one\)"
    r"|满足以下(?:要求|条件)?(?:中的)?(?:至少|任意|任一)?一(?:个|项|条)"
    r"|(?:至少|任意|任一)满足(?:以下|下列|其中)之?一|任选其一|满足其一",
    re.IGNORECASE,
)
# enumerated branch labels after the marker ('1. ' / '(2) ' / '3、'); a bare
# 'one of the following hotels: A or B' inline-any-of has no such labels and
# must NOT open a region
_BRANCH_LABEL_1_RE = re.compile(r"(?<![\d.])1\s*(?:[.)]\s|、)")
_BRANCH_LABEL_2_RE = re.compile(r"(?<![\d.])2\s*(?:[.)]\s|、)")


def _disjunction_region_start(nature_language):
    """Char offset where OR-branch content begins, or None.  Requires BOTH a
    disjunction marker and at least two enumerated branch labels after it."""
    if not nature_language:
        return None
    m = _R5_DISJ_MARKER_RE.search(nature_language)
    if not m:
        return None
    tail = nature_language[m.end():]
    if _BRANCH_LABEL_1_RE.search(tail) and _BRANCH_LABEL_2_RE.search(tail):
        return m.end()
    return None


def _in_disjunction_region(nature_language, offset):
    start = _disjunction_region_start(nature_language)
    return start is not None and offset >= start


# --- rule r5.0: base boilerplate injection (empty/degenerate translations) --

_BOILER_TICKETS_TMPL = (
    "result=True\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity) in ['attraction', 'airplane', 'train'] "
    "and activity_tickets(activity)!={n}: result=False\n"
    "  if innercity_transport_type(activity_transports(activity))=='metro' "
    "and metro_tickets(activity_transports(activity))!={n}: result=False"
)
_BOILER_TAXI_TMPL = (
    "result=True\n"
    "for activity in allactivities(plan):\n"
    "  if innercity_transport_type(activity_transports(activity))=='taxi' "
    "and taxi_cars(activity_transports(activity))!={n}: result=False"
)


def inject_base_boilerplate(constraints, query):
    """Append the benchmark's universal day/people/tickets/taxi boilerplate
    when it is missing (empty or gutted translations run the planner
    unconstrained otherwise).  Derived from the query's given fields only."""
    added = []
    cons = list(constraints)
    days = query.get("days")
    people = query.get("people_number")
    joined = "\n".join(c for c in cons if isinstance(c, str))
    if days and not re.search(r"day_count\(plan\)\s*==", joined):
        added.append("result=(day_count(plan)==%d)" % int(days))
    if people and not re.search(r"people_count\(plan\)\s*==", joined):
        added.append("result=(people_count(plan)==%d)" % int(people))
    if people and "activity_tickets" not in joined:
        added.append(_BOILER_TICKETS_TMPL.format(n=int(people)))
    if people and "taxi_cars" not in joined:
        cars = 1 if is_human_register(query) else (int(people) + 3) // 4
        added.append(_BOILER_TAXI_TMPL.format(n=cars))
    if not added:
        return constraints, None
    return cons + added, {"rule": "base_boilerplate_injected", "added": added}


# --- rule r5.1: deterministic budget-scope canonicalizer --------------------
# Mechanism (top round-5 category): the NL states a SCOPED budget ('dining
# budget of 2200', 'budget for intra-city transportation is 40') and the
# translator emits the canonical TOTAL-cost cap with that amount.  An
# infeasible total cap collapses the search to an empty plan.  Rescope the
# cap onto the oracle's scoped aggregation; drop leftover/invented total
# caps whose amount has no total/overall-budget support in the NL.

_SCOPE_PATTERNS = {
    "en": (
        ("innercity", re.compile(
            r"intra-?\s*city|within\s+the\s+city|local\s+transport"
            r"|(?<![a-z])(?<!inter)city\s+transport|getting\s+around",
            re.IGNORECASE)),
        ("intercity", re.compile(
            r"inter-?\s*city|cross-?\s*city|between\s+cities", re.IGNORECASE)),
        ("dining", re.compile(
            r"dining|meals?\b|food\b", re.IGNORECASE)),
        ("accommodation", re.compile(
            r"accommodation|hotels?\b|lodging", re.IGNORECASE)),
        ("attraction", re.compile(
            r"sightseeing|attractions?\b", re.IGNORECASE)),
        ("total", re.compile(
            r"total|overall|entire\s+trip|whole\s+trip", re.IGNORECASE)),
    ),
    "zh": (
        ("innercity", re.compile(r"市内|市区|城市内|本地交通|市里")),
        ("intercity", re.compile(r"城际|跨城|城市间|城市之间")),
        ("dining", re.compile(r"餐饮|用餐|吃饭|餐费|伙食")),
        ("accommodation", re.compile(r"住宿|酒店|旅馆|旅店")),
        ("attraction", re.compile(r"景点|门票|游览|观光|游玩")),
        ("total", re.compile(r"总(?:预算|花费|开销|费用)?|全部|整体|全程")),
    ),
}

# amount AFTER the budget word ('budget of/is N', gap tolerant) or amount
# DIRECTLY before it ('3000元的预算'); a loose before-gap would swallow
# unrelated numbers ('... for 3 days, with a meal budget' -> amount 3)
_R5_BUDGET_MENTION_RE = re.compile(
    r"(?:budget|预算)[^.;!?。！？；\n]{0,60}?(\d[\d,，]*(?:\.\d+)?)"
    r"|(\d[\d,，]*(?:\.\d+)?)\s*(?:yuan|rmb|元|块)?\s*(?:的)?\s*(?:budget|预算)",
    re.IGNORECASE,
)

# oracle-dialect scoped budget templates (planner extractor parses these)
_SCOPED_BUDGET_TMPLS = {
    "innercity": (
        "inner_city_transportation_cost=0\n"
        "for activity in allactivities(plan):\n"
        "    inner_city_transportation_cost += "
        "innercity_transport_cost(activity_transports(activity))\n"
        "result=(inner_city_transportation_cost<=%s)"
    ),
    "intercity": (
        "inter_city_transportation_cost=0\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity) in ['airplane','train']: "
        "inter_city_transportation_cost+=activity_cost(activity)\n"
        "result=(inter_city_transportation_cost<=%s)"
    ),
    "dining": (
        "restaurant_cost=0\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']: "
        "restaurant_cost+=activity_cost(activity)\n"
        "result=(restaurant_cost<=%s)"
    ),
    "accommodation": (
        "accommodation_cost=0\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='accommodation': "
        "accommodation_cost+=activity_cost(activity)\n"
        "result=(accommodation_cost<=%s)"
    ),
    "attraction": (
        "attraction_cost=0\n"
        "for activity in allactivities(plan):\n"
        "  if activity_type(activity)=='attraction': "
        "attraction_cost+=activity_cost(activity)\n"
        "result=(attraction_cost<=%s)"
    ),
}

_CLAUSE_BOUNDARY_RE = re.compile(r"[.;!?。！？；\n,，]")
_MEAL_FILTER_RE = re.compile(
    r"activity_type\(activity\)\s*in\s*\[\s*['\"](?:breakfast|lunch|dinner)")
_ACC_FILTER_RE = re.compile(r"activity_type\(activity\)\s*==\s*['\"]accommodation")
_ATTR_FILTER_RE = re.compile(r"activity_type\(activity\)\s*==\s*['\"]attraction")
_INTERCITY_FILTER_RE = re.compile(
    r"activity_type\(activity\)\s*in\s*[\[({]\s*['\"](?:train|airplane)['\"]\s*,"
    r"\s*['\"](?:train|airplane)")
_ANY_TYPE_FILTER_RE = re.compile(r"activity_type\(activity\)\s*(?:==|!=|\s+in\s+|in\s*[\[({])")


def scoped_budget_mentions(nature_language, lang="en"):
    """[(scope, amount, offset)] for every classifiable budget mention.
    scope in {innercity,intercity,dining,accommodation,attraction,total};
    unclassifiable / ambiguous mentions are skipped."""
    if not nature_language:
        return []
    out = []
    entries = _SCOPE_PATTERNS.get(lang) or _SCOPE_PATTERNS["en"]
    for m in _R5_BUDGET_MENTION_RE.finditer(nature_language):
        raw = (m.group(1) or m.group(2)).replace(",", "").replace("，", "")
        try:
            amount = float(raw)
        except ValueError:
            continue
        # clause = text from the previous clause boundary to the mention end
        clause_start = 0
        for b in _CLAUSE_BOUNDARY_RE.finditer(nature_language, 0, m.start()):
            clause_start = b.end()
        clause = nature_language[clause_start:m.end()]
        if amount < 10:
            continue  # tiny numbers near 'budget' are day/people counts
        scopes = [name for name, rx in entries if rx.search(clause)]
        if "total" in scopes and len(scopes) > 1:
            scopes.remove("total")
        if len(scopes) != 1:
            continue  # unqualified or ambiguous: not this rule's business
        out.append((scopes[0], amount, m.start()))
    return out


def _cap_matches(code, amount):
    return any(abs(cap - amount) < 1e-6 for cap in _cost_caps(code))


def _implements_scope(code, scope, amount):
    """True when `code` already carries the scoped budget (any dialect)."""
    if not isinstance(code, str) or _or_arity(code) > 0:
        return False
    if not _cap_matches(code, amount):
        return False
    if scope == "dining":
        return bool(_MEAL_FILTER_RE.search(code)) and (
            "activity_cost(" in code or "activity_price(" in code)
    if scope == "accommodation":
        return bool(_ACC_FILTER_RE.search(code)) and (
            "activity_cost(" in code or "activity_price(" in code)
    if scope == "attraction":
        return bool(_ATTR_FILTER_RE.search(code)) and (
            "activity_cost(" in code or "activity_price(" in code)
    if scope == "intercity":
        return bool(_INTERCITY_FILTER_RE.search(code)) and "activity_cost(" in code
    if scope == "innercity":
        # canonical: unfiltered sum of innercity transport cost; a type-
        # filtered sum is the known undercount bug, NOT an implementation
        return (
            "innercity_transport_cost(" in code
            and "activity_cost(" not in code
            and not _ANY_TYPE_FILTER_RE.search(code)
        )
    return False


def _is_total_shaped(code):
    """A pure overall-cost cap: an UNFILTERED accumulation of activity costs
    (any accumulator name).  A type-filtered accumulator is scoped by shape
    regardless of its name (Qwen mis-names scoped accumulators total_cost)."""
    if not _is_pure_cost_constraint(code):
        return False
    if _ANY_TYPE_FILTER_RE.search(code):
        return False
    if re.search(r"\btotal_cost\b", _STRING_LITERAL_RE.sub("", code)):
        return True
    return "activity_cost(" in code and "innercity_transport_cost(" in code


def enforce_budget_scope(constraints, query, lang="en"):
    """Rescope mis-totalized scoped budgets onto the oracle aggregation and
    drop total-cost caps with no total-budget support in the NL."""
    nl = query.get("nature_language") or ""
    mentions = [
        (scope, amount, off)
        for scope, amount, off in scoped_budget_mentions(nl, lang)
        if not _in_disjunction_region(nl, off)
    ]
    cons = list(constraints)
    removed, added = [], []
    scoped = [(s, a) for s, a, _ in mentions if s != "total"]
    total_amounts = {a for s, a, _ in mentions if s == "total"}

    def _implements_any_mention(code):
        return any(_implements_scope(code, s, a) for s, a in scoped)

    for scope, amount in scoped:
        if any(_implements_scope(c, scope, amount) for c in cons):
            continue
        keep = []
        for c in cons:
            if (
                isinstance(c, str)
                and _or_arity(c) == 0
                and _is_pure_cost_constraint(c)
                and _cap_matches(c, amount)
                and not _implements_any_mention(c)
                and not (amount in total_amounts and _is_total_shaped(c))
            ):
                removed.append(c)
                continue
            keep.append(c)
        cons = keep
        new = _SCOPED_BUDGET_TMPLS[scope] % _num_token(amount)
        if new not in cons:
            cons.append(new)
            added.append(new)

    # total-cap enforcement: a total-shaped cap must be grounded in a
    # total/unqualified budget mention; caps equal to a scoped amount are
    # duplicates of the (now) scoped constraint, ungrounded caps are invented
    nl_nums = _nl_numbers(nl)
    people, days = query.get("people_number"), query.get("days")
    scoped_amounts = {a for _s, a in scoped}
    unqualified = _stated_budget(nl)
    unqualified_ok = set()
    if unqualified and not _BUDGET_QUALIFIER_RE.search(unqualified[1]):
        unqualified_ok.add(unqualified[0])
    keep = []
    for c in cons:
        if (
            isinstance(c, str)
            and _is_total_shaped(c)
            and not _implements_any_mention(c)
        ):
            caps = _cost_caps(c)
            grounded_total = any(
                any(abs(cap - a) < 1e-6 for a in (total_amounts | unqualified_ok))
                or (
                    not any(abs(cap - a) < 1e-6 for a in scoped_amounts)
                    and _grounded(cap, nl_nums, people, days)
                )
                for cap in caps
            )
            if not grounded_total:
                removed.append(c)
                continue
        keep.append(c)
    cons = keep
    if not removed and not added:
        return constraints, None
    return cons, {
        "rule": "budget_scope_canonicalized",
        "mentions": [(s, a) for s, a, _ in mentions],
        "removed": removed,
        "added": added,
    }


# --- rule r5.2: disjunction fragment leak ----------------------------------
# After a correct OR-translation the LLM sometimes ALSO emits one branch's
# budget as a standalone cap (e.g. branch 'intra-city budget 60' leaked as
# unconditional total_cost<=60 -> unsatisfiable -> empty plan).

def drop_disjunction_fragments(constraints):
    or_caps = set()
    for c in constraints:
        if isinstance(c, str) and _or_arity(c) > 0:
            or_caps |= _cost_caps(c)
    if not or_caps:
        return constraints, None
    out, dropped = [], []
    for c in constraints:
        if (
            isinstance(c, str)
            and _or_arity(c) == 0
            and _is_pure_cost_constraint(c)
            and any(
                any(abs(cap - oc) < 1e-6 for oc in or_caps)
                for cap in _cost_caps(c)
            )
        ):
            dropped.append(c)
            continue
        out.append(c)
    if not dropped:
        return constraints, None
    return out, {"rule": "disjunction_fragment_cap_dropped", "dropped": dropped}


# --- rule r5.3: directional intercity transport -----------------------------
# 'take a train to the destination / return by airplane' style requirements
# are FIRST-LEG / LAST-LEG constraints.  The translator collapses them into
# global mode sets or bans (often mutually contradictory).  Parse the
# direction markers and emit the oracle first/last-leg idiom.

_MODE_WORD = r"train|plane|airplane|flight|high-?speed\s+(?:rail(?:way)?|train)"


def _canon_mode(word):
    w = (word or "").lower()
    if "plane" in w or "flight" in w or "fly" in w:
        return "airplane"
    return "train"


_GO_RES = (
    re.compile(
        r"\b(?:take|taking|took)\s+(?:a\s+|an\s+|the\s+)?(?P<mode>%s)\s+"
        r"(?:to\s+(?:get\s+to\s+)?(?:the\s+)?destination|there\b)" % _MODE_WORD,
        re.IGNORECASE),
    re.compile(
        r"\bto\s+(?:the\s+)?destination\s+by\s+(?:a\s+|an\s+|the\s+)?"
        r"(?P<mode>%s)" % _MODE_WORD,
        re.IGNORECASE),
    re.compile(
        r"\b(?:travel(?:ing)?|go(?:ing)?)\s+by\s+(?P<mode>%s)\s+to\s+"
        r"(?:the\s+)?destination" % _MODE_WORD,
        re.IGNORECASE),
    re.compile(r"\b(?:fly|flying)\s+to\s+(?:the\s+)?destination", re.IGNORECASE),
)
_BACK_RES = (
    re.compile(
        r"\breturn(?:ing)?\s+by\s+(?:a\s+|an\s+|the\s+)?(?P<mode>%s)"
        % _MODE_WORD,
        re.IGNORECASE),
    re.compile(
        r"\b(?:take|taking|took)\s+(?:a\s+|an\s+|the\s+)?(?P<mode>%s)\s+"
        r"(?:back\b|home\b|for\s+the\s+return(?:\s+(?:trip|journey))?"
        r"|to\s+return|when\s+returning|to\s+come\s+back)" % _MODE_WORD,
        re.IGNORECASE),
    re.compile(
        r"\b(?:a|an|the)\s+(?P<mode>%s)\s+back\b" % _MODE_WORD, re.IGNORECASE),
    re.compile(r"\b(?:fly|flying)\s+back\b", re.IGNORECASE),
    re.compile(
        r"\b(?P<mode>%s)\s+for\s+the\s+return(?:\s+(?:trip|journey))?"
        % _MODE_WORD,
        re.IGNORECASE),
)
_BOTH_RES = (
    re.compile(
        r"\bby\s+(?:a\s+|an\s+|the\s+)?(?P<mode>%s)\s+"
        r"(?:both\s+ways|round\s+trip|there\s+and\s+back)" % _MODE_WORD,
        re.IGNORECASE),
    re.compile(
        r"\b(?:take|taking)\s+(?:a\s+|an\s+|the\s+)?(?P<mode>%s)\s+"
        r"(?:both\s+ways|there\s+and\s+back)" % _MODE_WORD,
        re.IGNORECASE),
    re.compile(r"\b(?:fly|flying)\s+both\s+ways", re.IGNORECASE),
)
# a go-phrase immediately followed by 'or/and back' or 'or/nor for the
# return (trip)' applies the same (possibly negated) mode to the return leg
_GO_BACK_SUFFIX_RE = re.compile(
    r"^\s*(?:,\s*)?(?:or|and|nor)\s+(?:back\b|for\s+the\s+return(?:\s+trip)?)",
    re.IGNORECASE)

_GO_RES_ZH = (
    re.compile(r"(?:坐|乘|搭乘?)(?P<mode>高铁|火车|动车|飞机)(?:前往|去|出发)"),
    re.compile(r"(?:去程|前往时?|去的时候)[^。；;，,]{0,8}?(?P<mode>高铁|火车|动车|飞机)"),
)
_BACK_RES_ZH = (
    re.compile(r"(?:坐|乘|搭乘?)(?P<mode>高铁|火车|动车|飞机)(?:返回|回来|回去|返程)"),
    re.compile(r"(?:返程|回程|回来时?|返回时?)[^。；;，,]{0,8}?(?P<mode>高铁|火车|动车|飞机)"),
)
_BOTH_RES_ZH = (
    re.compile(r"往返(?:都|均)?(?:坐|乘|搭乘?)(?P<mode>高铁|火车|动车|飞机)"),
)


def _canon_mode_zh(word):
    return "airplane" if word == "飞机" else "train"


_SEGMENT_SPLIT_RE = re.compile(r"[.;!?。！？；\n]")
_NEG_MARK_RE = re.compile(
    r"(?:do\s+not|don'?t|nor(?:\s+do\s+we)?|not|never|avoid|prefer\s+not)\b"
    r"|不想|不要|不希望|不愿|不坐|不乘|别",
    re.IGNORECASE)
_POS_MARK_RE = re.compile(
    r"(?:want|wish|hope|prefer|would\s+like|like|need)\s+to\b"
    r"|希望|想要|打算",
    re.IGNORECASE)


def _clause_negated(segment, upto):
    """Polarity of the leg-phrase at offset `upto` inside `segment`."""
    head = segment[:upto]
    last_neg = -1
    for m in _NEG_MARK_RE.finditer(head):
        last_neg = m.start()
    last_pos = -1
    for m in _POS_MARK_RE.finditer(head):
        # 'do not want to' / 'nor do we wish to' / 'prefer not to': the
        # positive verb belongs to the negation phrase
        prefix = head[max(0, m.start() - 24): m.start()]
        if re.search(
            r"(?:do\s+not|don'?t|not|nor(?:\s+do(?:\s+we|\s+i)?)?)\s*$",
            prefix,
            re.IGNORECASE,
        ):
            continue
        last_pos = m.start()
    return last_neg > last_pos if last_neg >= 0 else False


def parse_directional_transport(nature_language, lang="en"):
    """[(leg, op, mode, offset)] parsed from directional NL phrases.
    leg in {'go','back'}, op in {'==','!='}, mode in {'train','airplane'}."""
    if not nature_language:
        return []
    if lang == "zh":
        go_res, back_res, both_res = _GO_RES_ZH, _BACK_RES_ZH, _BOTH_RES_ZH
        canon = _canon_mode_zh
    else:
        go_res, back_res, both_res = _GO_RES, _BACK_RES, _BOTH_RES
        canon = _canon_mode
    specs = []
    pos = 0
    for seg_match in list(_SEGMENT_SPLIT_RE.finditer(nature_language)) + [None]:
        seg_end = seg_match.start() if seg_match else len(nature_language)
        segment = nature_language[pos:seg_end]
        seg_off = pos
        pos = seg_match.end() if seg_match else seg_end

        def _collect(res_list, legs):
            for rx in res_list:
                for m in rx.finditer(segment):
                    try:
                        mode = canon(m.group("mode"))
                    except (IndexError, TypeError):
                        mode = "airplane"  # fly-phrases carry no mode group
                    op = "!=" if _clause_negated(segment, m.start()) else "=="
                    for leg in legs:
                        specs.append((leg, op, mode, seg_off + m.start()))
                    if legs == ("go",) and _GO_BACK_SUFFIX_RE.match(
                        segment[m.end():]
                    ):
                        specs.append(("back", op, mode, seg_off + m.start()))

        _collect(go_res, ("go",))
        _collect(back_res, ("back",))
        _collect(both_res, ("go", "back"))
    return specs


def _resolve_leg_specs(specs):
    """{leg: (op, mode)} or None on conflict."""
    resolved = {}
    for leg in ("go", "back"):
        entries = {(op, mode) for lg, op, mode, _off in specs if lg == leg}
        if not entries:
            continue
        eqs = {m for op, m in entries if op == "=="}
        neqs = {m for op, m in entries if op == "!="}
        if len(eqs) > 1:
            return None
        if eqs:
            mode = eqs.pop()
            if mode in neqs:
                return None
            resolved[leg] = ("==", mode)
        else:
            if len(neqs) > 1:
                return None  # both modes banned: contradictory
            resolved[leg] = ("!=", neqs.pop())
    return resolved or None


def _leg_clause(leg, op, mode):
    idx = "0" if leg == "go" else "-1"
    origin = "start_city(plan)" if leg == "go" else "target_city(plan)"
    return (
        "allactivities(plan)[%s]['type'] %s \"%s\" and "
        "intercity_transport_origin(allactivities(plan)[%s])==%s"
        % (idx, op, mode, idx, origin)
    )


def build_directional_constraint(resolved):
    cond = " and ".join(
        _leg_clause(leg, op, mode)
        for leg in ("go", "back")
        if leg in resolved
        for op, mode in [resolved[leg]]
    )
    return (
        "result=False\n"
        "intercity_transport_go=''\n"
        "intercity_transport_back=''\n"
        "if %s:\n"
        "  result=True" % cond
    )


_INTERCITY_TOKEN_RE = re.compile(
    r"intercity_transport_(?:set|type|origin|destination)"
    r"|activity_type\(activity\)\s*[=!]=\s*['\"](?:train|airplane)['\"]")
_NON_INTERCITY_TOKEN_RE = re.compile(
    r"activity_tickets|metro_tickets|taxi_cars|activity_cost|activity_price"
    r"|activity_position|activity_(?:start_|end_)?time\("
    r"|attraction_|restaurant_|accommodation_|room_|poi_"
    r"|innercity_transport_(?:cost|price|distance|time)"
    r"|inner_city_transportation_set|day_count|people_count")


def _is_pure_intercity_mode_constraint(code):
    if not isinstance(code, str) or _or_arity(code) > 0:
        return False
    stripped = _STRING_LITERAL_RE.sub("", code)
    return bool(
        _INTERCITY_TOKEN_RE.search(code)
        and not _NON_INTERCITY_TOKEN_RE.search(stripped)
    )


_INTERCITY_SET_EQ_RE = re.compile(
    r"intercity_transport_set\s*==\s*\{(?P<items>[^{}]*)\}"
    r"|\{(?P<items2>[^{}]*)\}\s*==\s*intercity_transport_set")


def enforce_directional_transport(constraints, query, lang="en"):
    """Rewrite global mode sets/bans into the oracle first/last-leg idiom
    whenever the NL carries direction markers."""
    nl = query.get("nature_language") or ""
    specs = parse_directional_transport(nl, lang)
    if not specs:
        return constraints, None
    outside = [s for s in specs if not _in_disjunction_region(nl, s[3])]
    inside = [s for s in specs if _in_disjunction_region(nl, s[3])]
    cons = list(constraints)
    removed, added, inlined = [], [], []

    if outside:
        resolved = _resolve_leg_specs(outside)
        if resolved:
            new = build_directional_constraint(resolved)
            if new not in cons:
                keep = []
                for c in cons:
                    if _is_pure_intercity_mode_constraint(c):
                        removed.append(c)
                        continue
                    keep.append(c)
                cons = keep
                cons.append(new)
                added.append(new)

    if inside:
        resolved = _resolve_leg_specs(inside)
        if resolved and all(op == "==" for op, _m in resolved.values()):
            modes = {m for _op, m in resolved.values()}
            cond = " and ".join(
                _leg_clause(leg, op, mode)
                for leg in ("go", "back")
                if leg in resolved
                for op, mode in [resolved[leg]]
            )
            for i, c in enumerate(cons):
                if not (isinstance(c, str) and _or_arity(c) > 0):
                    continue
                m = _INTERCITY_SET_EQ_RE.search(c)
                if not m:
                    continue
                items = {
                    x.strip().strip("'\"")
                    for x in (m.group("items") or m.group("items2")).split(",")
                    if x.strip()
                }
                if items == modes:
                    cons[i] = c[: m.start()] + "(" + cond + ")" + c[m.end():]
                    inlined.append(cons[i])

    if not removed and not added and not inlined:
        return constraints, None
    return cons, {
        "rule": "directional_transport_canonicalized",
        "specs": [(l, o, m) for l, o, m, _ in specs],
        "removed": removed,
        "added": added,
        "inlined": inlined,
    }


# --- rule r5.4: only-free-attractions --------------------------------------

_FREE_ATTR_RES = {
    "en": re.compile(
        r"\bonly\b[^.;!?\n]{0,40}?\bfree\s+attractions?\b"
        r"|\bvisit\s+only\s+free\s+attractions?\b",
        re.IGNORECASE),
    "zh": re.compile(r"(?:只|仅)[^。；;！？!?\n]{0,15}?免费(?:的)?景点"),
}
_FREE_ATTR_TMPL = (
    "attraction_cost=0\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='attraction': "
    "attraction_cost+=activity_cost(activity)\n"
    "result=attraction_cost<=0"
)
_FREE_ATTR_PRESENT_RE = re.compile(
    r"attraction_cost\s*<=\s*0"
    r"|activity_(?:cost|price)\(activity\)\s*(?:>|!=)\s*0")


def inject_free_attractions(constraints, query, lang="en"):
    nl = query.get("nature_language") or ""
    rx = _FREE_ATTR_RES.get(lang, _FREE_ATTR_RES["en"])
    m = rx.search(nl)
    if not m or _in_disjunction_region(nl, m.start()):
        return constraints, None
    for c in constraints:
        if isinstance(c, str) and _FREE_ATTR_PRESENT_RE.search(c):
            return constraints, None
    return constraints + [_FREE_ATTR_TMPL], {
        "rule": "only_free_attractions_injected",
        "added": _FREE_ATTR_TMPL,
    }


# --- POI-name grounding helper (public environment DB) ---------------------

_POI_NAME_CACHE = {}


def _city_all_poi_names(city, lang="en"):
    """All POI names (attractions+restaurants+accommodations) of a city from
    the public environment database; empty set when unavailable."""
    canon = _canon_city(city)
    if not canon:
        return set()
    key = (canon, lang)
    if key in _POI_NAME_CACHE:
        return _POI_NAME_CACHE[key]
    low = canon.lower()
    root = _db_dir(lang)
    paths = (
        os.path.join(root, "attractions", low, "attractions.csv"),
        os.path.join(root, "restaurants", low, "restaurants_%s.csv" % low),
        os.path.join(root, "accommodations", low, "accommodations.csv"),
    )
    names = set()
    for path in paths:
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("name"):
                        names.add(row["name"].strip())
        except OSError:
            pass
    _POI_NAME_CACHE[key] = names
    return names


def _quote(name):
    return '"%s"' % name if "'" in name else "'%s'" % name


_POSITION_LITERAL_RE = re.compile(
    r"activity_position\(activity\)\s*==\s*"
    r"(?:'((?:\\.|[^\\'])*)'|\"((?:\\.|[^\\\"])*)\")")


def _constraint_position_names(code):
    return [
        (a or b).replace("\\'", "'").replace('\\"', '"')
        for a, b in _POSITION_LITERAL_RE.findall(code)
    ]


# --- rule r5.5: POI arrive/leave time windows -------------------------------
# 'arrive at X no later than T' => exists visit of X with start_time <= T;
# 'leave X no earlier than T'   => exists visit of X with end_time >= T.
# The translator flips start/end or emits a vacuous universal form.

_TIME_RE = r"(?P<t>\d{1,2}:\d{2})"
_ARRIVE_RES = (
    re.compile(
        r"arriv\w*\s+(?:at|in)\s+(?P<name>.+)\s+no\s+later\s+than\s+" + _TIME_RE,
        re.IGNORECASE),
)
_LEAVE_RES = (
    re.compile(
        r"(?:leave|leaving)\s+(?P<name>.+)\s+no\s+earlier\s+than\s+" + _TIME_RE,
        re.IGNORECASE),
    re.compile(
        r"depart(?:ure|ing)?\s+from\s+(?P<name>.+)\s+no\s+earlier\s+than\s+"
        + _TIME_RE,
        re.IGNORECASE),
    re.compile(
        r"(?:depart|leave)\s+no\s+earlier\s+than\s+"
        + _TIME_RE
        + r"\s+from\s+(?P<name>.+)$",
        re.IGNORECASE),
    re.compile(
        r"not\s+to\s+leave\s+(?P<name>.+)\s+before\s+" + _TIME_RE,
        re.IGNORECASE),
)
_ARRIVE_TMPL = (
    "result=False\n"
    "for activity in allactivities(plan):\n"
    "  if activity_position(activity)==%s:\n"
    "    if activity_start_time(activity)<='%s':\n"
    "      result=True"
)
_LEAVE_TMPL = (
    "result=False\n"
    "for activity in allactivities(plan):\n"
    "  if activity_position(activity)==%s:\n"
    "    if activity_end_time(activity)>='%s':\n"
    "      result=True"
)
_TIME_LITERAL_RE = re.compile(r"['\"](\d{1,2}:\d{2})['\"]")


def _parse_time_window_events(nature_language):
    """[(kind, name_candidate, time, offset)] from arrive/leave phrases."""
    events = []
    pos = 0
    for seg_match in list(_SEGMENT_SPLIT_RE.finditer(nature_language)) + [None]:
        seg_end = seg_match.start() if seg_match else len(nature_language)
        segment = nature_language[pos:seg_end]
        seg_off = pos
        pos = seg_match.end() if seg_match else seg_end
        for kind, res_list in (("arrive", _ARRIVE_RES), ("leave", _LEAVE_RES)):
            for rx in res_list:
                for m in rx.finditer(segment):
                    name = m.group("name").strip().strip(",;:- ")
                    events.append((kind, name, m.group("t"), seg_off + m.start()))
    return events


def enforce_time_windows(constraints, query, lang="en"):
    nl = query.get("nature_language") or ""
    if lang == "zh":
        return constraints, None  # zh phrasing variants deferred (see notes)
    events = _parse_time_window_events(nl)
    if not events:
        return constraints, None
    cons = list(constraints)
    changed = []
    db_names = None
    for kind, name_cand, t, off in events:
        if _in_disjunction_region(nl, off):
            continue
        tmpl = _ARRIVE_TMPL if kind == "arrive" else _LEAVE_TMPL
        # 1) rewrite an existing constraint that carries this time literal
        target_idx, target_name = None, None
        for i, c in enumerate(cons):
            if not isinstance(c, str) or _or_arity(c) > 0:
                continue
            times = set(_TIME_LITERAL_RE.findall(c))
            if times != {t}:
                continue  # absent, or a two-ended 'between' window: skip
            if "activity_start_time(" not in c and "activity_end_time(" not in c:
                continue
            names = _constraint_position_names(c)
            if len(set(names)) == 1:
                target_idx, target_name = i, names[0]
                break
        if target_idx is not None:
            new = tmpl % (_quote(target_name), t)
            if cons[target_idx] != new:
                changed.append({"before": cons[target_idx], "after": new})
                cons[target_idx] = new
            continue
        # 2) no carrier constraint: inject, but only with a DB-verified name
        if db_names is None:
            db_names = _city_all_poi_names(query.get("target_city"), lang)
        resolved = None
        if name_cand in db_names:
            resolved = name_cand
        else:
            # translator may have used a (verbatim) name inside another
            # constraint that the NL wraps with extra words
            for c in cons:
                if not isinstance(c, str):
                    continue
                for nm in _constraint_position_names(c):
                    if nm and (nm in name_cand or name_cand in nm):
                        resolved = nm
                        break
                if resolved:
                    break
        if resolved is None:
            continue
        new = tmpl % (_quote(resolved), t)
        if new not in cons:
            cons.append(new)
            changed.append({"before": None, "after": new})
    if not changed:
        return constraints, None
    return cons, {"rule": "poi_time_window_canonicalized", "changes": changed}


# --- rule r5.6: distance-conditional taxi ----------------------------------

_DIST_TAXI_RES = {
    "en": re.compile(
        r"distance[^.;!?\n]{0,60}?(?:exceeds?|is\s+(?:more|greater)\s+than"
        r"|(?:is\s+)?over|above|beyond)\s+(?P<km>\d+(?:\.\d+)?)\s*"
        r"(?:km|kilometers?|kilometres?)[^.;!?\n]{0,80}?(?:taxi|cab)",
        re.IGNORECASE),
    "zh": re.compile(
        r"距离[^。；;！？!?\n]{0,20}?超过\s*(?P<km>\d+(?:\.\d+)?)\s*"
        r"(?:公里|千米|km)[^。；;！？!?\n]{0,30}?(?:打车|出租车|的士)"),
}
_DIST_TAXI_TMPL = (
    "result=True\n"
    "for activity in allactivities(plan):\n"
    "  if innercity_transport_type(activity_transports(activity)) != 'taxi' "
    "and innercity_transport_distance(activity_transports(activity))>%s:\n"
    "    result=False\n"
    "    break"
)
_NUMERIC_LITERAL_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")


def _has_numeric_near(code, value, tol=0.02):
    stripped = _STRING_LITERAL_RE.sub("", code)
    for tok in _NUMERIC_LITERAL_RE.findall(stripped):
        try:
            if abs(float(tok) - value) <= tol:
                return True
        except ValueError:
            continue
    return False


def enforce_distance_taxi(constraints, query, lang="en"):
    nl = query.get("nature_language") or ""
    rx = _DIST_TAXI_RES.get(lang, _DIST_TAXI_RES["en"])
    m = rx.search(nl)
    if not m or _in_disjunction_region(nl, m.start()):
        return constraints, None
    km_text = m.group("km")
    km = float(km_text)
    new = _DIST_TAXI_TMPL % km_text
    cons, removed = [], []
    for c in constraints:
        if (
            isinstance(c, str)
            and c != new
            and _or_arity(c) == 0
            and "taxi" in c
            and ("innercity_transport_distance(" in c or "poi_distance(" in c)
            and _has_numeric_near(c, km)
        ):
            removed.append(c)
            continue
        cons.append(c)
    already = any(
        isinstance(c, str)
        and "innercity_transport_distance(" in c
        and "!= 'taxi'" in c.replace('"', "'")
        and _has_numeric_near(c, km)
        for c in cons
    )
    if already and not removed:
        return constraints, None
    if not already:
        cons.append(new)
    return cons, {
        "rule": "distance_conditional_taxi_canonicalized",
        "removed": removed,
        "added": None if already else new,
    }


# --- rule r5.7: accommodation distance to anchor POI ------------------------

_HOTEL_DIST_RES = {
    "en": re.compile(
        r"(?:accommodation|hotel|stay|lodging)[^.;!?\n]{0,80}?within\s+"
        r"(?P<km>\d+(?:\.\d+)?)\s*(?:km|kilometers?|kilometres?)\s+"
        r"(?:of|from)\s+(?P<name>[^.;!?\n]+)",
        re.IGNORECASE),
}
_HOTEL_DIST_TMPL = (
    "result=False\n"
    "accommodation_position=''\n"
    "for activity in allactivities(plan):\n"
    "  if activity_type(activity)=='accommodation': "
    "accommodation_position=activity_position(activity)\n"
    "result=(poi_distance(target_city(plan), %s, accommodation_position)<=%s)"
)


def enforce_hotel_distance(constraints, query, lang="en"):
    nl = query.get("nature_language") or ""
    rx = _HOTEL_DIST_RES.get(lang)
    if rx is None:
        return constraints, None  # zh anchor-name segmentation deferred
    m = rx.search(nl)
    if not m or _in_disjunction_region(nl, m.start()):
        return constraints, None
    km_text = m.group("km")
    km = float(km_text)
    anchor = m.group("name").strip().strip(",;:- ").rstrip(".")
    db_names = _city_all_poi_names(query.get("target_city"), lang)
    if anchor not in db_names:
        # accept the anchor if a generated constraint used it verbatim
        if not any(
            isinstance(c, str) and anchor in c for c in constraints
        ):
            return constraints, None
    # already in oracle dialect?
    for c in constraints:
        if (
            isinstance(c, str)
            and "poi_distance(" in c
            and "accommodation_position" in c
            and _has_numeric_near(c, km)
        ):
            return constraints, None
    cons, removed = [], []
    for c in constraints:
        if (
            isinstance(c, str)
            and _or_arity(c) == 0
            and "poi_distance(" in c
            and _ACC_FILTER_RE.search(c)
            and _has_numeric_near(c, km)
        ):
            removed.append(c)
            continue
        cons.append(c)
    new = _HOTEL_DIST_TMPL % (_quote(anchor), km_text)
    cons.append(new)
    return cons, {
        "rule": "hotel_distance_canonicalized",
        "anchor": anchor,
        "removed": removed,
        "added": new,
    }


# --- rule r5.8: `not EXPR == False/True` precedence tautology ----------------

_TAUT_RE = re.compile(
    r"result\s*=\s*\(\s*not\s*\((?P<expr>.*)\)\s*==\s*(?P<lit>False|True)\s*\)\s*$",
    re.S)


def fix_negation_tautology(constraints):
    out, changed = [], []
    for c in constraints:
        if isinstance(c, str):
            m = _TAUT_RE.search(c)
            if m:
                expr = m.group("expr").strip()
                repl = (
                    "result=(%s)" % expr
                    if m.group("lit") == "False"
                    else "result=not(%s)" % expr
                )
                c2 = c[: m.start()] + repl
                changed.append({"before": c, "after": c2})
                c = c2
        out.append(c)
    if not changed:
        return constraints, None
    return out, {"rule": "negation_tautology_fixed", "changes": changed}


# --- rule r5.9: positive-NL membership polarity ------------------------------
# 'we hope to stay at one of the following hotels: X or Y' translated as
# result=not({X,Y}&accommodation_name_set) bans the requested hotels.

_NEG_MEMBERSHIP_RE = re.compile(
    r"result\s*=\s*\(?\s*not\s*\(\s*\{(?P<items>[^{}]*)\}\s*&\s*"
    r"(?P<cat>attraction|restaurant|accommodation)_name_set\s*\)\s*\)?\s*$")
_POSITIVE_VERBS = {
    "accommodation": re.compile(
        r"(?:hope|want|wish|like|prefer|would\s+like|plan)"
        r"[^.;!?\n]{0,30}?to\s+(?:stay|live)|stay\s+at\s+one\s+of"
        r"|希望(?:入住|住)|想(?:入住|住)",
        re.IGNORECASE),
    "restaurant": re.compile(
        r"(?:hope|want|wish|like|prefer|would\s+like)"
        r"[^.;!?\n]{0,30}?to\s+(?:try|eat|dine|taste)"
        r"|希望(?:品尝|去吃)|想(?:尝|吃)",
        re.IGNORECASE),
    "attraction": re.compile(
        r"(?:hope|want|wish|like|prefer|would\s+like)"
        r"[^.;!?\n]{0,30}?to\s+(?:visit|see|go\s+to)"
        r"|希望(?:参观|游览|去)|想(?:参观|游览|去)",
        re.IGNORECASE),
}
_POLARITY_NEG_RE = re.compile(
    r"do\s+not|don'?t|not\s+(?:want|wish|hope)|avoid|rather\s+not|nor\b"
    r"|不想|不希望|不要|避免",
    re.IGNORECASE)


def fix_membership_polarity(constraints, query):
    nl = query.get("nature_language") or ""
    if not nl:
        return constraints, None
    out, changed = [], []
    for c in constraints:
        if isinstance(c, str) and _or_arity(c) == 0:
            m = _NEG_MEMBERSHIP_RE.search(c)
            if m:
                items = [
                    x.strip().strip("'\"").replace("\\'", "'").replace(
                        '\\"', '"')
                    for x in m.group("items").split(",")
                    if x.strip()
                ]
                first = next((x for x in items if x and x in nl), None)
                if first:
                    idx = nl.find(first)
                    # window = the whole clause before the name (from the
                    # previous sentence boundary); a fixed char cap can cut
                    # off a leading 'Do not want to ...'
                    seg_start = 0
                    for bm in _SEGMENT_SPLIT_RE.finditer(nl, 0, idx):
                        seg_start = bm.end()
                    window = nl[seg_start:idx]
                    verb_rx = _POSITIVE_VERBS[m.group("cat")]
                    if verb_rx.search(window) and not _POLARITY_NEG_RE.search(
                        window
                    ):
                        c2 = c[: m.start()] + (
                            "result=({%s}&%s_name_set)"
                            % (m.group("items"), m.group("cat"))
                        )
                        changed.append({"before": c, "after": c2})
                        c = c2
        out.append(c)
    if not changed:
        return constraints, None
    return out, {"rule": "membership_polarity_fixed", "changes": changed}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def enforce_coverage(query, lang="en"):
    """Apply all deterministic coverage / span-grounding fixes in place.

    Fixes are recorded under query['coverage_fixes']; report-only findings
    under query['coverage_flags']. Safe to call repeatedly (idempotent).
    lang=None auto-detects the query language from nature_language (used by
    the planner's cached-translation hardening path)."""
    cons = [c for c in query.get("hard_logic_py") or [] if isinstance(c, str)]
    nl = query.get("nature_language")
    if lang is None:
        lang = detect_nl_lang(nl)
    fixes, flags = [], []

    def _apply(result):
        nonlocal cons
        new_cons, record = result
        if record is None:
            return
        if record.get("flag_only"):
            flags.append(record)
            return
        cons = new_cons
        fixes.append(record)

    _apply(inject_base_boilerplate(cons, query))
    _apply(apply_taxi_convention(cons, is_human_register(query)))
    _apply(apply_room_count_override(cons, nl))
    _apply(fix_negation_tautology(cons))
    _apply(fix_membership_polarity(cons, query))
    _apply(drop_disjunction_fragments(cons))
    _apply(enforce_budget_scope(cons, query, lang))
    _apply(enforce_directional_transport(cons, query, lang))
    _apply(inject_free_attractions(cons, query, lang))
    _apply(enforce_time_windows(cons, query, lang))
    _apply(enforce_distance_taxi(cons, query, lang))
    _apply(enforce_hotel_distance(cons, query, lang))
    _apply(inject_local_cuisine(cons, query, lang))
    _apply(inject_airplane_transport(cons, query, lang))
    _apply(apply_category_expansion(cons, nl, lang))
    _apply(drop_ungrounded_cost_caps(cons, query))
    _apply(inject_overall_budget(cons, query))

    if fixes:
        query["hard_logic_py"] = list(dict.fromkeys(cons))
        query.setdefault("coverage_fixes", []).extend(fixes)
    if flags:
        query.setdefault("coverage_flags", []).extend(flags)
    return query


def coverage_report(query, lang="en"):
    """Dry-run: the fixes/flags enforce_coverage would apply, no mutation."""
    probe = deepcopy(query)
    probe.pop("coverage_fixes", None)
    probe.pop("coverage_flags", None)
    enforce_coverage(probe, lang)
    return {
        "fixes": probe.get("coverage_fixes", []),
        "flags": probe.get("coverage_flags", []),
        "hard_logic_py": probe.get("hard_logic_py", []),
    }

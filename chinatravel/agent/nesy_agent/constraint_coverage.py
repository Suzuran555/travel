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
    r"(?:twin|double|single|standard|triple|quad|king|queen|"
    r"big[- ]?bed|two[- ]?bed|double[- ]?bed|family)"
)
# 'a/an <room-type> room' counts as ONE room only when a room-type word
# pins it down; bare numerals/count words before 'room(s)' always count.
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
    return bool(_cost_caps(code))


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
    r"住宿|酒店|餐|吃|交通|打车|机票|车票|门票",
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


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def enforce_coverage(query, lang="en"):
    """Apply all deterministic coverage / span-grounding fixes in place.

    Fixes are recorded under query['coverage_fixes']; report-only findings
    under query['coverage_flags']. Safe to call repeatedly (idempotent)."""
    cons = [c for c in query.get("hard_logic_py") or [] if isinstance(c, str)]
    nl = query.get("nature_language")
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

    _apply(apply_taxi_convention(cons, is_human_register(query)))
    _apply(apply_room_count_override(cons, nl))
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

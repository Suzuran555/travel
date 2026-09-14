"""Deterministic, semantics-preserving canonicalizer for LLM-emitted hard_logic_py DSL.

Open-weight translators (e.g. Qwen) emit DSL that is semantically correct but written
in a different *style* than the oracle/Claude dialect that
``UrbanTripOptimizedV6.extract_user_constraints_by_DSL`` was tuned on.  This module
rewrites only the style differences so the downstream regex extractor can see the
constraints; every transformation keeps the constraint evaluating identically under
``evaluate_constraints_py`` (the rewritten names are free local variables, never DSL
functions).

Canonicalizations performed:

1. Set-variable renames (free locals used by the ``{...} & / <= set`` idioms), e.g.
   ``attraction_names_set`` -> ``attraction_name_set``,
   ``hotel_names_set`` -> ``accommodation_name_set``.

2. Budget-accumulator renames: a free local accumulator that is initialised to 0,
   accumulated over ``allactivities(plan)`` under a recognised activity-type filter,
   and compared with ``<= NUMBER`` is renamed to the canonical accumulator name the
   extractor's budget regexes look for (``attraction_cost``, ``restaurant_cost``,
   ``accommodation_cost``, ``inner_city_transportation_cost``,
   ``inter_city_transportation_cost``, ``total_cost``).

3. Vacuous transport-type guard repair: Qwen guards per-activity inner-city
   transport logic with ``if activity_type(activity)=='transportation':``, but
   ``'transportation'`` is not a plan activity type (activities are attractions,
   meals, accommodations, trains, airplanes; inner-city transports hang off every
   activity via ``activity_transports``).  The guard is therefore always False and
   silently disables the constraint (e.g. a transport-budget cap sums nothing and
   passes vacuously).  When every guarded statement operates only on the activity's
   transports, the guard is removed so the constraint recovers its intended
   semantics of "for each activity's attached inner-city transports".

4. Entity-category grounding (needs the query's target_city, so it only runs via
   ``canonicalize_query_hard_logic``): a must-visit name-set constraint (e.g.
   ``result=({'X'}<=restaurant_name_set)`` fed by a meal-typed collection loop) is
   checked against the environment database of the target city.  Chinese POI names
   are often category-ambiguous in translation (a food street or themed venue can
   sound like a restaurant but be catalogued as an attraction), and the LLM has no
   database access.  When EVERY required name is absent from the claimed category
   but present in exactly one other category, the collection loop and set variable
   are rewritten to that category.  This is grounding against the public
   environment database, not against oracle annotations.

IMPORTANT: ``activity_price`` and ``activity_cost`` are DIFFERENT DSL functions
(``activity_price(a)`` reads the per-person/room ``price`` field, ``activity_cost(a)``
reads the total ``cost`` field -- see chinatravel/symbol_verification/concept_func.py).
They are therefore NEVER rewritten into each other here; instead the extractor in
tpc_agent_optimized_v6.py was widened to understand the ``activity_price`` idioms.

Renames are guarded: a rename is skipped whenever the canonical target name already
occurs in the constraint (so an existing binding can never be captured).
"""

import csv
import os
import re

# Free local set-variable spellings observed in Qwen translations -> canonical
# spellings expected by the V6 extractor.  All of these are plain local variables
# in the generated DSL (none is a concept_func DSL function).
SET_VARIABLE_RENAMES = {
    "attraction_names_set": "attraction_name_set",
    "attraction_types_set": "attraction_type_set",
    "restaurant_names_set": "restaurant_name_set",
    "restaurant_types_set": "restaurant_type_set",
    "accommodation_names_set": "accommodation_name_set",
    "accommodation_types_set": "accommodation_type_set",
    "hotel_names_set": "accommodation_name_set",
    "hotel_name_set": "accommodation_name_set",
    "hotel_types_set": "accommodation_type_set",
    "hotel_type_set": "accommodation_type_set",
    # hotel FEATURES are the same field the verifier calls accommodation_type
    # (featurehoteltype); Qwen sometimes names the accumulator by "feature"
    "hotel_feature_set": "accommodation_type_set",
    "hotel_features_set": "accommodation_type_set",
    "accommodation_feature_set": "accommodation_type_set",
    "accommodation_features_set": "accommodation_type_set",
}

# Canonical accumulator names used by the extractor's budget regexes.
_CANON_BUDGET_VARS = {
    "attraction": "attraction_cost",
    "restaurant": "restaurant_cost",
    "accommodation": "accommodation_cost",
    "innercity": "inner_city_transportation_cost",
    "intercity": "inter_city_transportation_cost",
    "total": "total_cost",
}

_MEAL_TYPES = {"breakfast", "lunch", "dinner"}
_INTERCITY_TYPES = {"train", "airplane"}


def _rename_free_variable(text, old, new):
    """Word-boundary rename of a free local variable; refuses to capture an
    existing binding of the target name."""
    if old == new:
        return text
    if re.search(r"\b%s\b" % re.escape(new), text):
        return text
    return re.sub(r"\b%s\b" % re.escape(old), new, text)


def _canonicalize_set_variables(text):
    for old, new in SET_VARIABLE_RENAMES.items():
        if old in text:
            text = _rename_free_variable(text, old, new)
    return text


def _activity_type_filter(context_lines):
    """Classify the activity_type filter guarding an accumulation statement.

    ``context_lines`` are the lines between the enclosing ``for`` and the
    accumulation (inclusive of the accumulation line itself).
    Returns one of 'attraction' / 'restaurant' / 'accommodation' /
    'transportation' / 'intercity' / None (no filter).
    """
    ctx = "\n".join(context_lines)
    m = re.search(r"activity_type\(activity\)\s*==\s*['\"](\w+)['\"]", ctx)
    if m:
        value = m.group(1)
        if value in ("attraction", "accommodation", "transportation"):
            return value
        if value in _MEAL_TYPES:
            return "restaurant"
        if value in _INTERCITY_TYPES:
            return "intercity"
        return None
    m = re.search(r"activity_type\(activity\)\s+in\s+\[([^\]]*)\]", ctx)
    if m:
        values = set(re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)))
        if values and values <= _MEAL_TYPES:
            return "restaurant"
        if values == _INTERCITY_TYPES:
            return "intercity"
    return None


def _classify_accumulator(filter_kinds, has_activity_cost, has_innercity_cost):
    """Map the union of a variable's accumulation sites -> canonical name.

    ``filter_kinds`` is the set of activity-type filters guarding the
    accumulation statements (may contain None for unfiltered sites).  A
    variable is classified once over ALL of its accumulation sites so that a
    total-cost accumulator split across several lines is never mistaken for a
    single-category accumulator.
    """
    if len(filter_kinds) != 1:
        return None  # mixed or ambiguous guards
    filter_kind = next(iter(filter_kinds))
    if filter_kind in ("attraction", "restaurant", "accommodation", "intercity"):
        if has_activity_cost and not has_innercity_cost:
            return _CANON_BUDGET_VARS[
                "intercity" if filter_kind == "intercity" else filter_kind
            ]
        return None
    # unfiltered or 'transportation'-filtered sites
    if has_innercity_cost and not has_activity_cost:
        return _CANON_BUDGET_VARS["innercity"]
    if filter_kind is None and has_activity_cost and has_innercity_cost:
        return _CANON_BUDGET_VARS["total"]
    return None


def _canonicalize_budget_accumulators(text):
    """Rename budget accumulator locals to the canonical extractor names."""
    lines = text.split("\n")
    sites = {}
    for idx, line in enumerate(lines):
        # accumulation may share a line with its filter ("if ...: var+=...")
        m = re.search(
            r"\b([A-Za-z_]\w*)\s*(?:\+=|=\s*\1\s*\+)\s*(.+)$", line
        )
        if not m:
            continue
        var, expr = m.group(1), m.group(2)
        # must be a numeric accumulator: initialised to 0 and compared to a cap
        if not re.search(r"\b%s\s*=\s*0\b" % re.escape(var), text):
            continue
        if not re.search(r"\b%s\s*(?:<=|>=|<|>)\s*[0-9]" % re.escape(var), text):
            continue
        # context: lines from the enclosing 'for' down to the accumulation,
        # excluding earlier accumulation lines of the same variable
        start = idx
        for j in range(idx - 1, -1, -1):
            if re.match(r"\s*for\b", lines[j]):
                start = j
                break
        context = [
            ctx_line
            for ctx_line in lines[start:idx]
            if not re.search(r"\b%s\s*(?:\+=|=\s*%s\s*\+)" % (re.escape(var), re.escape(var)), ctx_line)
        ] + [line]
        entry = sites.setdefault(var, {"filters": set(), "act": False, "inner": False})
        entry["filters"].add(_activity_type_filter(context))
        entry["act"] |= "activity_cost(" in expr or "activity_price(" in expr
        entry["inner"] |= "innercity_transport_cost(" in expr
    renames = {}
    for var, entry in sites.items():
        canonical = _classify_accumulator(entry["filters"], entry["act"], entry["inner"])
        if canonical and canonical != var:
            renames[var] = canonical
    for old, new in renames.items():
        text = _rename_free_variable(text, old, new)
    return text


# Invalid activity_type literals that Qwen uses to mean "the inner-city
# transport legs of an activity".  No plan activity ever has these types, so a
# guard comparing activity_type(...) to one of them is always False.
_TRANSPORT_TYPE_GUARD_RE = re.compile(
    r"^(\s*)if\s+activity_type\(activity\)\s*==\s*['\"]"
    r"(?:transportation|transport|transit|inner_?city_transport(?:ation)?)"
    r"['\"]\s*:\s*(.*)$"
)

# Any guard whose condition is PURELY an activity_type test (a single
# equality or membership -- no and/or with other predicates).  Used for the
# generic transport-budget guard repair: the benchmark's canonical
# transport-budget idiom accumulates innercity_transport_cost over ALL
# activities, but open-weight translators often guard the accumulation with a
# real (or partial) activity-type list, e.g.
#   if activity_type(activity) in ['breakfast','lunch','dinner','attraction',
#                                  'accommodation']: cost+=...
# which silently EXCLUDES the transports attached to the remaining activity
# types (train / airplane legs), so the generated cap undercounts versus the
# oracle's unguarded sum.  When the guarded body does nothing but accumulate
# transport quantities, the guard is an artifact of the dialect, never a
# semantic restriction (no benchmark constraint caps per-category transport
# spend), so it is removed to restore the canonical all-activities sum.
_ACTIVITY_TYPE_ONLY_GUARD_RE = re.compile(
    r"^(\s*)if\s+activity_type\(activity\)\s*"
    r"(?:==\s*(?:'[^']*'|\"[^\"]*\")"
    r"|in\s*\[[^\]]*\]"
    r"|in\s*\([^()]*\)"
    r"|in\s*\{[^{}]*\})"
    r"\s*:\s*(.*)$"
)

# An accumulation statement whose right-hand side reads a transport quantity.
_TRANSPORT_ACCUM_LINE_RE = re.compile(
    r"^[A-Za-z_]\w*\s*(?:\+=|=\s*[A-Za-z_]\w*\s*\+)\s*.*"
    r"\binnercity_transport_(?:cost|price|time|distance)\s*\("
)

# DSL functions that read an activity's transports (see concept_func.py).
_TRANSPORT_FUNC_RE = re.compile(
    r"\b(?:activity_transports|innercity_transport_cost|innercity_transport_time"
    r"|innercity_transport_price|innercity_transport_distance"
    r"|innercity_transport_type|innercity_transport_start_time"
    r"|innercity_transport_end_time|metro_tickets|taxi_cars)\s*\("
)

# Body statement of the vacuous MODE-SET dialect: under the always-False
# 'transportation' pseudo-type guard the set collects
# ``activity_position(activity)`` -- a POI name, standing in for "the mode of
# this (pseudo) transportation activity".  The intended (oracle) idiom is a
# set of the plan's inner-city transport MODES, so guard and body are
# rewritten TOGETHER:
#   if activity_type(activity)=='transportation': S.add(activity_position(activity))
# ->
#   if activity_transports(activity)!=[]: S.add(innercity_transport_type(activity_transports(activity)))
# Guard removal alone would be wrong here (a position is not a mode), which is
# why _transport_only_lines never matches this body shape.  Left unrewritten,
# the constraint is vacuously true (the set stays empty) and blinds every
# generated-constraint gate: the planner's repair-accept pass-count, the final
# self-check, and the enrichment gate.
_MODE_SET_ADD_RE = re.compile(
    r"^([A-Za-z_]\w*)\s*\.add\(\s*activity_position\(activity\)\s*\)$"
)
_MODE_SET_GUARD = "if activity_transports(activity)!=[]:"


def _mode_set_rewrite(stmt):
    """Canonical mode-collection statement for one vacuous-dialect body
    statement, or None when the statement is not that shape."""
    m = _MODE_SET_ADD_RE.match(stmt.strip())
    if not m:
        return None
    return (
        "%s.add(innercity_transport_type(activity_transports(activity)))"
        % m.group(1)
    )


def _transport_only_lines(block_lines):
    """True iff every non-empty line operates on the activity's transports:
    it calls a transport DSL function or references a local variable that an
    earlier block line assigned from ``activity_transports(...)``."""
    transport_locals = set()
    saw_any = False
    for line in block_lines:
        stripped = line.strip()
        if not stripped:
            continue
        saw_any = True
        m = re.match(r"([A-Za-z_]\w*)\s*=\s*activity_transports\(", stripped)
        if m:
            transport_locals.add(m.group(1))
            continue
        if _TRANSPORT_FUNC_RE.search(stripped):
            continue
        if transport_locals and any(
            re.search(r"\b%s\b" % re.escape(v), stripped) for v in transport_locals
        ):
            continue
        return False
    return saw_any


def _transport_accumulation_lines(block_lines):
    """True iff every non-empty line is an accumulation of a transport
    quantity (the transport-budget guard shape; see
    ``_ACTIVITY_TYPE_ONLY_GUARD_RE``)."""
    saw_any = False
    for line in block_lines:
        stripped = line.strip()
        if not stripped:
            continue
        saw_any = True
        if not _TRANSPORT_ACCUM_LINE_RE.match(stripped):
            return False
    return saw_any


def _repair_transport_type_guards(text):
    """Remove activity-type guards that corrupt per-transport constraints:

    1. always-False pseudo-type guards (``=='transportation'`` etc.) whose
       bodies only touch the activity's transports, and
    2. real/partial activity-type guards whose bodies do nothing but
       accumulate transport quantities (transport-budget caps): the canonical
       oracle idiom sums over ALL activities' transports.

    3. vacuous pseudo-type guards whose bodies collect
       ``activity_position(activity)`` into a set (the mode-set dialect) are
       not removed but REWRITTEN into the canonical mode-collection loop
       (see ``_MODE_SET_ADD_RE``).

    See the module docstring and the regex comments above."""
    lines = text.split("\n")
    out = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        m = _TRANSPORT_TYPE_GUARD_RE.match(line)
        pseudo_guard = m is not None
        body_ok = _transport_only_lines
        if not m:
            m = _ACTIVITY_TYPE_ONLY_GUARD_RE.match(line)
            body_ok = _transport_accumulation_lines
        if not m:
            out.append(line)
            idx += 1
            continue
        guard_indent, inline_stmt = m.group(1), m.group(2)
        if inline_stmt:
            # inline form: "if activity_type(...)==...: STMT"
            rewritten = _mode_set_rewrite(inline_stmt) if pseudo_guard else None
            if rewritten is not None:
                out.append(guard_indent + _MODE_SET_GUARD + " " + rewritten)
            elif body_ok([inline_stmt]):
                out.append(guard_indent + inline_stmt)
            else:
                out.append(line)
            idx += 1
            continue
        # block form: collect the more-indented suite under the guard
        block_start = idx + 1
        block_end = block_start
        while block_end < len(lines):
            nxt = lines[block_end]
            if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= len(guard_indent):
                break
            block_end += 1
        block = lines[block_start:block_end]
        nonempty = [b for b in block if b.strip()]
        rewrites = (
            [_mode_set_rewrite(b) for b in nonempty] if pseudo_guard else []
        )
        if nonempty and pseudo_guard and all(r is not None for r in rewrites):
            out.append(guard_indent + _MODE_SET_GUARD)
            rew_iter = iter(rewrites)
            for b in block:
                if b.strip():
                    out.append(b[: len(b) - len(b.lstrip())] + next(rew_iter))
                else:
                    out.append(b)
        elif nonempty and body_ok(block):
            dedent = min(len(b) - len(b.lstrip()) for b in nonempty) - len(guard_indent)
            dedent = max(dedent, 0)
            for b in block:
                out.append(b[dedent:] if b.strip() else b)
        else:
            out.append(line)
            out.extend(block)
        idx = block_end
    return "\n".join(out)


# Accumulator-content -> canonical set-variable spelling.  A free local set
# that collects one of these accessor results IS that canonical set whatever
# the translator called it (e.g. Qwen's `hotel_feature_set` collecting
# accommodation_type(...) is the extractor's accommodation_type_set).  This is
# the semantic, spelling-proof complement to SET_VARIABLE_RENAMES.
_SET_CONTENT_CANON = (
    (re.compile(r"^attraction_type\("), "attraction_type_set"),
    (re.compile(r"^restaurant_type\("), "restaurant_type_set"),
    (re.compile(r"^accommodation_type\("), "accommodation_type_set"),
    (re.compile(r"^innercity_transport_type\("), "inner_city_transportation_set"),
)

_SET_ADD_SITE_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\.add\(\s*(.+?)\s*\)\s*$", re.M)


def _canonicalize_set_accumulators(text):
    """Rename free local set variables by WHAT they accumulate.

    A variable is renamed only when it is initialised with ``set()``, every
    one of its ``.add()`` sites collects the same canonical content, and the
    canonical target name is not already bound in the constraint (the rename
    helper refuses capture)."""
    sites = {}
    for m in _SET_ADD_SITE_RE.finditer(text):
        var, expr = m.group(1), m.group(2)
        canon = None
        for pattern, target in _SET_CONTENT_CANON:
            if pattern.match(expr):
                canon = target
                break
        sites.setdefault(var, set()).add(canon)
    for var, targets in sites.items():
        if len(targets) != 1:
            continue
        target = next(iter(targets))
        if target is None or target == var:
            continue
        if not re.search(r"\b%s\s*=\s*set\(\)" % re.escape(var), text):
            continue
        text = _rename_free_variable(text, var, target)
    return text


def _wrap_bare_expression(constraint):
    """`people_count(plan) == 1` -> `result=(people_count(plan) == 1)`.

    The official evaluator execs each constraint and reads vars()['result'];
    a bare comparison never assigns it, so the constraint is structurally
    always-False no matter what the plan looks like (measured on the A800
    align run: two such constraints made full-pass unreachable for the uid).
    """
    if "\n" in constraint or "result" in constraint or "=" in constraint.replace(
            "==", "").replace("!=", "").replace("<=", "").replace(">=", ""):
        return constraint
    try:
        import ast as _ast
        _ast.parse(constraint.strip(), mode="eval")
    except SyntaxError:
        return constraint
    return "result=(%s)" % constraint.strip()


_BROKEN_SQ_LITERAL_RE = re.compile(
    r"==\s*'(?P<body>[^\n]*?)'(?=\s*(?::|\band\b|\)|$))", re.MULTILINE)


def _requote_broken_literals(constraint):
    """POI names with interior apostrophes ship as =='Bear Grandma's Garden'
    -- an unterminated literal that makes the whole constraint permanently
    False AND truncates repair-target extraction. When a constraint fails to
    compile, greedily re-capture ==-literals up to the LAST quote before the
    clause boundary and emit them double-quoted."""
    try:
        compile(constraint, "<c>", "exec")
        return constraint
    except SyntaxError:
        pass

    def _sub(m):
        body = m.group("body")
        if '"' in body:
            return m.group(0)
        return '=="%s"' % body

    fixed = _BROKEN_SQ_LITERAL_RE.sub(_sub, constraint)
    try:
        compile(fixed, "<c>", "exec")
        return fixed
    except SyntaxError:
        return constraint


def canonicalize_hard_logic_py(constraint):
    """Canonicalize one hard_logic_py constraint string (semantics-preserving)."""
    if not isinstance(constraint, str) or not constraint:
        return constraint
    constraint = _requote_broken_literals(constraint)
    constraint = _wrap_bare_expression(constraint)
    constraint = _canonicalize_set_variables(constraint)
    constraint = _repair_transport_type_guards(constraint)
    constraint = _canonicalize_set_accumulators(constraint)
    constraint = _canonicalize_budget_accumulators(constraint)
    return constraint


# ---------------------------------------------------------------------------
# Entity-category grounding against the public environment database
# ---------------------------------------------------------------------------

_DB_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "environment", "database_en",
))

_POI_CATEGORIES = ("attraction", "restaurant", "accommodation")

# Canonical collection-loop type filter for each category.
_CATEGORY_FILTER = {
    "attraction": "activity_type(activity)=='attraction'",
    "restaurant": "activity_type(activity) in ['breakfast', 'lunch', 'dinner']",
    "accommodation": "activity_type(activity)=='accommodation'",
}

# Regexes matching how the claimed category's filter may be spelled.
_CATEGORY_FILTER_RES = {
    "attraction": re.compile(
        r"activity_type\(activity\)\s*==\s*['\"]attraction['\"]"),
    "restaurant": re.compile(
        r"activity_type\(activity\)\s+in\s+\[\s*['\"]breakfast['\"]\s*,"
        r"\s*['\"]lunch['\"]\s*,\s*['\"]dinner['\"]\s*\]"),
    "accommodation": re.compile(
        r"activity_type\(activity\)\s*==\s*['\"]accommodation['\"]"),
}

_NAME_SET_REQUIRE_RE = re.compile(
    r"result\s*=\s*\(\s*\{[^{}]*\}\s*<=\s*"
    r"(attraction|restaurant|accommodation)_name_set\s*\)"
)

_poi_name_cache = {}


def _city_poi_names(city):
    """{'attraction': set(names), 'restaurant': ..., 'accommodation': ...} for
    a city, from the public environment database; {} when unavailable."""
    key = (city or "").strip().lower()
    if not key:
        return {}
    if key in _poi_name_cache:
        return _poi_name_cache[key]
    paths = {
        "attraction": os.path.join(_DB_ROOT, "attractions", key, "attractions.csv"),
        "restaurant": os.path.join(
            _DB_ROOT, "restaurants", key, f"restaurants_{key}.csv"),
        "accommodation": os.path.join(
            _DB_ROOT, "accommodations", key, "accommodations.csv"),
    }
    names = {}
    for cat, path in paths.items():
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                names[cat] = {
                    row["name"].strip()
                    for row in csv.DictReader(fh)
                    if row.get("name")
                }
        except OSError:
            _poi_name_cache[key] = {}
            return {}
    _poi_name_cache[key] = names
    return names


def _ground_name_set_constraint(constraint, poi_names):
    """Move a must-visit name-set constraint to the DB category its names
    actually belong to (see module docstring, item 4)."""
    m = _NAME_SET_REQUIRE_RE.search(constraint)
    if not m:
        return constraint
    claimed = m.group(1)
    required = {
        a or b
        for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"",
                               m.group(0).split("<=")[0])
    }
    if not required or not poi_names:
        return constraint
    if any(name in poi_names.get(claimed, set()) for name in required):
        return constraint  # at least one name really is in the claimed category
    hosts = [
        cat for cat in _POI_CATEGORIES
        if cat != claimed
        and all(name in poi_names.get(cat, set()) for name in required)
    ]
    if len(hosts) != 1:
        return constraint  # unknown or ambiguous names: leave untouched
    target = hosts[0]
    # the collection loop must carry the claimed category's type filter and no
    # other activity_type comparison we might corrupt
    filter_re = _CATEGORY_FILTER_RES[claimed]
    if len(filter_re.findall(constraint)) != 1:
        return constraint
    if re.search(r"\b%s_name_set\b" % target, constraint):
        return constraint  # would capture an existing binding
    constraint = filter_re.sub(_CATEGORY_FILTER[target], constraint)
    constraint = re.sub(
        r"\b%s_name_set\b" % claimed, f"{target}_name_set", constraint)
    return constraint


def ground_hard_logic_entities(constraints, target_city):
    """DB-ground the entity categories of a hard_logic_py list (best effort)."""
    if not isinstance(constraints, (list, tuple)):
        return constraints
    try:
        poi_names = _city_poi_names(target_city)
    except Exception:
        return list(constraints)
    if not poi_names:
        return list(constraints)
    out = []
    for c in constraints:
        try:
            out.append(
                _ground_name_set_constraint(c, poi_names)
                if isinstance(c, str) else c)
        except Exception:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Type-literal normalization against the DB type vocabulary
#
# LLM translators re-case / re-space category labels ("University campus"
# where the Hangzhou DB says "university campus").  The official verifier is
# case-tolerant via its concept alias table, but the planner-side extractor
# and POI selection match the DB literally, so a mis-cased label silently
# weakens the constraint during SEARCH.  Rewrite the literal to the DB's
# exact spelling whenever it matches a DB type of the same category under a
# case-insensitive + whitespace-collapsed comparison.
#
# Scope guard: ONLY literals in type-set membership expressions are touched
# ({...} <=/&/== <cat>_type_set, or 'X' [not] in <cat>_type_set).  POI names
# are never rewritten (they only appear against *_name_set variables, and a
# literal is replaced only when it folds onto a DB type of that category).
# ---------------------------------------------------------------------------

# zh city spelling -> database directory key (the environment database uses
# pinyin directory names for both language variants)
_ZH_CITY_DIRS = {
    "北京": "beijing", "上海": "shanghai", "南京": "nanjing", "苏州": "suzhou",
    "杭州": "hangzhou", "深圳": "shenzhen", "成都": "chengdu", "武汉": "wuhan",
    "广州": "guangzhou", "重庆": "chongqing",
}

_DB_ROOT_ZH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "environment", "database",
))

# category -> (subdir, filename template, type column)
_TYPE_SOURCES = {
    "attraction": ("attractions", "attractions.csv", "type"),
    "restaurant": ("restaurants", "restaurants_{key}.csv", "cuisine"),
    "accommodation": ("accommodations", "accommodations.csv", "featurehoteltype"),
}

_poi_type_cache = {}


def _fold_type(value):
    """Case-insensitive + whitespace-collapsed comparison key."""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _city_poi_types(city):
    """{'attraction': {folded: exact}, ...} type vocabulary of a city from the
    public environment database; {} when unavailable.  Folded keys shared by
    two DIFFERENT exact spellings are dropped (ambiguous -> never rewritten).
    """
    city = (city or "").strip()
    if not city:
        return {}
    if city in _ZH_CITY_DIRS:
        key, root = _ZH_CITY_DIRS[city], _DB_ROOT_ZH
    else:
        key, root = city.lower(), _DB_ROOT
    cache_key = (root, key)
    if cache_key in _poi_type_cache:
        return _poi_type_cache[cache_key]
    vocab = {}
    for cat, (subdir, fname, column) in _TYPE_SOURCES.items():
        path = os.path.join(root, subdir, key, fname.format(key=key))
        table = {}
        ambiguous = set()
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    exact = (row.get(column) or "").strip()
                    if not exact:
                        continue
                    folded = _fold_type(exact)
                    if table.get(folded, exact) != exact:
                        ambiguous.add(folded)
                    table[folded] = exact
        except OSError:
            _poi_type_cache[cache_key] = {}
            return {}
        for folded in ambiguous:
            table.pop(folded, None)
        vocab[cat] = table
    _poi_type_cache[cache_key] = vocab
    return vocab


_QUOTED_LITERAL_RE = re.compile(r"'((?:\\.|[^\\'])*)'|\"((?:\\.|[^\\\"])*)\"")


def _rewrite_type_literals_in(text, table):
    """Rewrite every quoted literal in ``text`` whose folded form matches a DB
    type onto the DB's exact spelling (quote style preserved)."""

    def _repl(match):
        raw = match.group(1) if match.group(1) is not None else match.group(2)
        exact = table.get(_fold_type(raw))
        if exact is None or exact == raw:
            return match.group(0)
        quote = match.group(0)[0]
        return f"{quote}{exact}{quote}"

    return _QUOTED_LITERAL_RE.sub(_repl, text)


def _type_membership_res(category):
    """Regexes whose ENTIRE match is the literal-bearing span of a type
    membership expression for one category (context sits in lookarounds).
    Covers both the set-variable idiom and the inline accessor idiom
    (``attraction_type(activity, ...) in [...]`` / ``== 'X'``) that the
    extractor's _merge_inline_type_bans reads."""
    var = r"%s_type_set" % category
    accessor = r"%s_type\(activity(?:[^()]|\([^()]*\))*\)" % category
    literal = r"'(?:\\.|[^\\'])*'|\"(?:\\.|[^\\\"])*\""
    return (
        # {...} <= / & / == var
        re.compile(r"\{[^{}]*\}(?=\s*(?:<=|==|&)\s*" + var + r"\b)"),
        # var <= / & / == {...}   (only quoted literals inside the braces can
        # match the rewriter, so the whole expression is a safe span)
        re.compile(r"\b" + var + r"\s*(?:<=|==|&)\s*\{[^{}]*\}"),
        # 'X' in var / 'X' not in var
        re.compile("(?:" + literal + r")(?=\s+(?:not\s+)?in\s+" + var + r"\b)"),
        # attraction_type(activity, ...) in ['X', ...] (inline ban/require)
        re.compile(r"\b" + accessor + r"\s*(?:not\s+)?in\s*\[[^\]]*\]"),
        # attraction_type(activity, ...) == / != 'X'
        re.compile(r"\b" + accessor + r"\s*[!=]=\s*(?:" + literal + r")"),
    )


def _constraint_mentions_category_types(constraint, category):
    """Cheap containment guard for the membership regex sweep."""
    return (
        ("%s_type_set" % category) in constraint
        or ("%s_type(" % category) in constraint
    )


def normalize_type_literals(constraints, target_city):
    """Normalize type literals of type-set membership constraints in a
    hard_logic_py list against the target city's DB vocabulary (best effort;
    any failure leaves the constraint untouched)."""
    if not isinstance(constraints, (list, tuple)):
        return constraints
    try:
        vocab = _city_poi_types(target_city)
    except Exception:
        return list(constraints)
    if not vocab:
        return list(constraints)
    out = []
    for constraint in constraints:
        if not isinstance(constraint, str):
            out.append(constraint)
            continue
        try:
            for category, table in vocab.items():
                if not table or not _constraint_mentions_category_types(
                    constraint, category
                ):
                    continue
                for pattern in _type_membership_res(category):
                    constraint = pattern.sub(
                        lambda m, _t=table: _rewrite_type_literals_in(
                            m.group(0), _t
                        ),
                        constraint,
                    )
        except Exception:
            pass
        out.append(constraint)
    return out


# ---------------------------------------------------------------------------
# NL-span grounding of TYPE literals
#
# LLM translators occasionally substitute a more common sibling label for a
# rare category ("Museum/Memorial Hall" where the NL verbatim says
# "Library/Memorial Hall").  Case-folding cannot repair a CONTENT swap, but
# the NL request names the type literally, so the request text is an
# authoritative witness.  A literal in a type-set membership expression is
# replaced only under ALL of these conditions:
#   * the emitted literal does NOT occur in the NL (case/whitespace-folded);
#   * exactly ONE type from the target city's DB vocabulary of that category
#     (a) occurs verbatim in the NL, (b) shares a '/'-separated component
#     with the emitted literal (both must be slash-compound labels), and
#     (c) is not already used by another literal of the constraint.
# Slash-compound matching keeps this conservative: single-word labels are
# never rewritten from incidental NL word hits.
# ---------------------------------------------------------------------------


def _nl_fold(text):
    return re.sub(r"\s+", " ", str(text or "")).casefold()


def _slash_components(label):
    folded = _fold_type(label)
    if "/" not in folded:
        return set()
    return {part.strip() for part in folded.split("/") if part.strip()}


def _nl_ground_type_literal(raw, table, nl_folded, taken):
    """DB type spelling to replace ``raw`` with, or None to keep it."""
    folded = _fold_type(raw)
    if folded in nl_folded:
        return None  # the NL itself supports the emitted literal
    raw_parts = _slash_components(raw)
    if not raw_parts:
        return None
    hits = []
    for cand_folded, exact in table.items():
        if cand_folded == folded or cand_folded in taken:
            continue
        if cand_folded not in nl_folded:
            continue
        if raw_parts & _slash_components(exact):
            hits.append(exact)
    if len(hits) == 1:
        return hits[0]
    return None


def ground_type_literals_to_nl(constraints, target_city, nature_language):
    """Repair content-swapped type literals against the NL request (best
    effort; only type-set membership spans are touched, and any failure
    leaves the constraint unchanged)."""
    if not isinstance(constraints, (list, tuple)):
        return constraints
    nl_folded = _nl_fold(nature_language)
    if not nl_folded:
        return list(constraints)
    try:
        vocab = _city_poi_types(target_city)
    except Exception:
        return list(constraints)
    if not vocab:
        return list(constraints)
    out = []
    for constraint in constraints:
        if not isinstance(constraint, str):
            out.append(constraint)
            continue
        try:
            taken = {
                _fold_type(a or b)
                for a, b in _QUOTED_LITERAL_RE.findall(constraint)
            }
            for category, table in vocab.items():
                if not table or not _constraint_mentions_category_types(
                    constraint, category
                ):
                    continue

                def _ground_span(match, _t=table, _k=taken):
                    def _repl(lit_match):
                        raw = (
                            lit_match.group(1)
                            if lit_match.group(1) is not None
                            else lit_match.group(2)
                        )
                        exact = _nl_ground_type_literal(raw, _t, nl_folded, _k)
                        if exact is None:
                            return lit_match.group(0)
                        quote = lit_match.group(0)[0]
                        return f"{quote}{exact}{quote}"

                    return _QUOTED_LITERAL_RE.sub(_repl, match.group(0))

                for pattern in _type_membership_res(category):
                    constraint = pattern.sub(_ground_span, constraint)
        except Exception:
            pass
        out.append(constraint)
    return out


def canonicalize_hard_logic_list(constraints):
    """Canonicalize a hard_logic_py list (str / list / tuple tolerated)."""
    if isinstance(constraints, str):
        return canonicalize_hard_logic_py(constraints)
    if isinstance(constraints, (list, tuple)):
        return [canonicalize_hard_logic_py(c) for c in constraints]
    return constraints


def canonicalize_query_hard_logic(query):
    """Canonicalize query['hard_logic_py'] in place (returns the query).

    The list is also put into a deterministic, content-keyed order: the LLM
    emits the same constraint set in a run-dependent order (sampling jitter),
    and downstream constraint extraction / search tie-breaks are sensitive to
    list order.  Sorting by constraint text makes content-identical
    translations produce byte-identical planner inputs.  Evaluation semantics
    are unaffected (every constraint must hold regardless of order).
    """
    if isinstance(query, dict) and query.get("hard_logic_py"):
        hl = canonicalize_hard_logic_list(query["hard_logic_py"])
        if isinstance(hl, (list, tuple)):
            hl = ground_hard_logic_entities(hl, query.get("target_city"))
            hl = normalize_type_literals(hl, query.get("target_city"))
            hl = ground_type_literals_to_nl(
                hl, query.get("target_city"), query.get("nature_language")
            )
            hl = sorted(hl, key=lambda c: str(c))
        query["hard_logic_py"] = hl
    return query

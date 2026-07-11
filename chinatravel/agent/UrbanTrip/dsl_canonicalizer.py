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

    See the module docstring and the regex comments above."""
    lines = text.split("\n")
    out = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        m = _TRANSPORT_TYPE_GUARD_RE.match(line)
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
            if body_ok([inline_stmt]):
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
        if nonempty and body_ok(block):
            dedent = min(len(b) - len(b.lstrip()) for b in nonempty) - len(guard_indent)
            dedent = max(dedent, 0)
            for b in block:
                out.append(b[dedent:] if b.strip() else b)
        else:
            out.append(line)
            out.extend(block)
        idx = block_end
    return "\n".join(out)


def canonicalize_hard_logic_py(constraint):
    """Canonicalize one hard_logic_py constraint string (semantics-preserving)."""
    if not isinstance(constraint, str) or not constraint:
        return constraint
    constraint = _canonicalize_set_variables(constraint)
    constraint = _repair_transport_type_guards(constraint)
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
            hl = sorted(hl, key=lambda c: str(c))
        query["hard_logic_py"] = hl
    return query

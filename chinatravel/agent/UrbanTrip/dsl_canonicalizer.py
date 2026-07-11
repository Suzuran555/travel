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

IMPORTANT: ``activity_price`` and ``activity_cost`` are DIFFERENT DSL functions
(``activity_price(a)`` reads the per-person/room ``price`` field, ``activity_cost(a)``
reads the total ``cost`` field -- see chinatravel/symbol_verification/concept_func.py).
They are therefore NEVER rewritten into each other here; instead the extractor in
tpc_agent_optimized_v6.py was widened to understand the ``activity_price`` idioms.

Renames are guarded: a rename is skipped whenever the canonical target name already
occurs in the constraint (so an existing binding can never be captured).
"""

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


def canonicalize_hard_logic_py(constraint):
    """Canonicalize one hard_logic_py constraint string (semantics-preserving)."""
    if not isinstance(constraint, str) or not constraint:
        return constraint
    constraint = _canonicalize_set_variables(constraint)
    constraint = _canonicalize_budget_accumulators(constraint)
    return constraint


def canonicalize_hard_logic_list(constraints):
    """Canonicalize a hard_logic_py list (str / list / tuple tolerated)."""
    if isinstance(constraints, str):
        return canonicalize_hard_logic_py(constraints)
    if isinstance(constraints, (list, tuple)):
        return [canonicalize_hard_logic_py(c) for c in constraints]
    return constraints


def canonicalize_query_hard_logic(query):
    """Canonicalize query['hard_logic_py'] in place (returns the query)."""
    if isinstance(query, dict) and query.get("hard_logic_py"):
        query["hard_logic_py"] = canonicalize_hard_logic_list(query["hard_logic_py"])
    return query

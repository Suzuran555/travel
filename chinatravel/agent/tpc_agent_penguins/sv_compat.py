"""Backward-compat shim: normalize_hard_logic_constraint.

Upstream stock chinatravel.symbol_verification.hard_constraint does not ship
this helper; vendored verbatim from our branch.
"""
import re

# --- Backward-compat shim (planner needs normalize_hard_logic_constraint) ---
_POI_DISTANCE_ACCOMMODATION_RE = re.compile(
    r"(poi_distance\(target_city\(plan\)\s*,\s*)(['\"])(.+)\2(\s*,\s*accommodation_position\))"
)


def _normalize_poi_distance_literals(constraint):
    def replace_match(match):
        poi_name = match.group(3).replace("\\'", "'").replace('\\"', '"')
        return f"{match.group(1)}{poi_name!r}{match.group(4)}"

    lines = []
    for line in constraint.splitlines():
        if "poi_distance" in line and "accommodation_position" in line:
            line = _POI_DISTANCE_ACCOMMODATION_RE.sub(replace_match, line)
        lines.append(line)
    return "\n".join(lines)


def _replace_comparison_literal(line, pattern):
    match = re.search(pattern, line)
    if not match:
        return line
    quote = match.group("quote")
    content_start = match.end()
    closing = None
    for idx in range(content_start, len(line)):
        if line[idx] != quote:
            continue
        if idx > content_start and line[idx - 1] == "\\":
            continue
        rest = line[idx + 1 :].lstrip()
        if not rest or rest[0] in ":)]},&|":
            closing = idx
    if closing is None:
        return line
    raw_value = line[content_start:closing]
    value = raw_value.replace("\\'", "'").replace('\\"', '"')
    return f"{line[:match.start('quote')]}{value!r}{line[closing + 1:]}"


def _normalize_activity_position_literals(constraint):
    pattern = (
        r"activity_position\(activity\)\s*(?:==|!=)\s*"
        r"(?P<quote>['\"])"
    )
    return "\n".join(
        _replace_comparison_literal(line, pattern)
        if "activity_position(activity)" in line
        else line
        for line in constraint.splitlines()
    )


def normalize_hard_logic_constraint(constraint):
    constraint = _normalize_poi_distance_literals(constraint)
    constraint = _normalize_activity_position_literals(constraint)
    return constraint

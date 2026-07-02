"""Extract must-visit POIs from query DSL and audit intracity segment coverage.

Shared by scripts/build_urbantrip_segments.py (must-aware segment building) and
scripts/audit_segment_must_coverage.py (coverage report). Extraction mirrors the
DSL parsing in tpc_agent_optimized_v5.py (_extract_set_constraints / extract_list).
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _unescape_literal_text(text):
    return text.replace("\\'", "'").replace('\\"', '"')


def extract_list(s):
    """Parse a comma-separated set literal body into a list of names."""
    items = []
    idx = 0
    while idx < len(s):
        while idx < len(s) and s[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(s):
            break
        quote = s[idx] if s[idx] in "'\"" else None
        if quote is None:
            end = s.find(",", idx)
            if end == -1:
                end = len(s)
            value = s[idx:end].strip()
            if value:
                items.append(value)
            idx = end + 1
            continue
        content_start = idx + 1
        closing = None
        scan = content_start
        while scan < len(s):
            if s[scan] == quote and s[scan - 1] != "\\":
                rest = s[scan + 1 :].lstrip()
                if not rest or rest[0] in ",}])":
                    closing = scan
                    break
            scan += 1
        if closing is None:
            break
        items.append(_unescape_literal_text(s[content_start:closing]))
        idx = closing + 1
    return items


def _extract_set_constraints(dsl_str, var_name):
    patterns = [
        rf"result\s*=\s*\(\s*\{{([^}}]*)\}}\s*(?:&|<=)\s*{var_name}",
        rf"result\s*=\s*\(\s*{var_name}\s*(?:&|<=)\s*\{{([^}}]*)\}}",
    ]
    values = []
    for pat in patterns:
        for match in re.finditer(pat, dsl_str):
            values.extend(extract_list(match.group(1)))
    return values


def extract_must_pois_from_query(query):
    """Return (attraction_names, restaurant_names) sets from a query dict."""
    attractions = set()
    restaurants = set()
    dsl_parts = query.get("hard_logic_py") or []
    if isinstance(dsl_parts, str):
        dsl_parts = [dsl_parts]
    for dsl in dsl_parts:
        if not isinstance(dsl, str):
            continue
        for name in _extract_set_constraints(dsl, "attraction_name_set"):
            attractions.add(name)
        for name in _extract_set_constraints(dsl, "restaurant_name_set"):
            restaurants.add(name)
    return attractions, restaurants


def read_ids(split_file):
    path = split_file
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_query_json(query_id, lang="en"):
    data_dir = os.path.join(PROJECT_ROOT, "chinatravel", "data")
    if lang == "en":
        data_dir = os.path.join(data_dir, "en")
    if not os.path.isdir(data_dir):
        return None
    for dir_name in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        path = os.path.join(dir_path, f"{query_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def load_must_names_by_city(split_file, lang="en"):
    """Map target_city -> {"attraction": set(), "restaurant": set()} from a split."""
    must = defaultdict(lambda: {"attraction": set(), "restaurant": set()})
    for query_id in read_ids(split_file):
        query = load_query_json(query_id, lang=lang)
        if not query:
            continue
        city = query.get("target_city")
        if not city:
            continue
        attrs, rests = extract_must_pois_from_query(query)
        must[city]["attraction"].update(attrs)
        must[city]["restaurant"].update(rests)
    return dict(must)


def count_segment_endpoints(segment_path, poi_names):
    """Count how many intracity segment rows touch each POI name as start/end."""
    counts = {name: {"as_start": 0, "as_end": 0} for name in poi_names}
    if not os.path.exists(segment_path):
        return counts
    with open(segment_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            start = row.get("start")
            end = row.get("end")
            if start in counts:
                counts[start]["as_start"] += 1
            if end in counts:
                counts[end]["as_end"] += 1
    return counts


def audit_must_coverage(segment_path, must_by_city):
    """Summarise must POI coverage in an intracity segment file."""
    summary = {
        "segment_path": os.path.abspath(segment_path),
        "cities": {},
        "totals": {
            "must_attractions": 0,
            "must_restaurants": 0,
            "covered_attractions": 0,
            "covered_restaurants": 0,
        },
    }
    all_names = set()
    for groups in must_by_city.values():
        all_names.update(groups.get("attraction", set()))
        all_names.update(groups.get("restaurant", set()))
    counts = count_segment_endpoints(segment_path, all_names)

    for city, groups in sorted(must_by_city.items()):
        city_report = {"attraction": [], "restaurant": []}
        for kind in ("attraction", "restaurant"):
            for name in sorted(groups.get(kind, set())):
                c = counts.get(name, {"as_start": 0, "as_end": 0})
                covered = (c["as_start"] + c["as_end"]) > 0
                city_report[kind].append(
                    {
                        "name": name,
                        "as_start": c["as_start"],
                        "as_end": c["as_end"],
                        "covered": covered,
                    }
                )
                total_key = (
                    "must_attractions" if kind == "attraction" else "must_restaurants"
                )
                covered_key = (
                    "covered_attractions"
                    if kind == "attraction"
                    else "covered_restaurants"
                )
                summary["totals"][total_key] += 1
                if covered:
                    summary["totals"][covered_key] += 1
        summary["cities"][city] = city_report
    return summary

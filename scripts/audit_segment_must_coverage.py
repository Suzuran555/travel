#!/usr/bin/env python3
"""Report intracity segment coverage for must-visit POIs in a query split."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chinatravel.agent.UrbanTrip.segment_must_pois import (
    audit_must_coverage,
    load_must_names_by_city,
)
from chinatravel.environment.language import normalize_lang


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit must POI coverage in the intracity segment index."
    )
    parser.add_argument(
        "--split",
        default="chinatravel/evaluation/default_splits/TPC_IJCAI_2026_phase1.txt",
    )
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument(
        "--segment-path",
        default=None,
        help="Path to intracity_segments.jsonl (default: environment/segments/<lang>/)",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--only-uncovered",
        action="store_true",
        help="Print only POIs with zero coverage",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lang = normalize_lang(args.lang)
    segment_path = args.segment_path or os.path.join(
        PROJECT_ROOT,
        "chinatravel",
        "environment",
        "segments",
        lang,
        "intracity_segments.jsonl",
    )
    must_by_city = load_must_names_by_city(args.split, lang=lang)
    report = audit_must_coverage(segment_path, must_by_city)

    totals = report["totals"]
    print("=== must POI segment coverage ===")
    print(f"segment: {report['segment_path']}")
    print(
        "attractions: {}/{} covered | restaurants: {}/{} covered".format(
            totals["covered_attractions"],
            totals["must_attractions"],
            totals["covered_restaurants"],
            totals["must_restaurants"],
        )
    )
    if args.only_uncovered:
        for city, groups in report["cities"].items():
            uncovered = [
                item["name"]
                for kind in ("attraction", "restaurant")
                for item in groups[kind]
                if not item["covered"]
            ]
            if uncovered:
                print(f"[{city}] uncovered: {uncovered}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

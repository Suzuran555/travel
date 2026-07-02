#!/usr/bin/env python3
import argparse
import json
import os
import sys

import pandas as pd
from geopy.distance import geodesic

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chinatravel.agent.UrbanTrip.segment_must_pois import load_must_names_by_city
from chinatravel.environment.language import CITY_NAMES, CITY_SLUGS, normalize_lang
from chinatravel.environment.tools.accommodations.apis import Accommodations
from chinatravel.environment.tools.attractions.apis import Attractions
from chinatravel.environment.tools.intercity_transport.apis import IntercityTransport
from chinatravel.environment.tools.restaurants.apis import Restaurants
from chinatravel.environment.tools.transportation.apis import Transportation


def time_to_minutes(time_str):
    hour, minute = str(time_str).split(":")[:2]
    return int(hour) * 60 + int(minute)


def duration_minutes(start_time, end_time):
    start = time_to_minutes(start_time)
    end = time_to_minutes(end_time)
    if end < start:
        end += 24 * 60
    return end - start


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def route_to_template(route):
    template = []
    for leg in route:
        template.append(
            {
                "start": leg.get("start"),
                "end": leg.get("end"),
                "mode": leg.get("mode"),
                "cost": safe_float(leg.get("cost")),
                "distance": safe_float(leg.get("distance")),
                "duration": duration_minutes(leg.get("start_time"), leg.get("end_time")),
            }
        )
    return template


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_intercity_segments(lang):
    intercity = IntercityTransport(lang=lang)
    cities = CITY_NAMES[lang]
    segments = []
    terminal_map = {city: set() for city in cities}

    for source in cities:
        for target in cities:
            if source == target:
                continue
            for mode in ("train", "airplane"):
                try:
                    data = intercity.select(source, target, mode)
                except TypeError:
                    data = None
                if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                    continue
                for _, row in data.iterrows():
                    segment_id = row.get("FlightID") if mode == "airplane" else row.get("TrainID")
                    from_station = row.get("From")
                    to_station = row.get("To")
                    terminal_map[source].add(from_station)
                    terminal_map[target].add(to_station)
                    duration = duration_minutes(row.get("BeginTime"), row.get("EndTime"))
                    cost = safe_float(row.get("Cost"))
                    base_score = cost + duration * 0.8
                    segments.append(
                        {
                            "from_city": source,
                            "to_city": target,
                            "mode": mode,
                            "id": segment_id,
                            "from": from_station,
                            "to": to_station,
                            "begin": row.get("BeginTime"),
                            "end": row.get("EndTime"),
                            "cost": cost,
                            "duration": duration,
                            "base_score": base_score,
                            "score_text": " ".join(
                                [
                                    str(source),
                                    str(target),
                                    str(mode),
                                    str(segment_id),
                                    str(from_station),
                                    str(to_station),
                                    "cheap" if cost <= data["Cost"].median() else "standard",
                                    "early" if time_to_minutes(row.get("BeginTime")) < 12 * 60 else "late",
                                ]
                            ),
                        }
                    )
    return segments, terminal_map


def point(name, point_type, lat, lon, tags=None, price=0.0):
    return {
        "name": str(name),
        "type": point_type,
        "lat": safe_float(lat),
        "lon": safe_float(lon),
        "tags": tags or [],
        "price": safe_float(price),
    }


def collect_city_points(
    city, terminal_names, accommodations, attractions, restaurants, top_k, must_names=None
):
    poi_lookup = accommodations.poi
    points = []
    must_names = must_names or {"attraction": set(), "restaurant": set()}

    for terminal in sorted(terminal_names):
        coord = poi_lookup.search(city, terminal)
        if not isinstance(coord, str):
            lat, lon = coord
            points.append(point(terminal, "terminal", lat, lon, ["terminal", "station", "airport"]))

    hotels = accommodations.data[city].sort_values(by=["price", "name"]).head(max(10, top_k))
    for _, row in hotels.iterrows():
        points.append(
            point(
                row["name"],
                "hotel",
                row["lat"],
                row["lon"],
                ["hotel", str(row.get("featurehoteltype", ""))],
                row.get("price"),
            )
        )

    attr = attractions.data[city].sort_values(by=["price", "name"]).head(max(30, top_k))
    for _, row in attr.iterrows():
        tags = ["attraction", str(row.get("type", ""))]
        if safe_float(row.get("price")) == 0:
            tags.append("free")
        points.append(point(row["name"], "attraction", row["lat"], row["lon"], tags, row.get("price")))

    rest = restaurants.data[city].sort_values(by=["price", "name"]).head(max(30, top_k))
    for _, row in rest.iterrows():
        points.append(
            point(
                row["name"],
                "restaurant",
                row["lat"],
                row["lon"],
                ["restaurant", str(row.get("cuisine", ""))],
                row.get("price"),
            )
        )

    # Must-aware: union every must-visit POI referenced by the split, even when
    # it falls outside the price-sorted head above. Tagged "must" so that forced
    # edges can be identified and so score_text carries the marker.
    _union_must_points(points, city, attractions.data.get(city), must_names.get("attraction"), "attraction")
    _union_must_points(points, city, restaurants.data.get(city), must_names.get("restaurant"), "restaurant")

    dedup = {}
    for item in points:
        # Merge tags if a must POI was already added via the price head.
        if item["name"] in dedup:
            existing = dedup[item["name"]]
            existing["tags"] = sorted(set(existing["tags"]) | set(item["tags"]))
        else:
            dedup[item["name"]] = item
    return list(dedup.values())


def _union_must_points(points, city, table, names, point_type):
    if table is None or not names:
        return
    type_col = "type" if point_type == "attraction" else "cuisine"
    for name in names:
        row = table[table["name"] == name]
        if row.empty:
            continue
        row = row.iloc[0]
        tags = [point_type, str(row.get(type_col, "")), "must"]
        if safe_float(row.get("price")) == 0:
            tags.append("free")
        points.append(
            point(name, point_type, row["lat"], row["lon"], tags, row.get("price"))
        )


def nearest_pairs(points, top_k):
    pairs = set()
    by_name = {item["name"]: item for item in points}

    terminals = [item for item in points if item["type"] == "terminal"]
    hotels = [item for item in points if item["type"] == "hotel"]
    pois = [item for item in points if item["type"] in {"attraction", "restaurant"}]
    must_points = [item for item in points if "must" in item["tags"]]

    for terminal in terminals:
        for hotel in hotels:
            pairs.add((terminal["name"], hotel["name"]))

    for hotel in hotels:
        scored = sorted(
            (
                (
                    geodesic((hotel["lat"], hotel["lon"]), (poi["lat"], poi["lon"])).kilometers,
                    poi["name"],
                )
                for poi in pois
            ),
            key=lambda item: (item[0], item[1]),
        )
        for _, poi_name in scored[:top_k]:
            pairs.add((hotel["name"], poi_name))

    for source in pois:
        scored = sorted(
            (
                (
                    geodesic((source["lat"], source["lon"]), (target["lat"], target["lon"])).kilometers,
                    target["name"],
                )
                for target in pois
                if target["name"] != source["name"]
            ),
            key=lambda item: (item[0], item[1]),
        )
        for _, target_name in scored[: min(top_k, 10)]:
            pairs.add((source["name"], target_name))

    # Must-aware forced edges: every must POI must be reachable from terminals
    # and any candidate hotel (full, not nearest-k truncated) so hotel ranking
    # and _geo_route_lb see the must geography. must<->must is capped to nearest
    # neighbours (full same-city pairwise is O(must^2) and dominates build time
    # while adding little: DFS computes must->must routes live, segments here
    # only aid ranking).
    for must in must_points:
        for terminal in terminals:
            pairs.add((terminal["name"], must["name"]))
        for hotel in hotels:
            pairs.add((hotel["name"], must["name"]))
    must_neighbor_cap = min(top_k, 10)
    for source in must_points:
        scored = sorted(
            (
                (
                    geodesic((source["lat"], source["lon"]), (target["lat"], target["lon"])).kilometers,
                    target["name"],
                )
                for target in must_points
                if target["name"] != source["name"]
            ),
            key=lambda item: (item[0], item[1]),
        )
        for _, target_name in scored[:must_neighbor_cap]:
            pairs.add((source["name"], target_name))

    return sorted(pairs), by_name


def build_intracity_segments(lang, terminal_map, top_k_neighbors, must_by_city=None):
    transportation = Transportation(lang=lang)
    accommodations = Accommodations(lang=lang)
    attractions = Attractions(lang=lang)
    restaurants = Restaurants(lang=lang)
    must_by_city = must_by_city or {}
    segments = []

    for city in CITY_NAMES[lang]:
        points = collect_city_points(
            city,
            terminal_map.get(city, set()),
            accommodations,
            attractions,
            restaurants,
            top_k_neighbors,
            must_names=must_by_city.get(city),
        )
        pairs, by_name = nearest_pairs(points, top_k_neighbors)
        for start, end in pairs:
            start_info = by_name[start]
            end_info = by_name[end]
            for mode in ("walk", "metro", "taxi"):
                try:
                    route = transportation.goto(city, start, end, "08:00", mode)
                except Exception:
                    continue
                if not isinstance(route, list) or not route:
                    continue
                template = route_to_template(route)
                total_cost = sum(safe_float(leg["cost"]) for leg in template)
                total_distance = sum(safe_float(leg["distance"]) for leg in template)
                total_duration = sum(safe_float(leg["duration"]) for leg in template)
                base_score = total_cost * 20 + total_duration + total_distance * 2
                tags = start_info["tags"] + end_info["tags"] + [mode]
                if total_cost == 0:
                    tags.append("free")
                if total_distance <= 3:
                    tags.append("near")
                segments.append(
                    {
                        "city": city,
                        "start": start,
                        "end": end,
                        "start_type": start_info["type"],
                        "end_type": end_info["type"],
                        "mode": mode,
                        "route": template,
                        "cost": total_cost,
                        "distance": total_distance,
                        "duration": total_duration,
                        "base_score": base_score,
                        "score_text": " ".join(str(tag) for tag in [city, start, end] + tags),
                    }
                )
    return segments


def parse_args():
    parser = argparse.ArgumentParser(description="Build UrbanTrip segment JSONL indexes.")
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--out", default=None)
    parser.add_argument("--top-k-neighbors", type=int, default=30)
    parser.add_argument(
        "--must-from-split",
        default=None,
        help="Split file whose query DSL must-visit POIs are force-added to the graph.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lang = normalize_lang(args.lang)
    out_dir = args.out or os.path.join(
        PROJECT_ROOT, "chinatravel", "environment", "segments", lang
    )
    must_by_city = None
    if args.must_from_split:
        must_by_city = load_must_names_by_city(args.must_from_split, lang=lang)
        n_must = sum(
            len(groups.get("attraction", set())) + len(groups.get("restaurant", set()))
            for groups in must_by_city.values()
        )
        print(f"loaded {n_must} must POIs across {len(must_by_city)} cities from {args.must_from_split}")
    intercity_segments, terminal_map = build_intercity_segments(lang)
    intracity_segments = build_intracity_segments(
        lang, terminal_map, args.top_k_neighbors, must_by_city=must_by_city
    )
    write_jsonl(os.path.join(out_dir, "intercity_segments.jsonl"), intercity_segments)
    write_jsonl(os.path.join(out_dir, "intracity_segments.jsonl"), intracity_segments)
    print(
        f"wrote {len(intercity_segments)} intercity and {len(intracity_segments)} intracity segments to {out_dir}"
    )


if __name__ == "__main__":
    main()

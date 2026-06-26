import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chinatravel.environment.language import normalize_lang


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _time_to_minutes(time_str):
    hour, minute = str(time_str).split(":")[:2]
    return int(hour) * 60 + int(minute)


def _minutes_to_time(minutes):
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _duration_minutes(start_time, end_time):
    start = _time_to_minutes(start_time)
    end = _time_to_minutes(end_time)
    if end < start:
        end += 24 * 60
    return end - start


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_lower_better(values):
    if not values:
        return []
    clean = [_safe_float(value, 10**6) for value in values]
    finite = [value for value in clean if value < 10**6]
    if not finite:
        return [1.0 for _ in clean]
    low = min(finite)
    high = max(finite)
    if high <= low:
        return [0.0 if value < 10**6 else 1.0 for value in clean]
    return [
        1.0 if value >= 10**6 else max(0.0, min(1.0, (value - low) / (high - low)))
        for value in clean
    ]


def _time_margin_minutes(current_time, end_time):
    if not current_time or not end_time:
        return 10**6
    try:
        current = _time_to_minutes(current_time)
        end = _time_to_minutes(end_time)
    except (TypeError, ValueError):
        return 10**6
    if end < current:
        end += 24 * 60
    return max(0, end - current)


class SegmentIndex:
    def __init__(self, lang="en", segment_dir=None, top_k=50):
        self.lang = normalize_lang(lang)
        self.top_k = top_k
        if segment_dir is None:
            segment_dir = os.path.join(
                PROJECT_ROOT, "chinatravel", "environment", "segments", self.lang
            )
        self.segment_dir = segment_dir
        self.intercity_segments = self._load_jsonl("intercity_segments.jsonl")
        self.intracity_segments = self._load_jsonl("intracity_segments.jsonl")
        self.has_intercity = bool(self.intercity_segments)
        self.has_intracity = bool(self.intracity_segments)

        self._intercity_by_key = defaultdict(list)
        for segment in self.intercity_segments:
            key = (
                segment.get("from_city"),
                segment.get("to_city"),
                segment.get("mode"),
            )
            self._intercity_by_key[key].append(segment)

        self._intracity_by_key = {}
        self._intracity_by_start = defaultdict(list)
        for segment in self.intracity_segments:
            key = (
                segment.get("city"),
                segment.get("start"),
                segment.get("end"),
                segment.get("mode"),
            )
            self._intracity_by_key[key] = segment
            self._intracity_by_start[(segment.get("city"), segment.get("start"))].append(segment)

        self._tfidf_vectorizer = None
        self._intracity_matrix = None
        self._intracity_docs = []
        if self.intracity_segments:
            self._build_tfidf()

    def _load_jsonl(self, filename):
        path = os.path.join(self.segment_dir, filename)
        if not os.path.exists(path):
            return []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _build_tfidf(self):
        docs = [segment.get("score_text", "") for segment in self.intracity_segments]
        if not any(docs):
            return
        try:
            self._tfidf_vectorizer = TfidfVectorizer()
            self._intracity_matrix = self._tfidf_vectorizer.fit_transform(docs)
            self._intracity_docs = docs
        except ValueError:
            self._tfidf_vectorizer = None
            self._intracity_matrix = None
            self._intracity_docs = []

    def _query_text(self, query, extra=None):
        parts = [str(query.get("nature_language", ""))]
        for item in query.get("hard_logic_py", []) or []:
            parts.append(str(item))
        if extra:
            parts.append(str(extra))
        return " ".join(parts)

    def _tfidf_scores(self, query_text, segments):
        if self._tfidf_vectorizer is None or self._intracity_matrix is None:
            return {}
        try:
            query_vec = self._tfidf_vectorizer.transform([query_text])
            scores = cosine_similarity(query_vec, self._intracity_matrix).ravel()
        except ValueError:
            return {}
        wanted_ids = {id(segment) for segment in segments}
        result = {}
        for idx, segment in enumerate(self.intracity_segments):
            if id(segment) in wanted_ids:
                result[id(segment)] = float(scores[idx])
        return result

    def rank_intercity(self, query, direction, df):
        if not self.has_intercity or df is None or len(df) == 0:
            return list(range(len(df))) if df is not None else []

        must = query.get("must_depart_transport") if direction == "go" else query.get("must_return_transport")
        must_not = query.get("must_not_depart_transport") if direction == "go" else query.get("must_not_return_transport")
        must = set(must or [])
        must_not = set(must_not or [])

        scored = []
        for pos, (_, row) in enumerate(df.iterrows()):
            mode = "airplane" if "FlightID" in row and not pd.isna(row.get("FlightID")) else "train"
            if must and mode not in must:
                continue
            if must_not and mode in must_not:
                continue
            segment_score = self._matching_intercity_score(row, mode)
            cost = _safe_float(row.get("Cost"), default=10**9)
            begin = _time_to_minutes(row.get("BeginTime"))
            time_score = begin if direction == "go" else -begin
            scored.append((pos, segment_score, cost, time_score))

        if not scored:
            return list(range(len(df)))
        scored.sort(key=lambda item: (item[1], item[2], item[3]))
        ordered = [idx for idx, _, _, _ in scored]
        remaining = [idx for idx in range(len(df)) if idx not in set(ordered)]
        return ordered + remaining

    def _matching_intercity_score(self, row, mode):
        from_station = row.get("From")
        to_station = row.get("To")
        row_id = row.get("FlightID") if mode == "airplane" else row.get("TrainID")
        candidates = []
        for segment in self.intercity_segments:
            if segment.get("mode") != mode:
                continue
            # Route must match first: the same TrainID/FlightID can appear on
            # multiple legs, so matching by id alone leaks a wrong-route score.
            if segment.get("from") != from_station or segment.get("to") != to_station:
                continue
            if segment.get("id") == row_id:
                return _safe_float(segment.get("base_score"))
            candidates.append(segment)
        if not candidates:
            return 10**9
        return min(_safe_float(segment.get("base_score")) for segment in candidates)

    def rank_hotels(self, query, hotel_df, go_transport, back_transport, candidate_idx):
        if not self.has_intracity or hotel_df is None or not candidate_idx:
            return candidate_idx

        anchors = []
        if go_transport is not None:
            anchors.append(go_transport.get("To"))
        if back_transport is not None:
            anchors.append(back_transport.get("From"))

        city = query.get("target_city")
        scored = []
        for idx in candidate_idx:
            hotel = hotel_df.iloc[idx]
            route_cost = 0.0
            route_distance = 0.0
            route_hits = 0
            for anchor in anchors:
                best = self._best_intracity_segment(city, anchor, hotel.get("name"))
                if best is None:
                    continue
                route_hits += 1
                route_cost += _safe_float(best.get("cost"))
                route_distance += _safe_float(best.get("distance"))
            if route_hits == 0:
                route_cost = 10**6
                route_distance = 10**6
            price = _safe_float(hotel.get("price"))
            scored.append((idx, route_cost, route_distance, price))

        scored.sort(key=lambda item: (item[1], item[2], item[3]))
        return [idx for idx, _, _, _ in scored]

    def rank_poi(self, query, current_position, poi_type, candidate_df, constraints):
        if candidate_df is None or len(candidate_df) == 0:
            return candidate_df
        if not self.has_intracity or not current_position:
            return candidate_df

        city = query.get("target_city")
        query_text = self._query_text(query, extra=f"{current_position} {poi_type}")
        constraints = constraints or {}
        dynamic = constraints.get("dynamic_ranking") or {}
        weights = dynamic.get("weights") or {}
        current_time = dynamic.get("current_time")
        pending = dynamic.get("pending") or {}
        rows = []
        candidate_segments = []
        for pos, (_, row) in enumerate(candidate_df.iterrows()):
            name = row.get("name")
            best_segment = self._best_intracity_segment(city, current_position, name)
            if best_segment is not None:
                candidate_segments.append(best_segment)
            hard_bonus = self._hard_anchor_bonus(name, row, poi_type, constraints)
            route_cost = _safe_float(best_segment.get("cost"), 10**6) if best_segment else 10**6
            route_distance = _safe_float(best_segment.get("distance"), 10**6) if best_segment else 10**6
            base_score = _safe_float(best_segment.get("base_score"), 10**6) if best_segment else 10**6
            price = _safe_float(row.get("price"), 0.0)
            close_margin = _time_margin_minutes(current_time, row.get("endtime"))
            coverage_gain = self._must_coverage_gain(name, row, poi_type, pending)
            order_penalty = self._order_block_penalty(name, pending)
            rows.append(
                {
                    "pos": pos,
                    "name": name,
                    "hard_bonus": hard_bonus,
                    "route_cost": route_cost,
                    "route_distance": route_distance,
                    "base_score": base_score,
                    "price": price,
                    "close_margin": close_margin,
                    "coverage_gain": coverage_gain,
                    "order_penalty": order_penalty,
                    "row": row,
                    "segment": best_segment,
                }
            )

        tfidf = self._tfidf_scores(query_text, candidate_segments)
        route_cost_norm = _normalize_lower_better([row["route_cost"] for row in rows])
        route_distance_norm = _normalize_lower_better([row["route_distance"] for row in rows])
        base_score_norm = _normalize_lower_better([row["base_score"] for row in rows])
        price_norm = _normalize_lower_better([row["price"] for row in rows])
        close_margin_norm = _normalize_lower_better([row["close_margin"] for row in rows])
        max_gain = max([row["coverage_gain"] for row in rows] or [0.0])
        max_penalty = max([row["order_penalty"] for row in rows] or [0.0])

        def weight(name, default):
            return max(0.0, _safe_float(weights.get(name), default))

        scored = []
        for idx, row in enumerate(rows):
            best_segment = row["segment"]
            semantic = tfidf.get(id(best_segment), 0.0) if best_segment else 0.0
            gain_norm = (row["coverage_gain"] / max_gain) if max_gain > 0 else 0.0
            weighted_score = (
                weight("route_cost", 0.45) * route_cost_norm[idx]
                + weight("route_distance", 0.15) * route_distance_norm[idx]
                + weight("base_score", 0.10) * base_score_norm[idx]
                + weight("poi_price", 0.20) * price_norm[idx]
                + weight("time_margin", 0.20) * close_margin_norm[idx]
                - weight("must_coverage", 1.00) * gain_norm
                + weight("order_block", 1.20)
                * ((row["order_penalty"] / max_penalty) if max_penalty > 0 else 0.0)
                - weight("semantic", 0.0) * semantic
            )
            fallback_key = (
                -row["hard_bonus"],
                row["route_cost"],
                row["route_distance"],
                row["base_score"],
                -semantic,
                row["pos"],
            )
            scored.append(((weighted_score, fallback_key), row["pos"]))

        order = [pos for _, pos in sorted(scored, key=lambda item: item[0])]
        return candidate_df.iloc[order].reset_index(drop=True)

    def _must_coverage_gain(self, name, row, poi_type, pending):
        gain = 0.0
        if name in set(pending.get("order_predecessors") or []):
            gain += 750.0
        if poi_type in {"lunch", "dinner", "restaurant"}:
            if name in set(pending.get("restaurant_names") or []):
                gain += 1000.0
            if row.get("cuisine") in set(pending.get("restaurant_types") or []):
                gain += 100.0
        if poi_type == "attraction":
            if name in set(pending.get("attraction_names") or []):
                gain += 1000.0
            if row.get("type") in set(pending.get("attraction_types") or []):
                gain += 100.0
        return gain

    def _order_block_penalty(self, name, pending):
        if name in set(pending.get("order_blocked") or []):
            return 500.0
        return 0.0

    def _hard_anchor_bonus(self, name, row, poi_type, constraints):
        bonus = 0
        constraints = constraints or {}
        if poi_type in {"lunch", "dinner", "restaurant"}:
            if name in set(constraints.get("must_visit_restaurant") or []):
                bonus += 1000
            if row.get("cuisine") in set(constraints.get("must_visit_restaurant_type") or []):
                bonus += 100
        if poi_type == "attraction":
            if name in set(constraints.get("must_see_attraction") or []):
                bonus += 1000
            if row.get("type") in set(constraints.get("must_see_attraction_type") or []):
                bonus += 100
        return bonus

    def _best_intracity_segment(self, city, start, end, mode=None):
        if not start or not end:
            return None
        modes = [mode] if mode else ["walk", "metro", "taxi"]
        candidates = []
        for candidate_mode in modes:
            direct = self._intracity_by_key.get((city, start, end, candidate_mode))
            if direct is not None:
                candidates.append(direct)
            reverse = self._intracity_by_key.get((city, end, start, candidate_mode))
            if reverse is not None:
                candidates.append(self._reverse_route(reverse))
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda segment: (
                _safe_float(segment.get("cost"), 10**6),
                _safe_float(segment.get("duration"), 10**6),
                _safe_float(segment.get("distance"), 10**6),
            ),
        )

    def _reverse_route(self, segment):
        route = []
        for leg in reversed(segment.get("route", [])):
            route.append(
                {
                    **leg,
                    "start": leg.get("end"),
                    "end": leg.get("start"),
                }
            )
        reversed_segment = dict(segment)
        reversed_segment["start"] = segment.get("end")
        reversed_segment["end"] = segment.get("start")
        reversed_segment["route"] = route
        return reversed_segment

    def get_intracity_route(self, city, start, end, mode, start_time, people_number):
        segment = self._best_intracity_segment(city, start, end, mode=mode)
        if segment is None:
            return None

        current_minutes = _time_to_minutes(start_time)
        result = []
        for leg in segment.get("route", []):
            duration = int(round(_safe_float(leg.get("duration"), 0)))
            leg_start = _minutes_to_time(current_minutes)
            current_minutes += duration
            leg_end = _minutes_to_time(current_minutes)
            mode_name = leg.get("mode")
            price = _safe_float(leg.get("cost"))
            item = {
                "start": leg.get("start"),
                "end": leg.get("end"),
                "mode": mode_name,
                "start_time": leg_start,
                "end_time": leg_end,
                "cost": price,
                "distance": _safe_float(leg.get("distance")),
                "price": price,
            }
            if mode_name == "metro":
                item["tickets"] = people_number
                item["cost"] = price * people_number
            elif mode_name == "taxi":
                item["cars"] = int((people_number - 1) / 4) + 1
                item["cost"] = price * item["cars"]
            result.append(item)
        return result

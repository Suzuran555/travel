"""Shared transport / constraint helpers for dfs_poi."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from chinatravel.agent.UrbanTrip.search_state import VisitingSnapshot, ensure_day_plan
from chinatravel.agent.UrbanTrip.segment_index import SegmentIndex
from chinatravel.agent.UrbanTrip.utils import (
    add_time_delta,
    time_compare_if_earlier_equal,
)


def dfs_log(agent, *args, **kwargs) -> None:
    if getattr(agent, "debug", False):
        print(*args, **kwargs)


def flatten_visiting_indices(visiting_list) -> List[Any]:
    indices: List[Any] = []
    for item in visiting_list:
        if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
            try:
                indices.extend(list(item))
            except TypeError:
                indices.append(item)
        else:
            indices.append(item)
    return indices


def transports_ranking_to(agent, query, start: str, end: str) -> List[str]:
    ranking = list(agent.innercity_transports_ranking)
    if agent.transport_rules_by_distance is not None:
        distance = agent.calculate_distance(query, start, end)
        ranking = agent.get_transport_by_distance(distance)
    return ranking


def arrived_time(current_time: str, transports_sel: List[Dict[str, Any]]) -> str:
    if not transports_sel:
        return current_time
    return transports_sel[-1]["end_time"]


def transport_rules_violated(agent, transports_sel: List[Dict[str, Any]]) -> bool:
    if agent.transport_rules_by_distance is None:
        return False
    distance = 0.0
    for transport in transports_sel:
        if transport.get("mode") is not None:
            distance += transport.get("distance", 0)
    mode = None
    if len(transports_sel) == 3:
        mode = transports_sel[1]["mode"]
    elif len(transports_sel) == 1:
        mode = transports_sel[0]["mode"]
    if mode is None:
        return True
    for rule in agent.transport_rules_by_distance:
        if rule.get("min_distance") is not None:
            if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                return True
        if rule.get("max_distance") is not None:
            if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                return True
    return False


def arrive_leave_violated(
    agent, poi_name: str, act_start_time: str, act_end_time: str
) -> bool:
    if agent.too_many_backtrack:
        return False
    if agent.activities_arrive_time_dict is not None:
        arrive_info = agent.activities_arrive_time_dict.get(poi_name)
        if arrive_info:
            arrive_type, arrive_time = arrive_info
            if arrive_type == "early" and not time_compare_if_earlier_equal(
                act_start_time, arrive_time
            ):
                return True
            if arrive_type == "late" and not time_compare_if_earlier_equal(
                arrive_time, act_start_time
            ):
                return True
    if agent.activities_leave_time_dict is not None:
        leave_info = agent.activities_leave_time_dict.get(poi_name)
        if leave_info:
            leave_type, leave_time = leave_info
            if leave_type == "early" and not time_compare_if_earlier_equal(
                act_end_time, leave_time
            ):
                return True
            if leave_type == "late" and not time_compare_if_earlier_equal(
                leave_time, act_end_time
            ):
                return True
    return False


def is_closed_at_arrival(
    opentime: str, endtime: str, arrived_time: str, *, allow_overnight: bool
) -> bool:
    if not time_compare_if_earlier_equal(endtime, arrived_time):
        return False
    if allow_overnight and time_compare_if_earlier_equal(endtime, opentime):
        return False
    return True


def rank_poi_dataframe(
    agent,
    query,
    current_position: str,
    filtered_df: pd.DataFrame,
    poi_kind: str,
    *,
    use_attraction_budget_key: bool = False,
) -> pd.DataFrame:
    # enable_transit_time_score (ATT lever): POI ranking goes through
    # SegmentIndex.rank_poi (whose weighted score carries the transit_time
    # term) even when use_segments=False, via a ranking-only index that leaves
    # agent.segment_index None for every other consumer. With the flag off,
    # _get_transit_rank_segment_index returns None and behavior is unchanged.
    segment_index = agent.segment_index
    if segment_index is None and getattr(agent, "enable_transit_time_score", False):
        segment_index = agent._get_transit_rank_segment_index()

    if segment_index is None and getattr(
        agent, "enable_segment_independent_must_rank", False
    ):
        return rank_poi_dataframe_must_aware(
            agent,
            query,
            current_position,
            filtered_df,
            poi_kind,
            use_attraction_budget_key=use_attraction_budget_key,
        )

    if use_attraction_budget_key:
        budget_sort = agent.overall_budget is not None or agent.attraction_budget is not None
    else:
        budget_sort = agent.overall_budget is not None or agent.attraction_budget is not None
    if budget_sort:
        ranked = filtered_df.sort_values(by="price").reset_index(drop=True)
    else:
        work = filtered_df.copy()
        work["distance"] = work.apply(
            lambda row: agent.calculate_distance(query, current_position, row["name"]),
            axis=1,
        )
        ranked = work.sort_values(by="distance").reset_index(drop=True)
    if segment_index is not None:
        ranked = segment_index.rank_poi(
            agent._segment_query(query),
            current_position,
            poi_kind,
            ranked,
            agent._segment_constraints(),
            # Sub-flag gated on the parent flag so enable_transit_time_score
            # off stays bit-identical regardless of the sub-flag value.
            transit_signal_min_duration=bool(
                getattr(agent, "enable_transit_time_score", False)
                and getattr(agent, "transit_signal_min_duration", False)
            ),
        )
    return ranked


def rank_poi_dataframe_must_aware(
    agent,
    query,
    current_position: str,
    filtered_df: pd.DataFrame,
    poi_kind: str,
    *,
    use_attraction_budget_key: bool = False,
) -> pd.DataFrame:
    """Segment-independent must-coverage promotion.

    When ``use_segments=False``, ``agent.segment_index`` is None and the plain
    distance/price sort in ``rank_poi_dataframe`` has no notion of pending
    must-visit names/types or order-blocking -- those live only inside
    ``SegmentIndex.rank_poi``'s weighted score, so disabling segments silently
    drops must-visit coverage from candidate ordering.

    Rather than reproduce ``SegmentIndex.rank_poi``'s full weighted rescore
    (which also folds in price/time-margin terms and, on a first attempt,
    regressed a budget-tight must-restaurant case that the plain price sort
    handled fine), this keeps the *exact* base ordering ``rank_poi_dataframe``
    would otherwise produce (price sort under a budget, distance sort
    otherwise) and only does a stable promotion: pending must-visit
    names/types move to the front (in their original relative order),
    order-blocked candidates move to the back. Everything else keeps its
    original price/distance order untouched.
    """
    if filtered_df is None or len(filtered_df) == 0:
        return filtered_df

    if use_attraction_budget_key:
        budget_sort = agent.overall_budget is not None or agent.attraction_budget is not None
    else:
        budget_sort = agent.overall_budget is not None or agent.attraction_budget is not None
    if budget_sort:
        base = filtered_df.sort_values(by="price").reset_index(drop=True)
    else:
        work = filtered_df.copy()
        work["distance"] = work.apply(
            lambda row: agent.calculate_distance(query, current_position, row["name"]),
            axis=1,
        )
        base = work.sort_values(by="distance").reset_index(drop=True)

    constraints = dict(agent._segment_constraints())
    dynamic = constraints.get("dynamic_ranking") or {}
    pending = dict(dynamic.get("pending") or {})

    # Loose OR-type constraints ("any one of these N types satisfies the
    # whole trip") show every not-yet-seen type as pending at *every* meal
    # slot until one is hit. Treating that like a must-cover-every-type
    # constraint promotes a same-type match even when it's far away -- a
    # greedy detour that can burn enough time budget to cascade into a full
    # skeleton collapse (observed: a 2-day trip with "any of 3 cuisine
    # types" lost all attractions/meals once type promotion applied). Only
    # promote by type when every listed type actually needs covering; exact
    # name matches (real must-visit POIs) always keep their promotion.
    if poi_kind in {"lunch", "dinner", "restaurant"} and getattr(
        agent, "must_visit_restaurant_type_match_any", False
    ):
        constraints["must_visit_restaurant_type"] = []
        pending["restaurant_types"] = []
    if poi_kind == "attraction" and getattr(
        agent, "must_see_attraction_type_match_any", False
    ):
        constraints["must_see_attraction_type"] = []
        pending["attraction_types"] = []

    sort_keys = []
    for _, row in base.iterrows():
        name = row.get("name")
        blocked = 1 if SegmentIndex._order_block_penalty(name, pending) > 0 else 0
        bonus = SegmentIndex._hard_anchor_bonus(
            name, row, poi_kind, constraints
        ) + SegmentIndex._must_coverage_gain(name, row, poi_kind, pending)
        sort_keys.append((blocked, -bonus))

    order = sorted(range(len(base)), key=lambda pos: sort_keys[pos])
    return base.iloc[order].reset_index(drop=True)


def filter_open_at_time(df: pd.DataFrame, current_time: str) -> pd.DataFrame:
    return df[
        df.apply(
            lambda row: (
                time_compare_if_earlier_equal(row["opentime"], current_time)
                and time_compare_if_earlier_equal(current_time, row["endtime"])
            ),
            axis=1,
        )
    ].copy()


def iterate_transports(
    agent,
    query,
    current_time: str,
    current_position: str,
    destination: str,
    on_success: Callable[[List[Dict[str, Any]], str], Optional[Tuple[bool, Any]]],
) -> Tuple[bool, Any]:
    """
    Try each inner-city transport mode to reach destination.
    on_success(transports_sel, arrived_time) -> None to continue, or (bool, payload) to stop.
    """
    for trans_type in transports_ranking_to(agent, query, current_position, destination):
        agent.search_nodes += 1
        if current_position == destination:
            transports_sel: List[Dict[str, Any]] = []
            arrival = current_time
        else:
            transports_sel = agent.collect_innercity_transport(
                query["target_city"],
                current_position,
                destination,
                current_time,
                trans_type,
            )
            if not isinstance(transports_sel, list):
                agent.backtrack_count += 1
                dfs_log(agent, "inner-city transport error, backtrack...")
                continue
            arrival = arrived_time(current_time, transports_sel)
        if transport_rules_violated(agent, transports_sel):
            if not agent.too_many_backtrack:
                agent.backtrack_count += 1
            continue
        result = on_success(transports_sel, arrival)
        if result is not None:
            return result
    return False, None

"""Incremental commonsense helpers for itinerary chains (space + time)."""

from .utils import (
    add_time_delta,
    clamp_time_to_day_end,
    sanitize_transport_times,
    time_compare_if_earlier_equal,
    time_to_minutes,
)
from chinatravel.symbol_verification.commonsense_constraint import (
    Is_attractions_correct,
    Is_restaurants_correct,
    Is_space_correct,
    Is_time_correct,
)


def _time_minutes(time_str):
    if not time_str:
        return 0
    time_str = str(time_str).split("次日")[-1]
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def is_intercity_activity(activity):
    if not isinstance(activity, dict):
        return False
    if activity.get("type") in ("train", "airplane"):
        return True
    return "TrainID" in activity or "FlightID" in activity


def activity_position(activity):
    if not isinstance(activity, dict):
        return ""
    if activity.get("position"):
        return activity["position"]
    if is_intercity_activity(activity):
        return activity.get("end", "")
    return ""


def needs_transport(prev_pos, curr_pos):
    if not prev_pos or not curr_pos:
        return False
    return prev_pos != curr_pos


_POI_MEMORY_KEY = {
    "attraction": "attractions",
    "lunch": "restaurants",
    "dinner": "restaurants",
}


def activity_open_hours(agent, act):
    if not isinstance(act, dict):
        return None, None
    memory_key = _POI_MEMORY_KEY.get(act.get("type"))
    if not memory_key:
        return None, None
    position = act.get("position")
    if not position:
        return None, None
    memory = getattr(agent, "memory", None) or {}
    df = memory.get(memory_key)
    if df is None or len(df) == 0:
        return None, None
    matches = df[df["name"] == position]
    if matches.empty:
        return None, None
    row = matches.iloc[0]
    return row.get("opentime"), row.get("endtime")


def _apply_meal_time_rules(act, opentime, endtime):
    poi_type = act.get("type")
    if not act.get("start_time"):
        return
    st = act["start_time"]
    if poi_type == "lunch":
        if opentime and time_compare_if_earlier_equal(st, opentime):
            st = opentime
        if time_compare_if_earlier_equal(st, "11:00"):
            st = "11:00"
        act["start_time"] = st
    elif poi_type == "dinner":
        if opentime and time_compare_if_earlier_equal(st, opentime):
            st = opentime
        if time_compare_if_earlier_equal(st, "17:00"):
            st = "17:00"
        act["start_time"] = st


def clamp_activity_start_to_open_hours(act, opentime, endtime):
    """Keep activity start/end inside POI open hours when transport arrives early."""
    if not act.get("start_time"):
        return
    st = act["start_time"]
    if opentime and time_compare_if_earlier_equal(st, opentime):
        act["start_time"] = opentime
        st = opentime
    _apply_meal_time_rules(act, opentime, endtime)
    st = act.get("start_time")
    et = act.get("end_time")
    if (
        et
        and endtime
        and time_compare_if_earlier_equal(endtime, et)
        and not time_compare_if_earlier_equal(endtime, opentime)
    ):
        act["end_time"] = endtime
        et = endtime
    if et and time_compare_if_earlier_equal(et, st):
        act["end_time"] = add_time_delta(st, 60)


def apply_activity_time_rules(agent, act):
    opentime, endtime = activity_open_hours(agent, act)
    if opentime:
        clamp_activity_start_to_open_hours(act, opentime, endtime)
    elif act.get("type") in {"lunch", "dinner"}:
        _apply_meal_time_rules(act, None, None)


def _transports_connect(transports, prev_pos, curr_pos):
    if not transports:
        return prev_pos == curr_pos
    return (
        transports[0].get("start") == prev_pos
        and transports[-1].get("end") == curr_pos
    )


def iter_activity_refs(itinerary):
    for day_idx, day in enumerate(itinerary):
        for act_idx, act in enumerate(day.get("activities", [])):
            yield day_idx, act_idx, act


def build_position_chain(itinerary):
    positions = []
    for _, _, act in iter_activity_refs(itinerary):
        pos = activity_position(act)
        if pos:
            positions.append(pos)
    return positions


def sanitize_itinerary_times(itinerary):
    """Fix activity clocks that rolled past 24:00; do not rewrite transport legs (eval checks durations)."""
    for day in itinerary:
        for act in day.get("activities", []):
            if is_intercity_activity(act):
                continue
            st = act.get("start_time")
            if not st:
                continue
            st_min = time_to_minutes(str(st).split("次日")[-1])
            day_end = time_to_minutes("24:00")
            if st_min >= day_end:
                if act.get("type") == "accommodation":
                    act["start_time"] = "23:00"
                else:
                    act["start_time"] = clamp_time_to_day_end(st)
            et = act.get("end_time")
            if et and act.get("type") != "accommodation":
                act["end_time"] = clamp_time_to_day_end(et)


def forward_time_chain(itinerary):
    """Align transport.end -> activity.start and ensure positive activity duration."""
    sanitize_itinerary_times(itinerary)
    for day in itinerary:
        for act in day.get("activities", []):
            if is_intercity_activity(act):
                continue
            transports = act.get("transports") or []
            st = act.get("start_time")
            et = act.get("end_time")
            if transports:
                te = transports[-1].get("end_time")
                if te and (not st or time_to_minutes(te) > time_to_minutes(st)):
                    act["start_time"] = te
                    st = te
            if not st or not et:
                continue
            st_min = time_to_minutes(st)
            et_min = time_to_minutes(et)
            if st_min < et_min:
                continue
            if act.get("type") == "accommodation":
                if st_min >= time_to_minutes("24:00"):
                    act["start_time"] = "23:00"
                elif st_min >= et_min:
                    act["start_time"] = "23:00"
                act["end_time"] = "24:00"
            else:
                act["end_time"] = add_time_delta(st, 60)


def _plan_shell(query, itinerary):
    return {
        "people_number": query["people_number"],
        "start_city": query["start_city"],
        "target_city": query["target_city"],
        "itinerary": itinerary,
    }


def incremental_space_time_ok(query, itinerary):
    plan_json = _plan_shell(query, itinerary)
    space_stats, _ = Is_space_correct(query, plan_json, verbose=False)
    time_stats, _ = Is_time_correct(query, plan_json, verbose=False)
    return space_stats.loc[0].sum() == 0 and time_stats.loc[0].sum() == 0


def incremental_poi_hours_ok(query, itinerary, lang=None):
    from chinatravel.symbol_verification.commonsense_constraint import (
        _infer_lang,
        _set_tool_lang,
    )

    _set_tool_lang(lang or _infer_lang(query))
    plan_json = _plan_shell(query, itinerary)
    attr_stats, _ = Is_attractions_correct(query, plan_json, verbose=False)
    rest_stats, _ = Is_restaurants_correct(query, plan_json, verbose=False)
    return attr_stats.loc[0].sum() == 0 and rest_stats.loc[0].sum() == 0


def incremental_commonsense_ok(query, itinerary, lang=None):
    return incremental_space_time_ok(query, itinerary) and incremental_poi_hours_ok(
        query, itinerary, lang=lang
    )


def _prev_activity(itinerary, day_idx, act_idx):
    if act_idx > 0:
        return itinerary[day_idx]["activities"][act_idx - 1]
    for d in range(day_idx - 1, -1, -1):
        acts = itinerary[d].get("activities", [])
        if acts:
            return acts[-1]
    return None


def repair_activity_edge(agent, query, itinerary, day_idx, act_idx):
    """Ensure transports exist and connect prev position when required."""
    acts = itinerary[day_idx].get("activities", [])
    if act_idx < 0 or act_idx >= len(acts):
        return True
    act = acts[act_idx]
    if is_intercity_activity(act):
        return True

    curr_pos = activity_position(act)
    prev_act = _prev_activity(itinerary, day_idx, act_idx)
    if prev_act is None:
        act["transports"] = act.get("transports") or []
        return True

    prev_pos = activity_position(prev_act)
    arrive_info = (getattr(agent, "activities_arrive_time_dict", None) or {}).get(
        curr_pos
    )
    arrive_deadline = None
    if arrive_info and arrive_info[0] == "early":
        arrive_deadline = arrive_info[1]
    if not needs_transport(prev_pos, curr_pos):
        act["transports"] = []
        if act.get("position"):
            act["position"] = curr_pos or act["position"]
        return True

    existing = act.get("transports") or []
    existing_arrival = existing[-1].get("end_time") if existing else None
    existing_meets_deadline = (
        arrive_deadline is None
        or (
            existing_arrival
            and time_compare_if_earlier_equal(existing_arrival, arrive_deadline)
        )
    )
    if _transports_connect(existing, prev_pos, curr_pos) and existing_meets_deadline:
        apply_activity_time_rules(agent, act)
        forward_time_chain(itinerary)
        apply_activity_time_rules(agent, act)
        return True

    start_time = prev_act.get("end_time") or prev_act.get("start_time") or "08:00"
    city = query["target_city"]
    ranking = getattr(agent, "innercity_transports_ranking", ["metro", "taxi", "walk"])
    segment_hint = None
    if getattr(agent, "segment_index", None) is not None:
        segment_hint = agent.segment_index._best_intracity_segment(city, prev_pos, curr_pos)
    modes = []
    if segment_hint and segment_hint.get("mode"):
        modes.append(segment_hint["mode"])
    for mode in ranking:
        if mode not in modes:
            modes.append(mode)

    fallback = None
    for mode in modes:
        transports = agent.collect_innercity_transport(
            city, prev_pos, curr_pos, start_time, mode
        )
        if isinstance(transports, list):
            if fallback is None:
                fallback = transports
            arrival = transports[-1].get("end_time") if transports else start_time
            if arrive_deadline is None or time_compare_if_earlier_equal(
                arrival, arrive_deadline
            ):
                fallback = transports
                break
    if fallback is not None:
        act["transports"] = fallback
        if fallback:
            act["start_time"] = fallback[-1]["end_time"]
        apply_activity_time_rules(agent, act)
        forward_time_chain(itinerary)
        apply_activity_time_rules(agent, act)
        return True
    return False


def sync_itinerary_commonsense(agent, query, itinerary, day_idx):
    """Repair latest activity on day_idx and validate incremental space/time."""
    acts = itinerary[day_idx].get("activities", [])
    if not acts:
        return True
    act_idx = len(acts) - 1
    if not repair_activity_edge(agent, query, itinerary, day_idx, act_idx):
        return False
    forward_time_chain(itinerary)
    for _, _, act in iter_activity_refs(itinerary):
        apply_activity_time_rules(agent, act)
    lang = getattr(agent, "lang", None)
    return incremental_commonsense_ok(query, itinerary, lang=lang)


def repair_full_itinerary(agent, query, itinerary):
    sanitize_itinerary_times(itinerary)
    for day_idx, day in enumerate(itinerary):
        for act_idx in range(len(day.get("activities", []))):
            repair_activity_edge(agent, query, itinerary, day_idx, act_idx)
    forward_time_chain(itinerary)
    for _, _, act in iter_activity_refs(itinerary):
        apply_activity_time_rules(agent, act)
    lang = getattr(agent, "lang", None)
    return incremental_commonsense_ok(query, itinerary, lang=lang)

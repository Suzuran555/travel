#!/usr/bin/env python3
"""Read-only headroom analysis for the DAV/ATT/DDR preference scores.

These three components of ``Overall Score`` (see eval_tpc.py: DEFAULT_ATTRACTION_PR,
DEFAULT_TRANS_PR, DEFAULT_RES_PR) are fixed templates evaluated on top of an
*already-passing* plan:

    DAV = total_attractions / (4 * day_count)                 -> target: total_attractions >= 4*days
    DDR = total_meals       / (3 * day_count)                  -> target: total_meals       >= 3*days
    ATT = -avg_transport_minutes/105 + 8/7                     -> target: avg_transport_minutes <= 15

This script does NOT touch the agent. It only estimates, for each already
generated plan, how much of the DAV/DDR gap looks *recoverable* by adding
extra (non-must) meals/attractions into idle daytime that the current plan
already leaves unused -- vs. how much of the gap looks *structural* (arrival /
departure day has no spare time no matter what).

"Idle time" per day is estimated as: the gap between consecutive activities
(and before the first / after the last) minus the transport time already
spent getting there. On day 1 the day-start bound is clamped to the inbound
intercity transport's arrival time (if any); on the last day the day-end
bound is clamped to the outbound intercity transport's boarding time (if
any), minus its own transit time. This is a coarse, upper-bound estimate:
it only checks *time* feasibility, not whether a matching POI (right type,
open hours, budget, geography) actually exists in the candidate pool.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

INTERCITY_TYPES = {"train", "airplane"}
MEAL_TYPES = {"breakfast", "lunch", "dinner"}
DAY_START_DEFAULT = 7 * 60
DAY_END_DEFAULT = 22 * 60
ATTRACTION_SLOT_MIN = 90
MEAL_SLOT_MIN = 30
MEAL_WINDOWS = {
    "breakfast": (6 * 60, 9 * 60),
    "lunch": (11 * 60, 14 * 60),
    "dinner": (17 * 60, 20 * 60),
}


def time_to_min(t):
    t = str(t).split("次日")[-1]
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def transport_duration(act):
    total = 0
    for leg in act.get("transports") or []:
        total += max(0, time_to_min(leg["end_time"]) - time_to_min(leg["start_time"]))
    return total


def overlap_minutes(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def analyze_day(day, day_idx, n_days):
    acts = sorted(
        day.get("activities", []),
        key=lambda a: time_to_min(a.get("start_time", "00:00")),
    )
    intercity = [a for a in acts if a.get("type") in INTERCITY_TYPES]
    day_start, day_end = DAY_START_DEFAULT, DAY_END_DEFAULT
    if day_idx == 0 and intercity:
        day_start = max(day_start, time_to_min(intercity[0]["end_time"]))
    tail_transit = 0
    if day_idx == n_days - 1 and intercity:
        day_end = min(day_end, time_to_min(intercity[-1]["start_time"]))
        tail_transit = transport_duration(intercity[-1])

    cursor = day_start
    idle_segments = []  # (start, end) free windows, for meal-window overlap checks
    meal_count = 0
    attr_count = 0
    for act in acts:
        if act.get("type") in INTERCITY_TYPES:
            continue
        start = time_to_min(act["start_time"])
        end = time_to_min(act["end_time"])
        tdur = transport_duration(act)
        gap = start - cursor - tdur
        if gap > 0:
            idle_segments.append((cursor, cursor + gap))
        cursor = max(cursor, end)
        if act.get("type") in MEAL_TYPES:
            meal_count += 1
        if act.get("type") == "attraction":
            attr_count += 1

    tail_gap = day_end - cursor - tail_transit
    if tail_gap > 0:
        idle_segments.append((cursor, cursor + tail_gap))

    total_idle = sum(e - s for s, e in idle_segments)
    present_meals = {
        act.get("type") for act in acts if act.get("type") in MEAL_TYPES
    }
    missing_meals = MEAL_TYPES - present_meals
    recoverable_meals = []
    for meal in missing_meals:
        w_start, w_end = MEAL_WINDOWS[meal]
        best = max(
            (overlap_minutes(s, e, w_start, w_end) for s, e in idle_segments),
            default=0,
        )
        if best >= MEAL_SLOT_MIN:
            recoverable_meals.append(meal)

    recoverable_attraction_slots = int(total_idle // ATTRACTION_SLOT_MIN)

    return {
        "idle_minutes": total_idle,
        "meal_count": meal_count,
        "attr_count": attr_count,
        "missing_meals": sorted(missing_meals),
        "recoverable_meals": recoverable_meals,
        "recoverable_attraction_slots": recoverable_attraction_slots,
    }


def analyze_plan(plan):
    itin = plan.get("itinerary")
    if not isinstance(itin, list) or not itin:
        return None
    n_days = len(itin)
    days_out = []
    for i, day in enumerate(itin):
        days_out.append(analyze_day(day, i, n_days))

    total_attr = sum(d["attr_count"] for d in days_out)
    total_meals = sum(d["meal_count"] for d in days_out)
    attr_deficit = max(0, 4 * n_days - total_attr)
    meal_deficit = max(0, 3 * n_days - total_meals)

    recoverable_attr = sum(d["recoverable_attraction_slots"] for d in days_out)
    recoverable_meal_slots = sum(len(d["recoverable_meals"]) for d in days_out)

    return {
        "n_days": n_days,
        "total_attr": total_attr,
        "total_meals": total_meals,
        "attr_deficit": attr_deficit,
        "meal_deficit": meal_deficit,
        "recoverable_attr": min(attr_deficit, recoverable_attr),
        "recoverable_meal_slots": min(meal_deficit, recoverable_meal_slots),
        "days": days_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        default="results/UrbanTripOptimizedV5_TPCLLM_en_oracletranslation",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dump_examples", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    if args.limit:
        files = files[: args.limit]

    per_plan = []
    by_position_idle = {"first": [], "mid": [], "last": []}
    by_position_missing_meal_recoverable = {"first": 0, "mid": 0, "last": 0}
    by_position_missing_meal_total = {"first": 0, "mid": 0, "last": 0}

    for f in files:
        try:
            plan = json.load(open(f))
        except Exception:
            continue
        res = analyze_plan(plan)
        if res is None:
            continue
        res["uid"] = os.path.splitext(os.path.basename(f))[0]
        per_plan.append(res)
        n = res["n_days"]
        for i, d in enumerate(res["days"]):
            pos = "first" if i == 0 else ("last" if i == n - 1 else "mid")
            by_position_idle[pos].append(d["idle_minutes"])
            by_position_missing_meal_total[pos] += len(d["missing_meals"])
            by_position_missing_meal_recoverable[pos] += len(d["recoverable_meals"])

    n_plans = len(per_plan)
    total_attr_deficit = sum(p["attr_deficit"] for p in per_plan)
    total_meal_deficit = sum(p["meal_deficit"] for p in per_plan)
    total_recoverable_attr = sum(p["recoverable_attr"] for p in per_plan)
    total_recoverable_meal = sum(p["recoverable_meal_slots"] for p in per_plan)

    print(f"plans analyzed: {n_plans}")
    print()
    print("== Global deficits (sum across all plans) ==")
    print(f"attraction deficit (need to reach 4/day): {total_attr_deficit}")
    print(f"  time-recoverable (idle >= {ATTRACTION_SLOT_MIN}min slots exist): {total_recoverable_attr} "
          f"({100.0*total_recoverable_attr/max(1,total_attr_deficit):.1f}% of deficit)")
    print(f"meal deficit (need to reach 3/day): {total_meal_deficit}")
    print(f"  time-recoverable (idle overlaps meal window >= {MEAL_SLOT_MIN}min): {total_recoverable_meal} "
          f"({100.0*total_recoverable_meal/max(1,total_meal_deficit):.1f}% of deficit)")
    print()

    print("== Idle minutes/day by day position ==")
    for pos in ("first", "mid", "last"):
        arr = np.array(by_position_idle[pos]) if by_position_idle[pos] else np.array([0])
        print(f"{pos:5s}: n={len(arr):4d} mean={arr.mean():6.1f}min median={np.median(arr):6.1f}min "
              f"p75={np.percentile(arr,75):6.1f}min max={arr.max():6.1f}min")
    print()

    print("== Missing-meal recoverability by day position ==")
    for pos in ("first", "mid", "last"):
        tot = by_position_missing_meal_total[pos]
        rec = by_position_missing_meal_recoverable[pos]
        print(f"{pos:5s}: missing_meal_slots={tot:4d} recoverable={rec:4d} "
              f"({100.0*rec/max(1,tot):.1f}%)")
    print()

    # Projected DAV/DDR if all time-recoverable slots were actually filled.
    days_total = sum(p["n_days"] for p in per_plan)
    proj_attr = sum(min(4 * p["n_days"], p["total_attr"] + p["recoverable_attr"]) for p in per_plan)
    proj_meals = sum(min(3 * p["n_days"], p["total_meals"] + p["recoverable_meal_slots"]) for p in per_plan)
    cur_attr = sum(p["total_attr"] for p in per_plan)
    cur_meals = sum(p["total_meals"] for p in per_plan)
    print("== Projected score components if all recoverable slots are filled ==")
    print(f"DAV: current_ratio={cur_attr/(4*days_total):.3f} -> projected_ratio={proj_attr/(4*days_total):.3f} "
          f"(clamped scores: {min(1,cur_attr/(4*days_total))*100:.1f} -> {min(1,proj_attr/(4*days_total))*100:.1f})")
    print(f"DDR: current_ratio={cur_meals/(3*days_total):.3f} -> projected_ratio={proj_meals/(3*days_total):.3f} "
          f"(clamped scores: {min(1,cur_meals/(3*days_total))*100:.1f} -> {min(1,proj_meals/(3*days_total))*100:.1f})")

    if args.dump_examples:
        print()
        print(f"== {args.dump_examples} example plans with largest recoverable headroom ==")
        ranked = sorted(
            per_plan,
            key=lambda p: p["recoverable_attr"] + p["recoverable_meal_slots"],
            reverse=True,
        )
        for p in ranked[: args.dump_examples]:
            print(
                f"{p['uid']}: days={p['n_days']} attr={p['total_attr']}(deficit {p['attr_deficit']}, "
                f"recoverable {p['recoverable_attr']}) meals={p['total_meals']}(deficit {p['meal_deficit']}, "
                f"recoverable {p['recoverable_meal_slots']})"
            )


if __name__ == "__main__":
    main()

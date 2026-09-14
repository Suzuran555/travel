"""Transport-position repair pass for the Phase-2 evaluator (upstream >= 764614c).

``Is_space_correct`` tracks positions ACROSS days and requires, for every
activity at a new position, that the transport chain's first leg departs from
the previous activity's position and its last leg arrives at the current one
(and that same-position hops carry NO transports). The planner occasionally
emits a leg anchored to the wrong POI (classically: the return-flight transfer
departing from an attraction instead of the hotel), which under the tightened
evaluator fails the whole plan on 'Invalid Transport information across
positions'.

Relabeling alone is not enough -- the evaluator also validates each inner-city
leg's price/distance/duration against the environment. So this pass REBUILDS
the offending chain with the environment's ``goto`` (via enrich_lateattr.goto,
which applies the repo's price/tickets/cars conventions), probing metro, walk,
and taxi, then time-shifts the probed legs so they arrive exactly at the
activity's start (never departing before the previous activity of the SAME day
ends -- ``Is_time_correct`` resets chronology per day, so a day's first
activity has no lower bound).

Empirical impact on the DashScope live-NL familiar-100 run: 15 plans recover
their commonsense verdict (MacEPR 84 -> 99, C-LPR 82.7 -> 95.5, FPR +2,
Overall 80.36 -> 85.98). Residual failures on those plans are pre-existing
hard-logic (translation) misses, untouched by this pass.
"""
import copy

from . import ctx
from . import enrich_lateattr as LA
from chinatravel.symbol_verification.concept_func import normalize_poi_name as _N


def _hm(t):
    h, m = str(t).split(":")
    return int(h) * 60 + int(m)


def _fmt(x):
    return f"{x // 60:02d}:{x % 60:02d}"


def _shift(tr, delta):
    tr = copy.deepcopy(tr)
    for leg in tr:
        leg["start_time"] = _fmt(_hm(leg["start_time"]) + delta)
        leg["end_time"] = _fmt(_hm(leg["end_time"]) + delta)
    return tr


def repair(plan, city, ppl):
    """Return a repaired deep copy (never mutates the input plan)."""
    p = copy.deepcopy(plan)
    prev = None  # position tracking is CROSS-day (Is_space_correct semantics)
    for day in p.get("itinerary") or []:
        day_prev_end = None  # chronology resets per day (Is_time_correct)
        for a in day.get("activities") or []:
            if a.get("position") is not None:
                cur = a["position"]
            elif a.get("start") and a.get("end"):
                cur = a["start"]  # intercity rows: current position = start
            else:
                cur = None
            trs = a.get("transports") or []
            if prev is not None and cur is not None:
                if _N(cur) != _N(prev):
                    bad = trs and (
                        _N(trs[0].get("start", "")) != _N(prev)
                        or _N(trs[-1].get("end", "")) != _N(cur)
                    )
                    if bad and a.get("start_time"):
                        st = _hm(a["start_time"])
                        for mode in ("metro", "walk", "taxi"):
                            probe = LA.goto(city, prev, cur, "09:00", mode, ppl)
                            if not probe:
                                continue
                            dur = _hm(probe[-1]["end_time"]) - _hm(probe[0]["start_time"])
                            dep = st - dur
                            if day_prev_end is not None:
                                dep = max(dep, day_prev_end)
                            if dep < 0 or dep + dur > st:
                                continue
                            a["transports"] = _shift(probe, dep - _hm(probe[0]["start_time"]))
                            break
                elif trs:
                    # same position: transports must be empty
                    a["transports"] = []
            if a.get("position") is not None:
                prev = a["position"]
            elif a.get("end"):
                prev = a["end"]
            if a.get("end_time"):
                try:
                    day_prev_end = _hm(a["end_time"])
                except Exception:
                    pass
    return p


def commonsense_ok(uid, plan):
    """Commonsense-only verdict via the stock evaluator (same modules ctx uses)."""
    from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints

    _, _, _, cp = evaluate_commonsense_constraints(
        [uid], ctx.qd, {uid: plan}, verbose=False, lang=ctx._lang
    )
    return uid in cp

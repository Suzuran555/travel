"""De-overlap repair pass for the Phase-2 evaluator (upstream >= 764614c).

The tightened `symbol_verification.commonsense_constraint.Is_time_correct` adds
cross-activity checks the old evaluator lacked:

  * an activity must not start before the previous activity ends (overlap);
  * a transport chain must not depart before the previous activity ends;
  * (unchanged) an activity must not start before its own transport arrives.

Enrichment inserts that used to pass under the old evaluator now create these
violations. The other stages are gated on ``ctx.passes`` and simply refuse the
bad insert, so they no longer ADD overlaps. This pass repairs overlaps already
present in the BASE plan (e.g. produced by the search before enrichment) using
a time-only, position-preserving rethread:

  * shift a late-starting activity forward to the previous activity's end,
    keeping its duration (clamped so it never wraps past 24:00);
  * shift a whole transport chain forward (all legs, keeping their order and
    POI endpoints, so ``Is_space_correct`` is untouched) so it departs after
    the previous activity ends and arrives no later than the activity start.

It never drops, reorders, or re-routes anything, so it cannot invalidate a
hard-logic or space constraint that the plan already satisfied. Overlaps that
require rescheduling/re-routing (e.g. a hard-window meal crowded by a
neighbour) are left untouched -- the caller gates on ``passes`` and keeps the
repair only when the whole plan becomes valid.
"""
import copy

_DAY_END = 1440  # 24:00 in minutes


def _t2r(s):
    if s is None:
        return None
    s = str(s).split("次日")[-1]
    try:
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _fmt(m):
    m = int(round(m))
    if m >= _DAY_END:
        return "24:00"
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def _rethread_day(acts):
    out = []
    prev_end = None
    for a in acts:
        a = copy.deepcopy(a)
        st = _t2r(a.get("start_time"))
        ed = _t2r(a.get("end_time"))
        dur = (ed - st) if (st is not None and ed is not None and ed > st) else 0
        # 1) overlap with previous activity -> push start to prev_end, keep duration
        if prev_end is not None and st is not None and st < prev_end:
            st = prev_end
            ed = min(st + dur, _DAY_END)
            a["start_time"] = _fmt(st)
            a["end_time"] = _fmt(ed)
        # 2) transport chain: shift whole chain (keep legs/positions) into [prev_end, st]
        trs = a.get("transports") or []
        if trs and prev_end is not None:
            first = _t2r(trs[0].get("start_time"))
            if first is not None and first < prev_end:
                delta = prev_end - first
                for leg in trs:
                    ls = _t2r(leg.get("start_time"))
                    le = _t2r(leg.get("end_time"))
                    if ls is not None:
                        leg["start_time"] = _fmt(ls + delta)
                    if le is not None:
                        leg["end_time"] = _fmt(le + delta)
            # 3) arrive-before-start: if last leg ends after activity start, push activity
            last = _t2r(trs[-1].get("end_time"))
            if last is not None and st is not None and st < last:
                st = last
                ed = min(st + dur, _DAY_END)
                a["start_time"] = _fmt(st)
                a["end_time"] = _fmt(ed)
        out.append(a)
        if ed is not None:
            prev_end = ed
    return out


def repair(plan):
    """Return a rethread-repaired deep copy of ``plan`` (never mutates input)."""
    p = copy.deepcopy(plan)
    for day in p.get("itinerary") or []:
        day["activities"] = _rethread_day(day.get("activities") or [])
    return p

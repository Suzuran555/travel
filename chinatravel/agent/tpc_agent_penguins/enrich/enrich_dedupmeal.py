"""Drop duplicate meal-type activities: one breakfast/lunch/dinner per day.

The tightened Phase-2 evaluator enforces "Only one <type> is allowed on day N."
The planner (tuned under the older evaluator, where stacking was legal) can
still emit two meals of the same type on one day; no other battery stage
REMOVES activities, so such plans fail commonsense forever ("Repeated Meal
Types in One Day") and every downstream gate rejects every candidate.

This module is removal-only. For each (day, meal-type) with more than one
activity it keeps the best one, ranked by:
  1. inside the evaluator's meal window (breakfast 06-09, lunch 11-14,
     dinner 17-20) -- an out-of-window duplicate is itself a violation;
  2. a real DB restaurant (price > 0) over a price-0 hotel filler -- the
     paid one is likelier a generated must-visit target (mustpoi recovers
     targets later either way);
  3. earlier start_time (stable).

Dropping an activity leaves the NEXT activity's transport chain anchored at
the removed POI; the runner stage stitches with enrich_deoverlap (time) and
enrich_fixspace (env-grounded chain rebuild) before gating.
"""
import copy

MEAL_TYPES = ("breakfast", "lunch", "dinner")
MEAL_WINDOW = {
    "breakfast": ("06:00", "09:00"),
    "lunch": ("11:00", "14:00"),
    "dinner": ("17:00", "20:00"),
}


def _in_window(act):
    w = MEAL_WINDOW.get(act.get("type"))
    if not w:
        return False
    st, et = act.get("start_time") or "", act.get("end_time") or ""
    return bool(st and et) and st >= w[0] and et <= w[1]


def _keep_rank(act):
    # lower sorts first = kept
    return (
        0 if _in_window(act) else 1,
        0 if float(act.get("price") or 0) > 0 else 1,
        act.get("start_time") or "99:99",
    )


def dedup(plan):
    """Return a deduped deep copy, or None if the plan has no duplicates."""
    changed = False
    out = copy.deepcopy(plan)
    for day in out.get("itinerary") or []:
        acts = day.get("activities") or []
        by_type = {}
        for a in acts:
            if a.get("type") in MEAL_TYPES:
                by_type.setdefault(a["type"], []).append(a)
        drop_ids = set()
        for t, lst in by_type.items():
            if len(lst) > 1:
                ranked = sorted(lst, key=_keep_rank)
                drop_ids.update(id(a) for a in ranked[1:])
        if drop_ids:
            changed = True
            day["activities"] = [a for a in acts if id(a) not in drop_ids]
    return out if changed else None

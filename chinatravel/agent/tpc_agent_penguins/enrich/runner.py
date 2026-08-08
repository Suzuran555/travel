"""In-run enrichment battery.

Per-query port of the root-repo enrichment chain that produced the phase-1
result, in the exact order used by the 113-uid honest-Qwen e2e sim
(run_enrich.sh, round 2):

    endday gapmeal att gapattr endattr dav2 travelday bfstack(--stack)
    fillerswap seqswap lateattr attdilute

Each stage re-implements the accept/reject gating of the corresponding
script's __main__ loop verbatim: a stage's edit is kept only if the plan
still passes schema + commonsense + the GENERATED hard logic (ctx.passes)
and its target soft metric improved. Everything runs in memory against a
deadline; a stage that raises is skipped (the plan is left as it was).
"""
import copy
import json
import time
import traceback

from . import ctx
from . import enrich_route as ER
from .ctx import passes, soft_of, agent_for, qd


def _stage_dedupmeal(uid, plan):
    """Drop duplicate meal-type activities (one breakfast/lunch/dinner per day).

    Runs first: the tightened evaluator's per-day meal-type limit is a
    commonsense check the planner (tuned pre-tightening) can still violate,
    and no other stage removes activities -- without this the whole battery
    is gate-blocked on such plans. After removal the neighbour transport
    chains are stitched (deoverlap rethread + fixspace env-grounded rebuild).
    Gated: commonsense micro score must strictly improve (or flip to pass)
    and the hard verdict must not degrade.
    """
    from . import enrich_dedupmeal as M
    from . import enrich_deoverlap as DO
    from . import enrich_fixspace as FS
    from chinatravel.evaluation.commonsense_constraint import (
        evaluate_commonsense_constraints,
    )
    from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
    if not plan.get("itinerary") or passes(uid, plan):
        return plan
    cand = M.dedup(plan)
    if cand is None:
        return plan
    def _cs(p):
        _mac, mic, _x, cp = evaluate_commonsense_constraints(
            [uid], qd, {uid: p}, verbose=False, lang=ctx._lang)
        return float(mic), (uid in cp)
    q = qd[uid]
    cand = DO.repair(cand)
    if not _cs(cand)[1]:
        try:
            c2 = FS.repair(cand, q["target_city"], int(q.get("people_number", 1) or 1))
            if _cs(c2)[0] >= _cs(cand)[0]:
                cand = c2
        except Exception:
            pass
    mic0, ok0 = _cs(plan)
    mic1, ok1 = _cs(cand)
    if (ok0 and not ok1) or mic1 < mic0 - 1e-9:
        return plan
    if not ok1 and mic1 <= mic0 + 1e-9:
        return plan            # no strict improvement -> keep original
    def _hard_ok(p):
        *_, lp = evaluate_hard_constraints_v2(
            [uid], qd, {uid: p}, env_pass_id=[uid], verbose=False, lang=ctx._lang)
        return uid in lp
    if _hard_ok(plan) and not _hard_ok(cand):
        return plan
    return cand


def _stage_deoverlap(uid, plan):
    """Repair base-plan chronology overlaps for the tightened Phase-2 evaluator.

    Runs first: a plan that fails only on the new no-overlap / transport-timing
    checks is rethreaded (time-only, position-preserving) so it passes, which
    also lets the downstream soft-metric stages run on it. Gated: the repair is
    kept only if the whole plan (schema + commonsense + generated hard logic)
    becomes valid.
    """
    from . import enrich_deoverlap as M
    if not plan.get("itinerary") or passes(uid, plan):
        return plan
    cand = M.repair(plan)
    if passes(uid, cand):
        return cand
    return plan


def _stage_fixspace(uid, plan):
    """Rebuild transport chains that violate the tightened cross-position check.

    Runs right after deoverlap. Adopts the candidate only if its commonsense
    verdict flips to pass AND the (env-forced) hard verdict is not degraded --
    even a still-hard-failing plan is worth adopting: it re-enters the C-LPR
    pool (measured +12.8 C-LPR / +5.6 Overall on the live-NL familiar-100).
    """
    from . import enrich_fixspace as M
    from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
    if not plan.get("itinerary") or passes(uid, plan):
        return plan
    if M.commonsense_ok(uid, plan):
        return plan            # commonsense already fine -> failure is elsewhere
    q = qd[uid]
    cand = M.repair(plan, q["target_city"], int(q.get("people_number", 1) or 1))
    if not M.commonsense_ok(uid, cand):
        return plan
    def _hard_ok(p):
        *_, lp = evaluate_hard_constraints_v2(
            [uid], qd, {uid: p}, env_pass_id=[uid], verbose=False, lang=ctx._lang)
        return uid in lp
    if _hard_ok(plan) and not _hard_ok(cand):
        return plan            # never trade a passing hard verdict away
    return cand


def _stage_mustpoi(uid, plan):
    """Recover required-POI hard constraints (generated-DSL targets).

    Runs after fixspace. Each fixer is internally gated on generated-hard
    satisfied-count improvement + commonsense non-degradation; measured
    FPR 78->84 / Overall 85.98->89.03 on the live-NL familiar-100.
    """
    from . import enrich_mustpoi as M
    if not plan.get("itinerary") or passes(uid, plan):
        return plan
    try:
        return M.repair(uid, plan)
    except Exception:
        return plan


def _stage_endday(uid, plan):
    from . import enrich_endday as M
    if not plan.get("itinerary") or not passes(uid, plan):
        return plan
    q = qd[uid]; ag = agent_for(q["target_city"]); ag.query = q
    r0 = M.ddr(plan); cand = copy.deepcopy(plan)
    a = M.enrich_plan(ag, q, cand)
    if a > 0 and M.ddr(cand) > r0 and passes(uid, cand):
        return cand
    return plan


def _stage_gapmeal(uid, plan):
    from . import enrich_gapmeal as M
    if not plan.get("itinerary") or not passes(uid, plan):
        return plan
    q = qd[uid]; ag = agent_for(q["target_city"]); ag.query = q
    r0 = M.ddr(plan); cand = copy.deepcopy(plan)
    a = M.enrich_plan(ag, q, cand)
    if a > 0 and M.ddr(cand) > r0 and passes(uid, cand):
        return cand
    return plan


def _stage_att(uid, plan):
    from . import enrich_att as M
    if not plan.get("itinerary") or not passes(uid, plan):
        return plan
    q = qd[uid]; ag = agent_for(q["target_city"]); ag.query = q
    a0 = M.att_of(plan); cand = copy.deepcopy(plan)
    s, _st = M.enrich_plan(ag, q, cand)
    if s > 0 and M.att_of(cand) > a0 and passes(uid, cand):
        return cand
    return plan


def _stage_gapattr(uid, plan):
    from . import enrich_gapattr as M
    it = plan.get("itinerary") or []
    if not it:
        return plan
    days = len(it)
    shortfall = 4 * days - M.attr_count(plan)
    if shortfall <= 0:
        return plan
    q = qd[uid]
    if not passes(uid, plan):
        return plan
    bud = M.binding_budget(q)
    slack = (bud - M.plan_cost(plan)) if bud is not None else None
    icap = M.binding_ic_cap(q)
    ic_slack = (icap - M.ic_cost(plan)) if icap is not None else None
    modes, free_first, free_only = M.ALL_MODES, False, False
    if slack is not None and slack < M.MIN_BUDGET_SLACK:
        modes, free_only = ("walk",), True
    elif slack is not None and slack <= M.FREE_FIRST_SLACK:
        free_first = True
    if ic_slack is not None and ic_slack < M.IC_WALK_SLACK:
        modes, free_only = ("walk",), True
    base_avg = M.avg_transit(plan)
    att_cap = M.ATT_GUARD_AVG if base_avg <= M.ATT_GUARD_AVG else None
    ag = agent_for(q["target_city"]); ag.query = q
    ex_names, ex_types = M.excl_sets(q)
    d0, a0 = M.dav(plan), M.att_metric(plan)
    cand = copy.deepcopy(plan)
    added, _evals = M.enrich_plan(ag, q, cand, shortfall, ex_names, ex_types,
                                  modes=modes, free_first=free_first,
                                  free_only=free_only, att_cap=att_cap)
    if added > 0:
        d1, a1 = M.dav(cand), M.att_metric(cand)
        guard_ok = att_cap is None or M.avg_transit(cand) <= att_cap
        if (d1 + a1) > (d0 + a0) and guard_ok and passes(uid, cand):
            return cand
    return plan


def _stage_endattr(uid, plan):
    from . import enrich_endattr as M
    it = plan.get("itinerary") or []
    if not it or M.attr_count(plan) >= 4 * len(it):
        return plan
    has_slot = False
    for d in it:
        acts = d["activities"]
        li = M._last_real_idx(acts)
        if (li >= 0 and li + 1 < len(acts)
                and acts[li + 1].get("type") == "accommodation"
                and acts[li].get("end_time")
                and M.hm(acts[li]["end_time"]) <= M.LAST_END):
            has_slot = True
            break
    if not has_slot or not passes(uid, plan):
        return plan
    q = qd[uid]; ag = agent_for(q["target_city"]); ag.query = q
    ex_names, ex_types, budget = M.query_meta(q)
    d0, a0 = M.dav(plan), M.att_avg(plan)
    cand = copy.deepcopy(plan)
    add = M.enrich_plan(ag, q, cand, ex_names, ex_types, budget)
    if add > 0:
        d1, a1 = M.dav(cand), M.att_avg(cand)
        att_ok = not (a0 <= M.ATT_ONE and a1 > M.ATT_ONE)
        if d1 > d0 and att_ok and passes(uid, cand):
            return cand
    return plan


def _stage_dav2(uid, plan):
    from . import enrich_dav2 as M
    from . import enrich_gapattr as GA
    it = plan.get("itinerary") or []
    if not it:
        return plan
    if 4 * len(it) - sum(1 for d in it for a in d["activities"]
                         if a.get("type") == "attraction") <= 0:
        return plan
    q = qd[uid]
    if not passes(uid, plan):
        return plan
    ag = agent_for(q["target_city"]); ag.query = q
    d0 = GA.dav(plan)
    cand = copy.deepcopy(plan)
    added, _evals, _parts, _tags, att_cap = M.enrich_plan(uid, ag, q, cand)
    if added > 0:
        d1, a1 = GA.dav(cand), GA.avg_transit(cand)
        guard_ok = att_cap is None or a1 <= att_cap
        if d1 > d0 and guard_ok and passes(uid, cand):
            return cand
    return plan


def _stage_travelday(uid, plan):
    from . import enrich_travelday_meals as M
    if not plan.get("itinerary"):
        return plan
    try:
        tasks = M.classify(plan)
    except Exception:
        return plan
    if not tasks or not passes(uid, plan):
        return plan
    q = qd[uid]; ag = agent_for(q["target_city"]); ag.query = q
    r0 = M.ddr(plan); cand = copy.deepcopy(plan)
    a, _done = M.enrich_plan(ag, q, cand, tasks)
    if a > 0 and M.ddr(cand) > r0 and passes(uid, cand):
        return cand
    return plan


def _stage_bfstack(uid, plan):
    from . import enrich_bfstack as M
    it = plan.get("itinerary") or []
    if not it or M.meal_count(plan) >= 3 * len(it):
        return plan
    if not passes(uid, plan):
        return plan
    r0 = M.ddr(plan)
    cand = copy.deepcopy(plan)
    b = M.add_default(cand)
    # --stack (add_stack) is DISABLED under the Phase-2 evaluator: it inserts a
    # 2nd/3rd breakfast per day, which now fails the new "Repeated Meal Types in
    # One Day" commonsense check (passes() would reject it anyway). Keep only the
    # single legitimate hotel breakfast per breakfast-less morning.
    if b > 0 and M.ddr(cand) > r0 and passes(uid, cand):
        return cand
    return plan


def _stage_fillerswap(uid, plan):
    from . import enrich_fillerswap as M
    s0 = soft_of(plan)
    if s0 is None or s0[2] >= 1.0 - 1e-9:
        return plan
    if not M.long_leg_targets(plan) or not passes(uid, plan):
        return plan
    q = qd[uid]; ag = agent_for(q["target_city"]); ag.query = q
    new_plan, msgs = M.enrich_plan(ag, q, uid, copy.deepcopy(plan), verbose=False)
    if not msgs:
        return plan
    s1 = soft_of(new_plan)
    if (s1 and s1[2] > s0[2] + 1e-9 and abs(s1[0] - s0[0]) < 1e-9
            and abs(s1[1] - s0[1]) < 1e-9):
        return new_plan
    return plan


def _stage_seqswap(uid, plan):
    from . import enrich_seqswap as M
    s0 = soft_of(plan) if plan.get("itinerary") else None
    if s0 is None or s0[2] >= 1.0 - 1e-9:
        return plan
    if not passes(uid, plan):
        return plan
    cand, s1, _info = M.remode_plan(uid, plan, s0)
    if cand is None:
        return plan
    # same round-trip re-verification the script does before writing
    cand = json.loads(json.dumps(cand, ensure_ascii=False, default=M._np))
    s2 = soft_of(cand)
    if (s2 is None or abs(s2[0] - s0[0]) > 1e-12 or abs(s2[1] - s0[1]) > 1e-12
            or s2[2] <= s0[2] + 1e-9 or not passes(uid, cand)):
        return plan
    return cand


def _stage_lateattr(uid, plan):
    from . import enrich_lateattr as M
    s0 = soft_of(plan)
    if not s0 or s0[0] >= 1.0:
        return plan
    if not passes(uid, plan):
        return plan
    q = qd[uid]
    new_plan, log = M.enrich(uid, copy.deepcopy(plan), q)
    if not log:
        return plan
    # every insert inside M.enrich is individually soft-gated + passes()-gated
    return new_plan


def _stage_attdilute(uid, plan):
    from . import enrich_attdilute as M
    s0 = soft_of(plan)
    if not s0 or s0[2] >= 1.0:
        return plan
    if not passes(uid, plan):
        return plan
    q = qd[uid]
    new_plan, msgs = M.enrich_plan(uid, q, copy.deepcopy(plan))
    if not msgs:
        return plan
    s1 = soft_of(new_plan)
    if s1 and s1[2] > s0[2] and s1[0] == s0[0] and s1[1] == s0[1]:
        return new_plan
    return plan


# DDR stages (endday/gapmeal/travelday/bfstack) run BEFORE the attraction
# stages: on the serving stack the battery is deadline-bounded and DDR misses
# (39 travelday slots measured on align-v3) were starving behind the pricier
# DAV/ATT passes.
STAGES = [
    ("dedupmeal", _stage_dedupmeal),
    ("deoverlap", _stage_deoverlap),
    ("fixspace", _stage_fixspace),
    ("mustpoi", _stage_mustpoi),
    ("endday", _stage_endday),
    ("gapmeal", _stage_gapmeal),
    ("travelday", _stage_travelday),
    ("bfstack", _stage_bfstack),
    ("att", _stage_att),
    ("gapattr", _stage_gapattr),
    ("endattr", _stage_endattr),
    ("dav2", _stage_dav2),
    ("fillerswap", _stage_fillerswap),
    ("seqswap", _stage_seqswap),
    ("lateattr", _stage_lateattr),
    ("attdilute", _stage_attdilute),
]


def run_battery(uid, plan, query, agent, world_env, deadline, lang="en", log=print):
    """Run the enrichment battery on one plan, in memory, against a deadline.

    query MUST be the translated query (generated hard_logic_py) -- it is the
    only constraint source every gate sees.
    """
    if not isinstance(plan, dict) or not plan.get("itinerary"):
        return plan
    ctx.setup(uid, query, agent, world_env, lang=lang)
    ER._cur = uid
    for name, stage in STAGES:
        remaining = deadline - time.time()
        if remaining < 5:
            log(f"[enrich] budget exhausted before {name} ({remaining:.0f}s left)")
            break
        t0 = time.time()
        try:
            plan = stage(uid, plan)
        except Exception:
            log(f"[enrich] stage {name} raised (skipped):\n{traceback.format_exc()}")
        else:
            log(f"[enrich] {name}: {time.time() - t0:.1f}s soft={soft_of(plan)}")
    return plan

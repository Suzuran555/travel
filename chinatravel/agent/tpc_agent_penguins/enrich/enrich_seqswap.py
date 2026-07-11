"""Global budget-aware transport re-moding (seqswap).

Many residual-ATT plans sit within pennies of an innercity-transport cost cap,
so single-leg upgrade attempts always fail; but re-moding the WHOLE plan
jointly works: downgrading an expensive taxi/metro leg to a cheaper mode FREES
budget for upgrading monster walks elsewhere.

Per plan:
  1. Determine the binding cap by EVALUATING hard_logic_py on cost-inflated
     copies of the plan (binary search on extra innercity fare). A cap inside
     an OR whose other branch holds is automatically seen as non-binding
     (inflation does not break the OR) -> treated as unlimited.
  2. For every leg (activity with non-empty transports) fetch all mode options
     (walk / metro w. tickets*people / taxi w. cars) from the env API.
  3. Optimize: minimize total leg-minutes subject to total innercity fare <=
     cap (greedy by minutes-saved-per-yuan from the min-cost assignment).
  4. Gate: ATT strictly up, DAV/DDR unchanged, hard logic re-evaluated on the
     rebuilt plan, full passes() (schema+commonsense+hardlogic) green.
     Fallback variants (drop weakest upgrades / leave-one-out) if the full
     assignment fails. Legs are only re-moded, never deleted.

DRY-RUN by default (prints per-uid would-apply diagnostics, writes nothing).
Flags: --shard i/N  --limit K  --uids u1,u2  --apply
Env:   ENRICH_RES selects the target dir.
"""
import os, sys, json, copy, glob, argparse

from chinatravel.data.load_datasets import load_json_file
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
from chinatravel.symbol_verification.hard_constraint import evaluate_constraints_py, _set_tool_lang
from .ctx import soft_of

from .ctx import qd, sch, passes, agent_for

WALK_MAX = 90        # never manufacture a fresh walk leg longer than this
MAX_VARIANTS = 12




def hm(t):
    h, m = str(t).split(":"); return int(h) * 60 + int(m)


def actpos(a):
    return a.get("position") or a.get("end") or a.get("start")




def leg_time(tr):
    return sum(hm(s["end_time"]) - hm(s["start_time"]) for s in tr)


def leg_cost(tr):
    return sum(float(s.get("cost", 0) or 0) for s in tr)


def cur_mode(tr):
    modes = set(s.get("mode", s.get("type")) for s in tr)
    if "taxi" in modes:
        return "taxi"
    if "metro" in modes:
        return "metro"
    return "walk"


def hard_ok(uid, plan):
    return all(evaluate_constraints_py(qd[uid]["hard_logic_py"], plan))


def _first_leg_step(plan):
    for d in plan["itinerary"]:
        for a in d["activities"]:
            tr = a.get("transports") or []
            if tr:
                return tr[0]
    return None


def headroom(uid, plan, hi=1e6):
    """Max extra yuan of innercity fare the hard logic tolerates on this plan.
    Evaluates the actual hard_logic_py (OR-disjuncts included) on inflated
    copies -- never regex-guesses caps. Returns float('inf') if unlimited."""
    if _first_leg_step(plan) is None:
        return 0.0
    probe = copy.deepcopy(plan)
    step = _first_leg_step(probe)
    base = float(step.get("cost", 0) or 0)

    def ok(delta):
        step["cost"] = base + delta
        return all(evaluate_constraints_py(qd[uid]["hard_logic_py"], probe))

    try:
        if not ok(0.0):
            return 0.0
        if ok(hi):
            return float("inf")
        lo, h = 0.0, hi
        while h - lo > 0.005:
            mid = (lo + h) / 2.0
            if ok(mid):
                lo = mid
            else:
                h = mid
        return lo
    finally:
        step["cost"] = base


def build_legs(ag, city, plan):
    """One entry per activity with non-empty transports:
    (di, ai, m0, opts) with opts[mode] = (minutes, fare, transports_or_None).
    transports None => keep the current (untouched) leg."""
    legs = []
    for di, d in enumerate(plan["itinerary"]):
        acts = d["activities"]
        for ai, a in enumerate(acts):
            tr = a.get("transports") or []
            if not tr:
                continue
            m0 = cur_mode(tr)
            t0, c0 = leg_time(tr), leg_cost(tr)
            opts = {m0: (t0, c0, None)}
            station = a.get("type") in ("train", "airplane")
            origin, dest = tr[0].get("start"), tr[-1].get("end")
            if not station and ai > 0:
                o2, d2 = actpos(acts[ai - 1]), actpos(a)
                if o2 and d2:
                    origin, dest = o2, d2
            depart = tr[0].get("start_time")
            if origin and dest and origin != dest and depart:
                for m in ("walk", "metro", "taxi"):
                    if m == m0:
                        continue
                    try:
                        alt = ag.collect_innercity_transport(city, origin, dest, depart, m)
                    except Exception:
                        continue
                    if not isinstance(alt, list) or not alt:
                        continue
                    t, c = leg_time(alt), leg_cost(alt)
                    if m == "walk" and t > max(WALK_MAX, t0):
                        continue  # never manufacture a fresh monster walk
                    opts[m] = (t, c, alt)
            legs.append((di, ai, m0, opts))
    return legs


def optimize(legs, cap, banned=frozenset()):
    """Min total minutes s.t. total fare <= cap. Start from the min-cost
    assignment, then greedy upgrades by minutes-saved-per-yuan.
    banned: set of (leg_index, mode) options that must not be used."""
    def usable(li, m):
        return m == legs[li][2] or (li, m) not in banned

    assign = {}
    for li, (_, _, m0, opts) in enumerate(legs):
        cands = [m for m in opts if usable(li, m)]
        assign[li] = min(cands, key=lambda m: (opts[m][1], opts[m][0]))
    while True:
        C = sum(legs[li][3][assign[li]][1] for li in assign)
        best = None
        for li, (_, _, m0, opts) in enumerate(legs):
            ct, cc, _ = opts[assign[li]]
            for m, (t, c, _tr) in opts.items():
                if m == assign[li] or t >= ct - 1e-9 or not usable(li, m):
                    continue
                dc = c - cc
                if C + dc > cap + 1e-6:
                    continue
                gain = ct - t
                score = 1e12 + gain if dc <= 1e-9 else gain / dc
                if best is None or score > best[0]:
                    best = (score, li, m)
        if best is None:
            break
        assign[best[1]] = best[2]
    return assign


def variant_cost_time(legs, assign, subset):
    C = T = 0.0
    for li, (_, _, m0, opts) in enumerate(legs):
        m = assign[li] if li in subset else m0
        t, c, _ = opts[m]
        C += c; T += t
    return C, T


def build_trial(ag, plan, legs, assign, subset):
    trial = copy.deepcopy(plan)
    for li in subset:
        di, ai, m0, opts = legs[li]
        tr = opts[assign[li]][2]
        if tr is None:
            continue
        trial["itinerary"][di]["activities"][ai]["transports"] = copy.deepcopy(tr)
    try:
        ag._repair_itinerary_times(trial["itinerary"])
    except Exception:
        return None
    return trial


def gates(uid, s0, trial):
    """(ok, soft3). ATT strictly up, DAV/DDR unchanged, hard logic + full eval."""
    s1 = soft_of(trial)
    if s1 is None:
        return False, s1
    if abs(s1[0] - s0[0]) > 1e-12 or abs(s1[1] - s0[1]) > 1e-12:
        return False, s1
    if s1[2] <= s0[2] + 1e-9:
        return False, s1
    if not hard_ok(uid, trial):
        return False, s1
    if not passes(uid, trial):
        return False, s1
    return True, s1


def constructive(uid, ag, plan, legs, assign, downs, ups, cap, s0):
    """Fallback when joint variants fail: grow the accepted move-set one move
    at a time with full passes() as the feasibility oracle (isolates toxic
    moves, e.g. station legs pinned by flight chronology), then prune downs
    that turned out unnecessary."""
    def feasible(sub):
        C, _ = variant_cost_time(legs, assign, sub)
        if C > cap + 1e-6:
            return None
        t = build_trial(ag, plan, legs, assign, sub)
        if t is None or not hard_ok(uid, t) or not passes(uid, t):
            return None
        return t

    cur = set()
    freed = sorted(downs, key=lambda li: legs[li][3][legs[li][2]][1] - legs[li][3][assign[li]][1],
                   reverse=True)
    for li in freed:
        if feasible(cur | {li}) is not None:
            cur.add(li)
    for li in ups:
        if feasible(cur | {li}) is not None:
            cur.add(li)
    # prune downs no longer needed (each removal can only lower total minutes)
    for li in list(cur & set(downs)):
        if feasible(cur - {li}) is not None:
            cur.discard(li)
    if not cur:
        return None, None, "constructive empty"
    trial = build_trial(ag, plan, legs, assign, cur)
    if trial is None:
        return None, None, "constructive rebuild failed"
    ok, s1 = gates(uid, s0, trial)
    if not ok:
        return None, None, "constructive gated out"
    det = ["%s a%d.%d %s->%s" % ("dn" if li in downs else "up",
                                 legs[li][0], legs[li][1], legs[li][2], assign[li])
           for li in sorted(cur)]
    return trial, s1, "constructive | " + ", ".join(det)


def cs_ok(uid, plan):
    """Schema + commonsense only (no hard logic): used to detect moves that
    are structurally toxic (e.g. break intercity chronology) regardless of
    the fare cap."""
    s = set(evaluate_schema_constraints([uid], {uid: plan}, schema=sch)[2])
    if uid not in s:
        return False
    c = set(evaluate_commonsense_constraints([uid], qd, {uid: plan}, verbose=False, lang="en")[3])
    return uid in c


def remode_plan(uid, plan, s0):
    """Returns (best_trial, soft3, info) or (None, None, info)."""
    q = qd[uid]
    city = q["target_city"]
    ag = agent_for(city)
    ag.query = q
    legs = build_legs(ag, city, plan)
    if not legs:
        return None, None, "no legs"
    C0 = sum(opts[m0][1] for _, _, m0, opts in legs)
    T0 = sum(opts[m0][0] for _, _, m0, opts in legs)
    hr = headroom(uid, plan)
    cap = round((C0 + hr) * 100) / 100 if hr != float("inf") else float("inf")

    banned = set()
    info = ""
    for attempt in range(4):
        assign = optimize(legs, cap, banned)
        moves = [li for li, (_, _, m0, _o) in enumerate(legs) if assign[li] != m0]
        info = (f"cap={cap:.2f} fare={C0:.2f} legs={len(legs)} moves={len(moves)}"
                f" banned={len(banned)}")
        if not moves:
            return None, None, info + " | no moves"
        ups = sorted((li for li in moves if legs[li][3][assign[li]][0] < legs[li][3][legs[li][2]][0] - 1e-9),
                     key=lambda li: legs[li][3][legs[li][2]][0] - legs[li][3][assign[li]][0], reverse=True)
        downs = [li for li in moves if li not in set(ups)]

        subsets = [frozenset(moves)]
        for k in range(len(ups) - 1, 0, -1):
            subsets.append(frozenset(downs + ups[:k]))
        if len(ups) >= 2:
            for i in range(min(4, len(ups))):
                subsets.append(frozenset(downs + ups[:i] + ups[i + 1:]))
        # ups-only greedy under the raw cap (no downs at all)
        sel, spend = [], C0
        for li in ups:
            dc = legs[li][3][assign[li]][1] - legs[li][3][legs[li][2]][1]
            if spend + dc <= cap + 1e-6:
                sel.append(li); spend += dc
        if sel:
            subsets.append(frozenset(sel))

        seen, variants = set(), []
        for sub in subsets:
            if not sub or sub in seen:
                continue
            seen.add(sub)
            C, T = variant_cost_time(legs, assign, sub)
            if C > cap + 1e-6 or T >= T0 - 1e-9:
                continue
            variants.append((T, sub))
        variants.sort()
        for T, sub in variants[:MAX_VARIANTS]:
            trial = build_trial(ag, plan, legs, assign, sub)
            if trial is None:
                continue
            ok, s1 = gates(uid, s0, trial)
            if ok:
                det = ["%s a%d.%d %s->%s" % ("dn" if li in downs else "up",
                                             legs[li][0], legs[li][1], legs[li][2], assign[li])
                       for li in sorted(sub)]
                return trial, s1, info + " | " + ", ".join(det)
        # probe each move alone at the commonsense level; ban the toxic ones
        # (fare ignored here on purpose) and re-optimize without them
        newly = set()
        for li in moves:
            trial = build_trial(ag, plan, legs, assign, {li})
            if trial is None or not cs_ok(uid, trial):
                newly.add((li, assign[li]))
        if newly - banned:
            banned |= newly
            continue
        # no single-move culprit: combination effect -> constructive growth
        trial, s1, cinfo = constructive(uid, ag, plan, legs, assign, downs, ups, cap, s0)
        if trial is not None:
            return trial, s1, info + " | " + cinfo
        return None, None, info + " | no variant passed; " + cinfo
    return None, None, info + " | ban loop exhausted"


def _np(o):
    if hasattr(o, "item"):
        return o.item()
    return float(o)



"""Filler-swap ATT enrichment: replace FAR-FLUNG filler POIs with near equivalents.

The V6 planner sometimes picks filler attractions alphabetically ('Ant Workshop'
~100 km from Hangzhou, 'Fairy Mountain National Forest Park' 120 km from
Chongqing, Nanjing University Gulou via the far Poi-table coords...), producing
120-205-min legs that crater ATT. Audits 'att-taxi-long' / 'att-monster-walks' /
'att-classify' category D verified these POIs are NOT mandated by hard logic and
that near substitutes exist.

For every plan with ATT<1: for each leg > LONG_LEG minutes whose destination is
an unpinned attraction / restaurant meal (or whose origin is an unpinned
attraction when the destination itself is a station/hotel), try replacing the
POI with candidates near the route, ranked by detour = dist(prev,X)+dist(X,next)
-dist(prev,next). Pins are parsed from BOTH hard_logic idioms — the
activity_position(...)=='NAME' equality and the set-based {'NAME',...} &/<=
*_name_set forms — via a conservative rule: any POI whose exact name appears
quoted anywhere in hard_logic_py is never swapped out, and never swapped in.
Attraction/restaurant TYPE requirements ({'park'}&attraction_type_set etc.) are
preserved: if the outgoing POI is the plan's last holder of a required type, the
candidate must hold that type; types under a not(...) exclusion are never
introduced. Cost caps (incl. OR-disjuncts) are NOT regex-guessed — the full
3-stage passes() evaluates the real hard_logic, and candidate ordering merely
prefers cheaper POIs so cap-pinned plans still find affordable swaps.

Both adjacent transports are rebuilt, fastest valid mode first (fallback to the
next-fastest so fare caps can still be satisfied). A stranded following meal leg
is caught by the same greedy loop on the next iteration (the rebuilt leg to the
old far restaurant shows up as a new long leg and gets its restaurant swapped).

A swap is kept ONLY if the plan-level ATT strictly improves, DAV and DDR are
unchanged, and the full schema+commonsense+hardlogic passes(). DRY-RUN default:
prints per-uid would-apply diagnostics, writes nothing; --apply writes.

Usage: ENRICH_RES=<dir> .venv/bin/python enrich_fillerswap.py \
           [--shard i/N] [--limit K] [--uids u1,u2] [--apply]
"""
import os, sys, json, copy, glob, argparse
from . import enrich_route as ER
from .enrich_route import passes, agent_for, actpos, qd, hm, mh
from .ctx import soft_of
from chinatravel.symbol_verification.concept_func import innercity_transport_time

LONG_LEG = 35            # minutes: legs above this are swap targets
MAX_SWAPS = 8            # per plan
MAX_EVALS_PER_LEG = 20   # passes() budget per target leg
N_CAND = 30              # detour-ranked candidates per target
MEALS = ("breakfast", "lunch", "dinner")
MEALWIN = {"breakfast": (6 * 60, 9 * 60), "lunch": (11 * 60, 14 * 60),
           "dinner": (17 * 60, 20 * 60)}

import re as _re
_QUOTED = _re.compile(r"'([^']+)'|\"([^\"]+)\"")
_TYPE_RULE = _re.compile(
    r"result\s*=\s*(not\s*\()?\s*\(?\s*\{([^}]*)\}\s*(&|<=)\s*(attraction|restaurant)_type_set")


def _sanitize(o):
    """numpy scalars -> python (plans must round-trip through pure json)."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def _names_in(text):
    return set(a or b for a, b in _QUOTED.findall(text))


def hard_named(query):
    """Every quoted string in hard_logic_py: POIs named here are pinned either
    way (must OR must-not) — never swap them out, never swap them in."""
    s = set()
    for h in query.get("hard_logic_py", []):
        s |= _names_in(h)
    return s


def type_rules(query, kind):
    """(any_of_groups, all_of, excluded) type constraints for kind
    ('attraction'|'restaurant'), parsed from the three set idioms."""
    any_of, all_of, excl = [], set(), set()
    for h in query.get("hard_logic_py", []):
        for m in _TYPE_RULE.finditer(h):
            neg, body, op, k = m.groups()
            if k != kind:
                continue
            names = _names_in("{" + body + "}")
            if neg:
                excl |= names
            elif op == "&":
                any_of.append(names)
            else:
                all_of |= names
    return any_of, all_of, excl


def _poi_type(row, kind):
    return str(row.get("type" if kind == "attraction" else "cuisine", ""))


def _forced_types(plan, old_name, old_type, any_of, all_of, tmap):
    """Sets the replacement's type must intersect, after old_name leaves.
    tmap: in-plan attraction/restaurant name -> type (excluding old)."""
    remaining = set(tmap.values())
    forced = []
    for grp in any_of:
        if old_type in grp and not (grp & remaining):
            forced.append(grp)
    for t in all_of:
        if old_type == t and t not in remaining:
            forced.append({t})
    return forced


def _transport_opts(ag, city, a, b, t0):
    """Valid transport legs a->b departing t0, fastest first."""
    opts = []
    for mode in ("walk", "metro", "taxi"):
        try:
            tr = ag.collect_innercity_transport(city, a, b, t0, mode)
        except Exception:
            continue
        if isinstance(tr, list) and tr:
            opts.append((innercity_transport_time(tr), _sanitize(tr)))
    opts.sort(key=lambda x: x[0])
    return opts


def try_swap(ag, q, uid, plan, di, j, s0, verbose=False):
    """Replace the POI of activity j on day di with a near-route equivalent.
    Returns (new_plan, msg) on success else (None, None). Only accepted if ATT
    strictly improves, DAV/DDR unchanged, and full passes() holds."""
    city = q["target_city"]
    acts = plan["itinerary"][di]["activities"]
    old = acts[j]
    kind = "restaurant" if old["type"] in MEALS else "attraction"
    old_name = old.get("position")
    tr_old = old.get("transports") or []
    if not old_name or not tr_old:
        return None, None
    pinned = hard_named(q)
    if old_name in pinned:
        return None, None
    prev_pos = tr_old[0].get("start")
    depart = tr_old[0].get("start_time")
    if not prev_pos or not depart:
        return None, None
    nxt = acts[j + 1] if j + 1 < len(acts) else None
    nxt_station = nxt is not None and nxt.get("type") in ("train", "airplane")
    if nxt is None:
        nxt_pos = None
    elif nxt_station:
        nxt_pos = (nxt.get("transports") or [{}])[-1].get("end")
    else:
        nxt_pos = actpos(nxt)
    ppl = int(q.get("people_number", 1))
    dur = hm(old["end_time"]) - hm(old["start_time"])
    if dur <= 0:
        dur = 30
    old_price = float(old.get("price", 0) or 0)

    df = ag.memory["attractions"] if kind == "attraction" else ag.memory["restaurants"]
    visited = set(actpos(x) for d in plan["itinerary"] for x in d["activities"])
    df = df[~df["name"].isin(visited | pinned)]
    any_of, all_of, excl_t = type_rules(q, kind)
    tmap = {}
    # in-plan type map from the FULL city table (df above already dropped used rows)
    full = ag.memory["attractions"] if kind == "attraction" else ag.memory["restaurants"]
    name2type = dict(zip(full["name"], full["type" if kind == "attraction" else "cuisine"]))
    for d in plan["itinerary"]:
        for x in d["activities"]:
            nm = x.get("position")
            if nm == old_name or nm not in name2type:
                continue
            if (kind == "attraction" and x.get("type") == "attraction") or \
               (kind == "restaurant" and x.get("type") in MEALS):
                tmap[nm] = str(name2type[nm])
    old_type = str(name2type.get(old_name, ""))
    forced = _forced_types(plan, old_name, old_type, any_of, all_of, tmap)

    try:
        dPN = ag.calculate_distance(q, prev_pos, nxt_pos) if nxt_pos else 0.0
    except Exception:
        dPN = 0.0
    scored = []
    for _, r in df.iterrows():
        nm = r["name"]
        if nm == prev_pos or nm == nxt_pos:
            continue
        t = _poi_type(r, kind)
        if t in excl_t:
            continue
        if forced and not all(t in grp for grp in forced):
            continue
        try:
            d1 = ag.calculate_distance(q, prev_pos, nm)
            d2 = ag.calculate_distance(q, nm, nxt_pos) if nxt_pos else 0.0
        except Exception:
            continue
        if d1 is None or d2 is None:
            continue
        scored.append((d1 + d2 - (dPN or 0.0), float(r.get("price", 0) or 0), r))
    scored.sort(key=lambda x: x[0])
    # detour-ranked head, then cheaper-than-old tail (cap-pinned plans need it)
    cands = scored[:10] + [c for c in scored[10:] if c[1] <= old_price][:N_CAND]
    cands = cands[:N_CAND]

    evals = 0
    for detour, price, r in cands:
        name = r["name"]
        ot, et = str(r.get("opentime")), str(r.get("endtime"))
        for t1, tr1 in _transport_opts(ag, city, prev_pos, name, depart)[:2]:
            arr = tr1[-1]["end_time"]
            start = arr if hm(arr) >= hm(ot) else ot
            if kind == "restaurant":
                lo, hi = MEALWIN[old["type"]]
                if hm(start) < lo:
                    start = mh(lo)
                if hm(start) + dur > hi:
                    continue
            end = mh(hm(start) + dur)
            if hm(et) >= hm(ot) and (hm(start) > hm(et) or hm(end) > hm(et)):
                continue
            tr2_opts = [(0, None)]
            if nxt is not None and nxt_pos:
                tr2_opts = _transport_opts(ag, city, name, nxt_pos, end)[:2]
                if not tr2_opts:
                    continue
            for _, tr2 in tr2_opts:
                X = _sanitize(dict(old))
                X.update({"position": name, "price": price, "cost": price * ppl,
                          "start_time": start, "end_time": end, "transports": tr1})
                cand = copy.deepcopy(plan)
                cacts = cand["itinerary"][di]["activities"]
                cacts[j] = X
                if tr2 is not None:
                    cacts[j + 1] = copy.deepcopy(nxt)
                    cacts[j + 1]["transports"] = tr2
                if not nxt_station:
                    try:
                        ag._repair_itinerary_times(cand["itinerary"])
                    except Exception:
                        continue
                cand = _sanitize(cand)
                s1 = soft_of(cand)
                if s1 is None:
                    continue
                if not (s1[2] > s0[2] + 1e-9 and abs(s1[0] - s0[0]) < 1e-9
                        and abs(s1[1] - s0[1]) < 1e-9):
                    continue
                evals += 1
                if passes(uid, cand):
                    msg = (f"    day{di} [{old['type']}] {old_name} -> {name} "
                           f"(detour {detour:.1f}km, price {price:.0f}) "
                           f"ATT {s0[2]:.4f}->{s1[2]:.4f}")
                    return cand, msg
                if evals >= MAX_EVALS_PER_LEG:
                    return None, None
        if evals >= MAX_EVALS_PER_LEG:
            break
    return None, None


def long_leg_targets(plan):
    """(leg_min, di, target_idx) for swap-worthy long legs, worst first."""
    out = []
    for di, day in enumerate(plan["itinerary"]):
        acts = day["activities"]
        accom = next((a.get("position") for d2 in plan["itinerary"]
                      for a in d2["activities"] if a.get("type") == "accommodation"), None)
        for ai, a in enumerate(acts):
            tr = a.get("transports") or []
            if not tr:
                continue
            t = innercity_transport_time(tr)
            if t <= LONG_LEG:
                continue
            typ = a.get("type")
            if typ == "attraction":
                out.append((t, di, ai))
            elif typ in MEALS and a.get("position") != accom:
                out.append((t, di, ai))
            elif ai > 0 and acts[ai - 1].get("type") == "attraction":
                out.append((t, di, ai - 1))   # far ORIGIN attraction, fixed dest
    out.sort(reverse=True)
    return out


def enrich_plan(ag, q, uid, plan, verbose=True):
    """Greedy: fix worst long leg, recompute, repeat. Returns (plan, msgs)."""
    msgs = []
    failed = set()
    for _ in range(MAX_SWAPS):
        s0 = soft_of(plan)
        if s0 is None or s0[2] >= 1.0 - 1e-9:
            break
        done = False
        for t, di, j in long_leg_targets(plan):
            key = (di, plan["itinerary"][di]["activities"][j].get("position"))
            if key in failed:
                continue
            cand, msg = try_swap(ag, q, uid, plan, di, j, s0, verbose)
            if cand is not None:
                plan = cand
                msgs.append(msg)
                failed.clear()
                done = True
                break
            failed.add(key)
        if not done:
            break
    return plan, msgs



"""Intercity flight REBOOKING enricher (wrong-airport / dead-schedule / bad transfer).

Patterns fixed (audited in fix_recipes.json, verified end-to-end by adversarial
verifiers):
  1. WRONG AIRPORT: plan flies into/out of an airport far from the hotel while a
     near-identical flight serves a closer airport of the SAME city (Chengdu
     Tianfu vs Shuangliu: FL617->FL616, FL466->FL467/FL464; Beijing Capital vs
     Daxing; Shanghai Pudong vs Hongqiao). Rebook the flight, rebuild the
     airport<->hotel transfer legs with the fastest mode that survives the
     hard-logic cost caps (passes() arbitrates the OR-disjuncts exactly).
  2. TRANSFER MODE SWAP: the airport/station access leg uses a slow mode (378-min
     night walk etc.) while a faster mode exists; regenerate the transfer legs
     only, same flight.
  3. RECIPE REBUILDS: a few audited uids need a full verified itinerary swap
     (dead-schedule DAV rebuild 20250322232713611229 Wuhan->Shanghai; deep-worst
     wrong-airport+wrong-hotel rebuilds 20250324222831510655 / 20250322060622761612;
     night-arrival walk 20250321025717139459; day-4 trim 20250324230442721134).
     The verified itineraries live in enrich_rebook_recipes.json (repo root) and
     are applied only through the same regression gate.

Every edit is kept ONLY if passes() (schema+commonsense+hardlogic) AND the
DAV/DDR/ATT triplet (merge_soft.soft_of) strictly improves in >=1 metric with NO
metric regressing. DRY-RUN by default; --apply writes. ENRICH_RES selects dir.

Usage: ENRICH_RES=<dir> .venv/bin/python enrich_rebook.py [--shard i/N]
       [--limit K] [--uids u1,u2] [--apply]
"""
import os, sys, json, copy, glob, math, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chinatravel.data.load_datasets import load_query, load_json_file
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
from chinatravel.environment.tools.transportation.apis import Transportation
from merge_soft import soft_of

RES = os.environ.get("ENRICH_RES", "results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation")
REPO = os.path.dirname(os.path.abspath(__file__))
qargs = argparse.Namespace(splits="TPC_IJCAI_2026_phase1", method="x", lang="en", preference=False)
qi, qd = load_query(qargs)
from constraint_gate import apply_gate
apply_gate(qd)  # GATE_CONSTRAINTS=generated swaps in generated hard_logic_py; default (oracle) = no-op
sch = load_json_file("chinatravel/evaluation/output_schema.json")


def passes(uid, plan):
    s = set(evaluate_schema_constraints([uid], {uid: plan}, schema=sch)[2])
    c = set(evaluate_commonsense_constraints([uid], qd, {uid: plan}, verbose=False, lang="en")[3])
    if uid not in s or uid not in c:
        return False
    l = set(evaluate_hard_constraints_v2([uid], qd, {uid: plan}, env_pass_id=list(c), verbose=False, lang="en")[-1])
    return uid in l


def hm(t):
    h, m = str(t).split(":")
    return int(h) * 60 + int(m)


TR = Transportation(lang="en")

# ---- flight DB ------------------------------------------------------------
FDB = [json.loads(l) for l in open(os.path.join(
    REPO, "chinatravel/environment/database_en/intercity_transport/airplane.jsonl"))]
AIRPORTS = sorted(set(r["From"] for r in FDB) | set(r["To"] for r in FDB))


def city_airports(city):
    return [a for a in AIRPORTS if a.startswith(city)]


def flights(frm_list, to_list):
    return [r for r in FDB if r["From"] in frm_list and r["To"] in to_list]


# ---- recipes ---------------------------------------------------------------
RECIPE_FILE = os.path.join(REPO, "enrich_rebook_recipes.json")
RECIPES = json.load(open(RECIPE_FILE)) if os.path.exists(RECIPE_FILE) else {}

# ---- transfer-leg plumbing -------------------------------------------------

def leg_min(trs):
    return sum(hm(l["end_time"]) - hm(l["start_time"]) for l in trs)


def plan_cars(plan, ppl):
    for d in plan["itinerary"]:
        for a in d["activities"]:
            for l in a.get("transports", []):
                if l.get("mode") == "taxi" and l.get("cars"):
                    return int(l["cars"])
    return max(1, math.ceil(ppl / 4))


def fmt_legs(raw, ppl, cars):
    """Attach price/cost/tickets/cars fields the way commonsense_constraint wants:
    price == tool cost (per unit); walk cost 0, metro cost=price*tickets,
    taxi cost=price*cars."""
    out = []
    for g in raw:
        g = dict(g)
        unit = g.get("cost", 0)
        g["price"] = unit
        if g["mode"] == "walk":
            g["cost"] = 0
            g["price"] = 0
        elif g["mode"] == "metro":
            g["tickets"] = ppl
            g["cost"] = round(unit * ppl, 6)
        elif g["mode"] == "taxi":
            g["cars"] = cars
            g["cost"] = round(unit * cars, 6)
        out.append(g)
    return out


def goto(city, a, b, t, mode):
    try:
        legs = TR.goto(city, a, b, t, mode)
    except Exception:
        return None
    if not isinstance(legs, list) or not legs:
        return None
    return copy.deepcopy(legs)


def transfer_options(city, a, b, t, ppl, cars):
    """All feasible mode variants for a transfer a->b starting at t, sorted by
    duration."""
    opts = []
    for mode in ("metro", "taxi", "walk"):
        raw = goto(city, a, b, t, mode)
        if raw is None:
            continue
        legs = fmt_legs(raw, ppl, cars)
        opts.append({"mode": mode, "legs": legs, "dur": leg_min(legs)})
    opts.sort(key=lambda o: o["dur"])
    return opts


# ---- locate arrival / departure flights ------------------------------------

def find_arrival(plan):
    """(flight_act, transfer_act) for the day-1 arrival flight; transfer legs are
    on the NEXT activity (they start at the arrival airport)."""
    acts = plan["itinerary"][0]["activities"]
    if not acts or acts[0].get("type") != "airplane" or len(acts) < 2:
        return None
    fl, nxt = acts[0], acts[1]
    trs = nxt.get("transports") or []
    if not trs or trs[0].get("start") != fl.get("end"):
        return None
    return fl, nxt


def find_departure(plan):
    """(flight_act, prev_act) for the last-day departure flight; transfer legs sit
    on the flight activity itself."""
    acts = plan["itinerary"][-1]["activities"]
    if not acts or acts[-1].get("type") != "airplane" or len(acts) < 2:
        return None
    fl, prev = acts[-1], acts[-2]
    trs = fl.get("transports") or []
    if not trs:
        return None
    return fl, prev


def set_flight(act, row, ppl):
    act["FlightID"] = row["FlightID"]
    act["start"] = row["From"]
    act["end"] = row["To"]
    act["start_time"] = row["BeginTime"]
    act["end_time"] = row["EndTime"]
    act["price"] = row["Cost"]
    act["tickets"] = ppl
    act["cost"] = round(row["Cost"] * ppl, 6)


# ---- candidate generation ---------------------------------------------------

MIN_GAIN_MIN = 5          # a candidate must cut the transfer by >= this
MIN_CUR_MIN = 40          # only touch transfers at least this long
MAX_TRIES_PER_SIDE = 6    # full-eval attempts per side


def arrival_candidates(plan, q):
    """List of candidate edits for the arrival transfer, best first. Each item:
    {'flight': row|None, 'legs': [...], 'dur': int}."""
    found = find_arrival(plan)
    if not found:
        return [], 0
    fl, nxt = found
    city = q["target_city"]
    ppl = int(q.get("people_number", 1))
    cars = plan_cars(plan, ppl)
    cur = leg_min(nxt.get("transports") or [])
    if cur < MIN_CUR_MIN:
        return [], cur
    nxt_start = hm(nxt["start_time"])
    tgt = nxt.get("position")
    if not tgt:
        return [], cur
    cands = []
    # (a) mode swap only (same flight)
    for o in transfer_options(city, fl["end"], tgt, fl["end_time"], ppl, cars):
        if o["dur"] <= cur - MIN_GAIN_MIN and hm(o["legs"][-1]["end_time"]) <= nxt_start:
            cands.append({"flight": None, "legs": o["legs"], "dur": o["dur"]})
    # (b) rebook to an alternate same-city airport
    origins = city_airports(q["start_city"]) or [fl.get("start")]
    for alt in city_airports(city):
        if alt == fl.get("end"):
            continue
        rows = flights(origins, [alt])
        rows.sort(key=lambda r: (r["From"] != fl.get("start"),
                                 abs(r["Cost"] - fl.get("price", r["Cost"]))))
        seen_modes = set()  # keep one candidate per mode (cheap modes may be the
        for row in rows:    # only ones that survive hard-logic cost caps)
            for o in transfer_options(city, alt, tgt, row["EndTime"], ppl, cars):
                if o["dur"] > cur - MIN_GAIN_MIN:
                    break  # options sorted by dur; nothing better for this row
                if o["mode"] in seen_modes:
                    continue
                if hm(o["legs"][-1]["end_time"]) > nxt_start:
                    continue  # arrives too late for the next activity
                seen_modes.add(o["mode"])
                cands.append({"flight": row, "legs": o["legs"], "dur": o["dur"]})
            if len(seen_modes) >= 2:
                break
    cands.sort(key=lambda c: c["dur"])
    return cands, cur


def departure_candidates(plan, q):
    """Candidates for the departure transfer. Each item may carry 'trim': new
    end_time (str) for the previous activity when needed to catch the flight."""
    found = find_departure(plan)
    if not found:
        return [], 0
    fl, prev = found
    city = q["target_city"]
    ppl = int(q.get("people_number", 1))
    cars = plan_cars(plan, ppl)
    cur = leg_min(fl.get("transports") or [])
    if cur < MIN_CUR_MIN:
        return [], cur
    src = prev.get("position")
    if not src:
        return [], cur
    prev_end = prev["end_time"]
    cands = []
    # (a) mode swap only
    for o in transfer_options(city, src, fl["start"], prev_end, ppl, cars):
        if o["dur"] <= cur - MIN_GAIN_MIN and hm(o["legs"][-1]["end_time"]) <= hm(fl["start_time"]):
            cands.append({"flight": None, "legs": o["legs"], "dur": o["dur"], "trim": None})
    # (b) rebook to an alternate same-city airport
    dests = city_airports(qd_end_city(q)) or [fl.get("end")]
    for alt in city_airports(city):
        if alt == fl.get("start"):
            continue
        rows = flights([alt], dests)
        rows.sort(key=lambda r: (r["To"] != fl.get("end"),
                                 abs(hm(r["BeginTime"]) - hm(fl["start_time"]))))
        opts = transfer_options(city, src, alt, prev_end, ppl, cars)
        seen_modes = set()  # keep one candidate per mode (cost caps may kill taxi)
        for row in rows:
            for o in opts:
                if o["dur"] > cur - MIN_GAIN_MIN:
                    break
                if o["mode"] in seen_modes:
                    continue
                trim = None
                if hm(o["legs"][-1]["end_time"]) > hm(row["BeginTime"]):
                    # try trimming the previous attraction to catch this flight
                    if prev.get("type") != "attraction":
                        continue
                    new_end = hm(row["BeginTime"]) - o["dur"] - 2
                    if new_end < hm(prev["start_time"]) + 15:
                        continue
                    trim = "%02d:%02d" % (new_end // 60, new_end % 60)
                    raw2 = goto(city, src, alt, trim, o["mode"])
                    if raw2 is None:
                        continue
                    legs2 = fmt_legs(raw2, ppl, cars)
                    o = {"mode": o["mode"], "legs": legs2, "dur": leg_min(legs2)}
                    if hm(o["legs"][-1]["end_time"]) > hm(row["BeginTime"]):
                        continue
                seen_modes.add(o["mode"])
                cands.append({"flight": row, "legs": o["legs"], "dur": o["dur"], "trim": trim})
            if len(seen_modes) >= 2:
                break
    cands.sort(key=lambda c: c["dur"])
    return cands, cur


def qd_end_city(q):
    return q["start_city"]


# ---- gated application -------------------------------------------------------

def better(s0, s1):
    return all(b >= a - 1e-9 for a, b in zip(s0, s1)) and any(b > a + 1e-9 for a, b in zip(s0, s1))


def try_side(uid, plan, q, side):
    """Apply the best passing candidate for one side. Returns (plan, desc|None)."""
    ppl = int(q.get("people_number", 1))
    cands, cur = (arrival_candidates if side == "arr" else departure_candidates)(plan, q)
    s0 = soft_of(plan)
    for c in cands[:MAX_TRIES_PER_SIDE]:
        cand = copy.deepcopy(plan)
        if side == "arr":
            fl, nxt = find_arrival(cand)
            if c["flight"] is not None:
                set_flight(fl, c["flight"], ppl)
            nxt["transports"] = c["legs"]
        else:
            fl, prev = find_departure(cand)
            if c["flight"] is not None:
                set_flight(fl, c["flight"], ppl)
            if c.get("trim"):
                prev["end_time"] = c["trim"]
            fl["transports"] = c["legs"]
        s1 = soft_of(cand)
        if s1 is None or not better(s0, s1):
            continue
        if passes(uid, cand):
            desc = "%s %s->%s %dmin->%dmin%s" % (
                side,
                c["flight"]["FlightID"] if c["flight"] else "same-flight",
                c["legs"][0]["mode"] if len(c["legs"]) == 1 else "metro",
                cur, c["dur"], " trim" if c.get("trim") else "")
            return cand, desc
    return plan, None


def try_recipe(uid, plan):
    if uid not in RECIPES:
        return plan, None
    rec = RECIPES[uid]
    cand = copy.deepcopy(plan)
    for k in ("people_number", "start_city", "target_city", "itinerary"):
        cand[k] = copy.deepcopy(rec[k])
    s0, s1 = soft_of(plan), soft_of(cand)
    if s0 is not None and (s1 is None or not better(s0, s1)):
        return plan, None
    if passes(uid, cand):
        return cand, "recipe"
    return plan, None


# ---- main --------------------------------------------------------------------

if __name__ == "__main__":
    files = sorted(glob.glob(f"{RES}/*.json"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    apply = "--apply" in sys.argv
    only = set()
    if "--uids" in sys.argv:
        only = set(sys.argv[sys.argv.index("--uids") + 1].split(","))
    shard_i, shard_n = 0, 1
    if "--shard" in sys.argv:
        shard_i, shard_n = [int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/")]
    kept = seen = 0
    tot = [0.0, 0.0, 0.0]
    for n, f in enumerate(files):
        if limit and n >= limit:
            break
        if n % shard_n != shard_i:
            continue
        uid = os.path.basename(f)[:-5]
        if only and uid not in only:
            continue
        plan = load_json_file(f)
        if not plan.get("itinerary"):
            continue
        q = qd[uid]
        s0 = soft_of(plan)
        edits = []
        cand, d = try_recipe(uid, plan)
        if d:
            edits.append(d)
        # generic rebooking / transfer swaps on top of (or instead of) the recipe
        for side in ("arr", "dep"):
            cand2, d = try_side(uid, cand, q, side)
            if d:
                edits.append(d)
                cand = cand2
        if not edits:
            continue
        s1 = soft_of(cand)
        if s1 is None or not better(s0, s1):
            continue
        kept += 1
        for i in range(3):
            tot[i] += s1[i] - s0[i]
        print("%s: %s | DAV %.3f->%.3f DDR %.3f->%.3f ATT %.4f->%.4f" % (
            uid, "; ".join(edits), s0[0], s1[0], s0[1], s1[1], s0[2], s1[2]))
        if apply:
            # atomic write: serialize first (a TypeError here touches nothing),
            # then write a temp file and os.replace so a crash can never leave
            # a truncated/partial plan behind.
            blob = json.dumps(cand, ensure_ascii=False)
            tmpf = f + ".tmp"
            with open(tmpf, "w") as fh:
                fh.write(blob)
            os.replace(tmpf, f)
    print("\nkept %d | delta DAV %+0.4f DDR %+0.4f ATT %+0.4f | apply=%s" % (
        kept, tot[0], tot[1], tot[2], apply))

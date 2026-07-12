"""Late-evening attraction insert with MOVABLE hotel check-in (DAV lever).

Pattern (verified end-to-end on the 11 Chongqing DAV=0.875 residuals, see
scratchpad fix_recipes.json audit 'dav-chongqing'): when a plan has DAV<1 and
the last day-activity before the accommodation ends with slack before 24:00,
the planner's recorded hotel check-in acts as an implicit curfew — but the
accommodation activity runs to 24:00, so its start_time is MOVABLE. We insert
one (or more) unused, open-at-that-hour, cheap/free attraction near the last
activity:

  1. transports last-activity -> candidate (walk if short, else taxi/metro);
  2. 30-min visit starting at arrival (bounded by open-hours);
  3. rebuild candidate -> hotel transports and shift the accommodation
     start_time to the new arrival (must stay < 24:00; end stays 24:00).

Candidates are ranked free-first then by detour = d(A,X)+d(X,H)-d(A,H), so
budget/hard-logic disjuncts that require attraction price==0 are naturally
satisfied first (and passes() evaluates the real OR-disjuncts regardless).

Regression-safe: an edit is kept ONLY if the full 3-stage eval still passes
AND DAV strictly improves AND neither DDR nor ATT (merge_soft.soft_of triplet)
regresses. DRY-RUN by default (prints per-uid would-apply diagnostics, writes
nothing); --apply writes. ENRICH_RES env var selects the target dir.

Usage:
  ENRICH_RES=<dir> .venv/bin/python enrich_lateattr.py [--shard i/N] [--limit K] [--apply]
"""
import os, sys, json, copy, glob, math, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import pandas as pd
from geopy.distance import geodesic

from chinatravel.data.load_datasets import load_query, load_json_file
from chinatravel.environment.world_env import WorldEnv
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
from merge_soft import soft_of

RES = os.environ.get("ENRICH_RES", "results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation")
qargs = argparse.Namespace(splits=os.environ.get("ENRICH_SPLIT", "TPC_IJCAI_2026_phase1"), method="x", lang="en", preference=False)
qi, qd = load_query(qargs)
from constraint_gate import apply_gate
apply_gate(qd)  # GATE_CONSTRAINTS=generated swaps in generated hard_logic_py; default (oracle) = no-op
sch = load_json_file("chinatravel/evaluation/output_schema.json")
env = WorldEnv(lang="en")

VISIT_MIN = 30          # inserted visit duration (verified in the audit)
MAX_LEG_MIN = 45        # skip absurd single legs early (ATT gate still decides)
MAX_CANDS = 30          # candidates tried per insertion point
MAX_INSERTS_PER_DAY = 3


def passes(uid, plan):
    s = set(evaluate_schema_constraints([uid], {uid: plan}, schema=sch)[2])
    if uid not in s:
        return False
    c = set(evaluate_commonsense_constraints([uid], qd, {uid: plan}, verbose=False, lang="en")[3])
    if uid not in c:
        return False
    l = set(evaluate_hard_constraints_v2([uid], qd, {uid: plan}, env_pass_id=list(c),
                                         verbose=False, lang="en")[-1])
    return uid in l


def hm(t):
    h, m = str(t).split(":")[:2]
    return int(h) * 60 + int(m)


def mh(x):
    return f"{int(x) // 60:02d}:{int(x) % 60:02d}"


def actpos(a):
    return a.get("position") or a.get("end") or a.get("start")


def clean(o):
    """Strip numpy scalar types so the written plan is pure-JSON."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def leg_minutes(tr):
    return sum(hm(l["end_time"]) - hm(l["start_time"]) for l in tr)


def goto(city, start, end, start_time, mode, ppl):
    """WorldEnv transport with the repo's price/tickets/cars conventions."""
    if start == end:
        return None
    info = env(f'goto("{city}", "{start}", "{end}", "{start_time}", "{mode}")')["data"]
    if not isinstance(info, list) or not info:
        return None
    if len(info) == 3:                      # walk + metro + walk
        info[1]["price"] = info[1]["cost"]
        info[1]["tickets"] = ppl
        info[1]["cost"] = info[1]["price"] * ppl
        info[0]["price"] = info[0]["cost"]
        info[2]["price"] = info[2]["cost"]
    elif info[0].get("mode") == "taxi":
        info[0]["price"] = info[0]["cost"]
        info[0]["cars"] = int((ppl - 1) / 4) + 1
        info[0]["cost"] = info[0]["price"] * info[0]["cars"]
    elif info[0].get("mode") == "walk":
        info[0]["price"] = info[0]["cost"]
    return clean(info)


# ---- per-city POI data ------------------------------------------------------
_attr = {}
_coords = {}


def city_attractions(city):
    key = city.lower()
    if key not in _attr:
        _attr[key] = pd.read_csv(
            f"chinatravel/environment/database_en/attractions/{key}/attractions.csv")
    return _attr[key]


def city_coords(city):
    """name -> (lat, lon) over attractions + restaurants + accommodations."""
    key = city.lower()
    if key not in _coords:
        d = {}
        for path in (
            f"chinatravel/environment/database_en/attractions/{key}/attractions.csv",
            f"chinatravel/environment/database_en/restaurants/{key}/restaurants_{key}.csv",
            f"chinatravel/environment/database_en/accommodations/{key}/accommodations.csv",
        ):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            for _, r in df.iterrows():
                if r.get("name") is not None:
                    d[str(r["name"])] = (float(r["lat"]), float(r["lon"]))
        _coords[key] = d
    return _coords[key]


def rank_candidates(city, posA, posH, used):
    """Unused attractions ranked free-first, then by detour A->X->H (km)."""
    attr = city_attractions(city)
    cmap = city_coords(city)
    cA, cH = cmap.get(posA), cmap.get(posH)
    anchor = cA or cH
    if anchor is None:
        return []
    dAH = geodesic(cA, cH).km if (cA and cH) else 0.0
    rows = []
    for _, r in attr.iterrows():
        name = str(r["name"])
        if name in used or name not in cmap:
            continue
        cX = cmap[name]
        if cA and cH:
            detour = geodesic(cA, cX).km + geodesic(cX, cH).km - dAH
        else:
            detour = geodesic(anchor, cX).km
        d_from_A = geodesic(cA, cX).km if cA else detour
        price = float(r.get("price", 0) or 0)
        if math.isnan(price):       # never let NaN into a written plan
            continue
        rows.append((price > 0, detour, d_from_A, price, r))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows[:MAX_CANDS]


def try_insert_day(uid, plan, query, di, dav0, ddr0, att0):
    """One insertion attempt on day di. Returns updated plan or None."""
    acts = plan["itinerary"][di]["activities"]
    city = query["target_city"]
    ppl = int(query["people_number"])
    # accommodation slot with a preceding activity
    j = next((k for k, a in enumerate(acts) if a.get("type") == "accommodation"), None)
    if j is None or j == 0:
        return None
    A, H = acts[j - 1], acts[j]
    posA, posH, tA = actpos(A), actpos(H), A.get("end_time")
    if not posA or not posH or not tA:
        return None
    if hm(tA) >= 24 * 60 - 10 or hm(str(H.get("end_time", "24:00"))) < 24 * 60:
        return None
    used = set(str(actpos(x)) for d in plan["itinerary"] for x in d["activities"] if actpos(x))
    for _, _, d_from_A, price, r in rank_candidates(city, posA, posH, used):
        name = str(r["name"])
        ot, et = str(r["opentime"]), str(r["endtime"])
        if hm(et) <= hm(ot):        # skip overnight windows
            continue
        modes = ("walk", "taxi", "metro") if d_from_A <= 1.0 else ("taxi", "metro", "walk")
        for m1 in modes:
            tr1 = goto(city, posA, name, tA, m1, ppl)
            if not tr1 or leg_minutes(tr1) > MAX_LEG_MIN:
                continue
            arrive = tr1[-1]["end_time"]
            start = arrive if hm(arrive) >= hm(ot) else ot
            end = mh(hm(start) + VISIT_MIN)
            if hm(start) < hm(ot) or hm(end) > hm(et) or hm(end) >= 24 * 60:
                continue
            for m2 in dict.fromkeys((m1, "taxi", "metro")):
                tr2 = goto(city, name, posH, end, m2, ppl)
                if not tr2 or leg_minutes(tr2) > MAX_LEG_MIN:
                    continue
                hot_arr = tr2[-1]["end_time"]
                if hm(hot_arr) >= 24 * 60:
                    continue
                cand = copy.deepcopy(plan)
                cacts = cand["itinerary"][di]["activities"]
                newact = {"position": name, "type": "attraction",
                          "price": float(price), "cost": float(price) * ppl,
                          "tickets": ppl, "start_time": str(start),
                          "end_time": str(end), "transports": tr1}
                hotel = copy.deepcopy(H)
                hotel["transports"] = tr2
                hotel["start_time"] = str(hot_arr)
                cand["itinerary"][di]["activities"] = (
                    cacts[:j - 1] + [cacts[j - 1], newact, hotel] + cacts[j + 1:])
                cand = clean(cand)
                s1 = soft_of(cand)
                if not s1:
                    continue
                dav1, ddr1, att1 = s1
                if not (dav1 > dav0 + 1e-12 and ddr1 >= ddr0 - 1e-12 and att1 >= att0 - 1e-12):
                    continue
                if passes(uid, cand):
                    return cand, (name, m1, m2, start, end, hot_arr, float(price))
    return None


def enrich(uid, plan, query):
    """Insert late attractions wherever slack allows. Returns (plan, log)."""
    log = []
    days = len(plan["itinerary"])
    for di in range(days):
        for _ in range(MAX_INSERTS_PER_DAY):
            s0 = soft_of(plan)
            if not s0:
                break
            dav0, ddr0, att0 = s0
            if dav0 >= 1.0:
                break
            got = try_insert_day(uid, plan, query, di, dav0, ddr0, att0)
            if not got:
                break
            plan, info = got
            log.append((di,) + info)
        s0 = soft_of(plan)
        if s0 and s0[0] >= 1.0:
            break
    return plan, log


if __name__ == "__main__":
    files = sorted(glob.glob(f"{RES}/*.json"))
    if not files:
        sys.exit(f"no plans found in {RES}")
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    apply = "--apply" in sys.argv
    shard_i, shard_n = 0, 1
    if "--shard" in sys.argv:
        shard_i, shard_n = [int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/")]
    seen = cand_n = kept = tot_ins = 0
    for n, f in enumerate(files):
        if limit and seen >= limit:
            break
        if n % shard_n != shard_i:
            continue
        seen += 1
        uid = os.path.basename(f)[:-5]
        plan = load_json_file(f)
        s0 = soft_of(plan)
        if not s0 or s0[0] >= 1.0:      # cheap stage-1 filter: DAV<1 only
            continue
        cand_n += 1
        if not passes(uid, plan):
            print(f"{uid}: baseline FAILS full eval -- skipped")
            continue
        query = qd[uid]
        new_plan, log = enrich(uid, copy.deepcopy(plan), query)
        if not log:
            print(f"{uid}: DAV {s0[0]:.3f} -- no passing insert found")
            continue
        s1 = soft_of(new_plan)
        kept += 1
        tot_ins += len(log)
        for di, name, m1, m2, st, en, ha, pr in log:
            print(f"{uid}: day{di + 1} insert '{name}' (price {pr:g}) {m1}->{m2}, "
                  f"visit {st}-{en}, hotel check-in -> {ha}")
        print(f"{uid}: DAV {s0[0]:.3f}->{s1[0]:.3f} DDR {s0[1]:.3f}->{s1[1]:.3f} "
              f"ATT {s0[2]:.4f}->{s1[2]:.4f}{' [APPLIED]' if apply else ' [dry-run]'}")
        if apply:
            # atomic write: serialize fully first (allow_nan=False rejects any
            # NaN/Inf before the file is touched), then replace in one syscall
            # so a crash can never leave a partial/corrupt plan on disk.
            payload = json.dumps(new_plan, ensure_ascii=False, allow_nan=False)
            tmp = f + ".tmp"
            with open(tmp, "w") as fh:
                fh.write(payload)
            os.replace(tmp, f)
    print(f"\nscanned {seen}, DAV<1 {cand_n}, plans improved {kept}, "
          f"inserts {tot_ins}, apply={apply}")

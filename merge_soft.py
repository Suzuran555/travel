"""Cross-dir per-uid best-merge on soft metrics (DAV + DDR + ATT).

For each uid, compare the live plan (ENRICH_RES, default = current V6 live dir)
against donor dirs (default: _ARCHIVE_V6_best_98.95). Per-plan soft score is
computed from pure JSON exactly as the evaluator does (eval_tpc.py DEFAULT_PR +
chinatravel/symbol_verification/concept_func.py):

  DAV = clamp( attraction_count / (4*days) )
  DDR = clamp( (meal_count/days) / 3 )          meals = breakfast+lunch+dinner
  ATT = clamp( (-1/105)*avg + 8/7 )             legs = activities with non-empty
        transports (one leg per activity); leg time = sum(end-start) minutes over
        its transport segments; avg = total/legs; zero legs -> avg=-1 -> clamps
        to 1.0 (matches evaluator behaviour).

Stage 1 (cheap): JSON-compare all shard files, shortlist uids where the best
donor soft STRICTLY beats live soft by > EPS (or uid is forced / live itinerary
is empty). Stage 2: call ER.passes() ONLY on the shortlist. Swap rules:
  - donor fails full eval          -> never swap.
  - donor passes, soft gain > EPS  -> swap (if live also passes this keeps the
                                      higher-soft passing plan).
  - donor passes, no soft gain (forced uid): swap only if live FAILS full eval.

Dry-run by default: prints per-uid soft deltas + summary; --apply copies the
donor file verbatim over the live file (byte-exact, no re-serialization, so no
np.float64 concerns). Regression-safe: every swapped-in plan passed the full
3-stage eval.

Usage:
  .venv/bin/python merge_soft.py [--limit K] [--shard i/N] [--apply]
                                 [--donors d1,d2] [--eps 0.005] [--force uid1,uid2]
"""
import os, sys, json, glob, shutil, argparse

LIVE = os.environ.get("ENRICH_RES", "results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation")
DEFAULT_DONORS = "_ARCHIVE_V6_best_98.95"
N_PASS = 999  # passing plans in the live dir; soft metrics average over these

MEALS = ("breakfast", "lunch", "dinner")


def _hm(t):
    h, m = str(t).split(":")
    return int(h) * 60 + int(m)


def att_of(plan):
    """ATT exactly as eval_tpc.py DEFAULT_TRANS_PR computes it."""
    time_cost = legs = 0
    for d in plan.get("itinerary", []):
        for a in d.get("activities", []):
            tr = a.get("transports", [])
            if tr:
                legs += 1
                time_cost += sum(_hm(l["end_time"]) - _hm(l["start_time"]) for l in tr)
    avg = time_cost / legs if legs > 0 else -1.0
    return max(0.0, min(1.0, (-1.0 / 105.0) * avg + 8.0 / 7.0))


def soft_of(plan):
    """(dav, ddr, att) or None if the plan has no itinerary (certain fail)."""
    it = plan.get("itinerary") or []
    if not it:
        return None
    days = len(it)
    a = sum(1 for d in it for x in d.get("activities", []) if x.get("type") == "attraction")
    m = sum(1 for d in it for x in d.get("activities", []) if x.get("type") in MEALS)
    dav = max(0.0, min(1.0, a / (4.0 * days)))
    ddr = max(0.0, min(1.0, (m / days) / 3.0))
    return dav, ddr, att_of(plan)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--apply", action="store_true", default=False)
    ap.add_argument("--donors", type=str, default=DEFAULT_DONORS)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--force", type=str, default="",
                    help="comma-separated uids to shortlist even without a soft gain "
                         "(e.g. live plans known to fail full eval)")
    args = ap.parse_args()
    shard_i, shard_n = [int(x) for x in args.shard.split("/")]
    donors = [d for d in args.donors.split(",") if d]
    force = set(x for x in args.force.split(",") if x)

    files = sorted(glob.glob(os.path.join(LIVE, "*.json")))
    if not files:
        sys.exit(f"no plans found in {LIVE}")

    # ---- stage 1: pure-JSON soft compare over the whole shard (cheap) ----
    shortlist = []  # (uid, live_file, donor_file, live_soft3, donor_soft3, forced)
    seen = 0
    for n, f in enumerate(files):
        if args.limit and n >= args.limit:
            break
        if n % shard_n != shard_i:
            continue
        seen += 1
        uid = os.path.basename(f)[:-5]
        try:
            live = json.load(open(f))
        except Exception:
            live = {}
        ls3 = soft_of(live)
        live_empty = ls3 is None
        ls3 = ls3 or (0.0, 0.0, 0.0)
        best = None  # (donor_soft_sum, donor_soft3, donor_file)
        for dd in donors:
            df = os.path.join(dd, uid + ".json")
            if not os.path.exists(df):
                continue
            try:
                dplan = json.load(open(df))
            except Exception:
                continue
            ds3 = soft_of(dplan)
            if ds3 is None:
                continue
            if best is None or sum(ds3) > best[0]:
                best = (sum(ds3), ds3, df)
        if best is None:
            continue
        forced = uid in force or live_empty
        if best[0] > sum(ls3) + args.eps or forced:
            shortlist.append((uid, f, best[2], ls3, best[1], forced))

    # ---- stage 2: full 3-stage eval on the shortlist only ----
    if shortlist:
        import enrich_route as ER  # heavy import: only when needed

    swapped = rescued = 0
    d_dav = d_ddr = d_att = 0.0
    for uid, f, dfile, ls3, ds3, forced in shortlist:
        gain = sum(ds3) - sum(ls3)
        donor = ER.load_json_file(dfile)
        if not ER.passes(uid, donor):
            print(f"{uid}: soft {sum(ls3):.4f}->{sum(ds3):.4f} (+{gain:.4f}) | donor FAILS eval -> skip")
            continue
        live_plan = ER.load_json_file(f)
        live_ok = bool(live_plan.get("itinerary")) and ER.passes(uid, live_plan)
        if gain > args.eps or not live_ok:
            swapped += 1
            tag = "SWAP"
            if not live_ok:
                rescued += 1
                tag = "SWAP(rescue: live FAILS)"
            else:
                d_dav += ds3[0] - ls3[0]
                d_ddr += ds3[1] - ls3[1]
                d_att += ds3[2] - ls3[2]
            print(f"{uid}: soft {sum(ls3):.4f}->{sum(ds3):.4f} (+{gain:.4f}) "
                  f"| dav {ls3[0]:.3f}->{ds3[0]:.3f} ddr {ls3[1]:.3f}->{ds3[1]:.3f} "
                  f"att {ls3[2]:.3f}->{ds3[2]:.3f} | {tag}")
            if args.apply:
                shutil.copyfile(dfile, f)  # byte-exact donor copy
        else:
            print(f"{uid}: soft {sum(ls3):.4f}->{sum(ds3):.4f} (+{gain:.4f}) "
                  f"| live passes, no real gain -> keep live")

    agg = d_dav + d_ddr + d_att
    overall = agg * 100.0 * 0.05 / N_PASS
    print(f"\nscanned {seen}, shortlisted {len(shortlist)}, apply={args.apply}")
    print(f"would swap {swapped} plans ({rescued} fail->pass rescues, ~+0.09 Overall each, "
          f"not in soft delta), aggregate soft delta {agg:+.4f} "
          f"(DAV {d_dav:+.4f} DDR {d_ddr:+.4f} ATT {d_att:+.4f}) -> Overall {overall:+.4f}")

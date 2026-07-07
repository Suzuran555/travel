"""A/B: bias-only vs bias+compact_dwell (Patch 6). Measures DAV/DDR + pass rate
on a sample of multi-day trips. Reuses validate_bias helpers."""
import os, sys
sys.path.insert(0, os.path.abspath("."))
import validate_bias as V

if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 14
    # sample: multi-day trips with DAV<1 in current results (room to densify)
    import glob, json
    sample = []
    for f in sorted(glob.glob(f"{V.D}/*.json")):
        uid = os.path.basename(f)[:-5]; p = V.load_json_file(f)
        it = p.get("itinerary") or []
        if len(it) < 2:
            continue
        days = len(it); na = sum(1 for d in it for x in d["activities"] if x.get("type") == "attraction")
        if na / (4 * days) < 1.0:
            sample.append(uid)
        if len(sample) >= n:
            break
    print(f"sample ({len(sample)} DAV<1 multi-day): {sample}", flush=True)
    base_ag = V.make_agent({"enable_travelday_time_bias": True})
    pat_ag = V.make_agent({"enable_travelday_time_bias": True, "enable_compact_dwell": True})
    import statistics as st
    rows = []
    for uid in sample:
        q = V.qd[uid]
        try:
            _, bp = base_ag.run(q, prob_idx=uid, oralce_translation=True)
        except Exception as e:
            bp = {}
        V.sys.stdout = V.sys.__stdout__
        try:
            _, pp = pat_ag.run(q, prob_idx=uid, oralce_translation=True)
        except Exception as e:
            pp = {}
        V.sys.stdout = V.sys.__stdout__
        bp = V.clean(bp); pp = V.clean(pp)
        bpass = V.passes(uid, bp); ppass = V.passes(uid, pp)
        def dav(p):
            days = max(1, len(p.get("itinerary", [])))
            na = sum(1 for d in p["itinerary"] for x in d["activities"] if x.get("type") == "attraction")
            return min(1.0, na / (4 * days))
        def ddr(p):
            days = max(1, len(p.get("itinerary", [])))
            m = sum(1 for d in p["itinerary"] for x in d["activities"] if x.get("type") in ("breakfast", "lunch", "dinner"))
            return min(1.0, (m / days) / 3)
        bd, pdv = (dav(bp) if bpass else 0), (dav(pp) if ppass else 0)
        br, pr = (ddr(bp) if bpass else 0), (ddr(pp) if ppass else 0)
        rows.append((uid, bpass, ppass, bd, pdv, br, pr))
        print(f"{uid}: pass {bpass}->{ppass}  DAV {bd:.2f}->{pdv:.2f}  DDR {br:.2f}->{pr:.2f}", flush=True)
    both = [r for r in rows if r[1] and r[2]]
    print(f"\n=== SUMMARY over {len(rows)} ===")
    print(f"pass: base {sum(r[1] for r in rows)}/{len(rows)}  compact {sum(r[2] for r in rows)}/{len(rows)}")
    print(f"regressions (base pass, compact fail): {sum(1 for r in rows if r[1] and not r[2])}")
    if both:
        print(f"on {len(both)} both-pass: DAV {st.mean(r[3] for r in both)*100:.1f}->{st.mean(r[4] for r in both)*100:.1f}"
              f"  DDR {st.mean(r[5] for r in both)*100:.1f}->{st.mean(r[6] for r in both)*100:.1f}")

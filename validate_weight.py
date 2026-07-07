"""A/B: current bias weight 0.15 vs a higher weight. Does higher weight shift MORE
travel-day timing (more DDR/DAV) without more pass regressions?"""
import os, sys, statistics as st
sys.path.insert(0, os.path.abspath("."))
import validate_bias as V

W = float(os.environ.get("HIWEIGHT", "0.30"))
n = int(sys.argv[sys.argv.index("--n")+1]) if "--n" in sys.argv else 20
sample = V.pick_sample(n)
print(f"sample ({len(sample)}): weight 0.15 vs {W}", flush=True)
lo = V.make_agent({"enable_travelday_time_bias": True, "travelday_arr_weight":0.15, "travelday_dep_weight":0.15})
hi = V.make_agent({"enable_travelday_time_bias": True, "travelday_arr_weight":W, "travelday_dep_weight":W})
def ddr(p):
    d=max(1,len(p.get("itinerary",[]))); m=sum(1 for dd in p["itinerary"] for x in dd["activities"] if x.get("type") in ("breakfast","lunch","dinner")); return min(1.0,m/d/3)
def dav(p):
    d=max(1,len(p.get("itinerary",[]))); a=sum(1 for dd in p["itinerary"] for x in dd["activities"] if x.get("type")=="attraction"); return min(1.0,a/(4*d))
rows=[]
for uid in sample:
    q=V.qd[uid]
    try: _,lp=lo.run(q,prob_idx=uid,oralce_translation=True)
    except: lp={}
    V.sys.stdout=V.sys.__stdout__
    try: _,hp=hi.run(q,prob_idx=uid,oralce_translation=True)
    except: hp={}
    V.sys.stdout=V.sys.__stdout__
    lp=V.clean(lp); hp=V.clean(hp)
    lpass=V.passes(uid,lp); hpass=V.passes(uid,hp)
    la,ld=V.intercity_times(lp); ha,hd=V.intercity_times(hp)
    rows.append((uid,lpass,hpass,ld,hd,ddr(lp) if lpass else 0,ddr(hp) if hpass else 0,dav(lp) if lpass else 0,dav(hp) if hpass else 0))
    print(f"{uid}: pass {lpass}->{hpass} dep {ld}->{hd} DDR {rows[-1][5]:.2f}->{rows[-1][6]:.2f} DAV {rows[-1][7]:.2f}->{rows[-1][8]:.2f}", flush=True)
both=[r for r in rows if r[1] and r[2]]
print(f"\n=== SUMMARY {len(rows)} ===")
print(f"pass: lo {sum(r[1] for r in rows)}  hi {sum(r[2] for r in rows)}  regressions(lo pass,hi fail): {sum(1 for r in rows if r[1] and not r[2])}")
print(f"dep moved later (hi vs lo): {sum(1 for r in rows if r[3] and r[4] and r[4]>r[3])}")
if both: print(f"both-pass: DDR {st.mean(r[5] for r in both)*100:.1f}->{st.mean(r[6] for r in both)*100:.1f}  DAV {st.mean(r[7] for r in both)*100:.1f}->{st.mean(r[8] for r in both)*100:.1f}")

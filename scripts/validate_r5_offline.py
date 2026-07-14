"""Round-5 offline validation over the full-1000 cached translations.

For every uid in cache/translation_Qwen3.6-27B_reflect that has a plan in
results/full1000_sweep_en:

  OLD pipeline (what the honest sweep ran):
      cached hard_logic_py -> canonicalize_query_hard_logic
  NEW pipeline (round-5):
      cached hard_logic_py -> enforce_coverage(lang=None)
                           -> canonicalize_query_hard_logic

Both constraint lists are executed against the uid's EMITTED sweep plan via
the official evaluate_constraints_py.  Report:

  * per-rule fire counts, split by the uid's sweep outcome (pass/fail against
    oracle constraints, from the r5 oracle score dump);
  * REGRESSION check: currently-passing uids where the OLD generated list
    fully passes on its own plan but the NEW list does not (must be empty or
    individually justified);
  * EXPECTED-FLIP census: currently-failing uids whose constraint list
    changed (the fixes are exactly the autopsied defect categories).

Usage:  .venv/bin/python scripts/validate_r5_offline.py [--dump OUT.json]
"""
import argparse
import copy
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CACHE = os.path.join(ROOT, "cache", "translation_Qwen3.6-27B_reflect")
PLANS = os.path.join(ROOT, "results", "full1000_sweep_en")
ORACLE_SCORE = (
    "/private/tmp/claude-501/-Users-zhanggangyi-Desktop-TPC2026/"
    "ccadc8f1-a9ae-4981-8039-90c33e359ae2/scratchpad/r5_oracle_full1000.json"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    from chinatravel.agent.nesy_agent.constraint_coverage import enforce_coverage
    from chinatravel.agent.UrbanTrip.dsl_canonicalizer import (
        canonicalize_query_hard_logic,
    )
    from chinatravel.symbol_verification.hard_constraint import (
        evaluate_constraints_py,
    )

    fail_uids = set()
    if os.path.exists(ORACLE_SCORE):
        fail_uids = set(json.load(open(ORACLE_SCORE))["fail"])

    uids = sorted(
        f[:-5]
        for f in os.listdir(PLANS)
        if f.endswith(".json") and os.path.exists(os.path.join(CACHE, f))
    )
    print(f"{len(uids)} uids with cached translation + sweep plan "
          f"({len(fail_uids)} known sweep fails)")

    rule_fires = Counter()
    rule_fires_by_outcome = defaultdict(Counter)
    changed = {"pass": [], "fail": []}
    regressions = []
    details = {}

    for n, uid in enumerate(uids, 1):
        cached = json.load(open(os.path.join(CACHE, uid + ".json")))
        plan = json.load(open(os.path.join(PLANS, uid + ".json")))
        outcome = "fail" if uid in fail_uids else "pass"

        old_q = copy.deepcopy(cached)
        old_q = canonicalize_query_hard_logic(old_q)
        old_list = [c for c in old_q.get("hard_logic_py") or []
                    if isinstance(c, str)]

        new_q = copy.deepcopy(cached)
        pre_fixes = len(new_q.get("coverage_fixes") or [])
        new_q = enforce_coverage(new_q, lang=None)
        new_q = canonicalize_query_hard_logic(new_q)
        new_list = [c for c in new_q.get("hard_logic_py") or []
                    if isinstance(c, str)]
        fired = [f["rule"] for f in (new_q.get("coverage_fixes") or [])[pre_fixes:]]

        for r in fired:
            rule_fires[r] += 1
            rule_fires_by_outcome[outcome][r] += 1

        if sorted(old_list) != sorted(new_list):
            changed[outcome].append(uid)
            old_res = evaluate_constraints_py(old_list, plan)
            new_res = evaluate_constraints_py(new_list, plan)
            old_ok = all(old_res) if old_list else True
            new_ok = all(new_res) if new_list else True
            details[uid] = {
                "outcome": outcome,
                "rules": fired,
                "old_all_pass_on_plan": old_ok,
                "new_all_pass_on_plan": new_ok,
                "new_failing": [
                    c for c, r in zip(new_list, new_res) if not r
                ],
                "removed": [c for c in old_list if c not in new_list],
                "added": [c for c in new_list if c not in old_list],
            }
            if outcome == "pass" and old_ok and not new_ok:
                regressions.append(uid)

        if n % 200 == 0:
            print(f"  ... {n}/{len(uids)}")

    print("\n=== rule fire counts (all 1000) ===")
    for r, c in rule_fires.most_common():
        print(f"  {c:4d}  {r}   [on fails: "
              f"{rule_fires_by_outcome['fail'][r]}, on passes: "
              f"{rule_fires_by_outcome['pass'][r]}]")
    print(f"\nconstraint lists changed: {len(changed['fail'])} of "
          f"{len(fail_uids)} failing uids, {len(changed['pass'])} of "
          f"{len(uids) - len(fail_uids)} passing uids")
    print(f"\n=== REGRESSIONS (passing uid, old list clean on own plan, "
          f"new list fails on own plan): {len(regressions)} ===")
    for uid in regressions:
        print(f"  {uid}: rules={details[uid]['rules']}")
        for c in details[uid]["new_failing"]:
            print(f"      FAILS: {c[:160]!r}")
    print(f"\n=== expected-flip candidates (failing uids with changed "
          f"constraints): {len(changed['fail'])} ===")
    print(sorted(changed["fail"]))

    if args.dump:
        json.dump(
            {
                "rule_fires": dict(rule_fires),
                "rule_fires_by_outcome": {
                    k: dict(v) for k, v in rule_fires_by_outcome.items()
                },
                "changed_fail": changed["fail"],
                "changed_pass": changed["pass"],
                "regressions": regressions,
                "details": details,
            },
            open(args.dump, "w"),
            ensure_ascii=False,
            indent=1,
        )
        print(f"\ndumped -> {args.dump}")


if __name__ == "__main__":
    main()

"""Semantic DSL comparison: execute generated vs oracle constraints on the
real (ceiling) plan for each uid and compare verdicts.

String-matching (compare_dsl.py) undercounts badly: the LLM writes
semantically identical constraints with different variable names / list
orders. What matters functionally is verdict agreement when executed by the
official evaluator machinery on an actual plan.

Usage: .venv/bin/python compare_dsl_semantic.py [Qwen3-8B]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chinatravel.symbol_verification.hard_constraint import (
    evaluate_constraints_py,
    _set_tool_lang,
)

LIVE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "UrbanTripOptimizedV6_TPCLLM_en_oracletranslation",
)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-8B"
    cache = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cache",
        f"translation_{name}_reflect",
    )
    _set_tool_lang("en")
    files = sorted(glob.glob(os.path.join(cache, "*.json")))
    if not files:
        sys.exit(f"no cached translations in {cache}")

    agree_all = 0
    n = 0
    details = []
    for f in files:
        uid = os.path.basename(f)[:-5]
        plan_file = os.path.join(LIVE, f"{uid}.json")
        if not os.path.exists(plan_file):
            continue
        d = json.load(open(f))
        gen = d.get("hard_logic_py") or []
        ora = d.get("_oracle_hard_logic_py") or []
        if isinstance(gen, dict):
            gen = list(gen.values())
        plan = json.load(open(plan_file))
        try:
            vg = list(evaluate_constraints_py(gen, plan))
        except Exception as exc:  # noqa: BLE001 - a crashing constraint is a fail
            vg = [f"CRASH: {exc}"]
        try:
            vo = list(evaluate_constraints_py(ora, plan))
        except Exception as exc:  # noqa: BLE001
            vo = [f"CRASH: {exc}"]
        # empty generation = extraction failure, never "agreement"
        ok = (
            len(vg) > 0
            and all(v is True for v in vg)
            and all(v is True for v in vo)
        )
        n += 1
        agree_all += ok
        details.append((uid, ok, sum(1 for v in vg if v is True), len(vg),
                        sum(1 for v in vo if v is True), len(vo)))

    print(f"uids: {n}")
    print(f"full verdict agreement on real plan: {agree_all}/{n} = {agree_all/max(n,1):.1%}")
    print("\nper-uid (gen pass/total vs oracle pass/total):")
    for uid, ok, gp, gt, op, ot in details:
        flag = "OK " if ok else "MISS"
        print(f"  {flag} {uid}: gen {gp}/{gt}  oracle {op}/{ot}")


if __name__ == "__main__":
    main()

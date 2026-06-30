#!/usr/bin/env python
"""Summarise Phase 1 UrbanTrip failures without modifying official eval code."""

import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chinatravel.data.load_datasets import load_query
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.utils import load_json_file
from eval_tpc import cal_default_pr_score, load_result


DEFAULT_METHOD = "UrbanTripOptimizedV5_TPCLLM_en_oracletranslation"


def _parse_method_from_log(log_path):
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.search(r"^Method:\s*(\S+)", line.strip())
            if match:
                return match.group(1)
    return None


def _ids(values):
    return [str(v) for v in values]


def _sample(values, limit):
    return _ids(sorted(values)[:limit])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", "--splits", dest="splits", required=True)
    parser.add_argument("--lang", "--locale", choices=["zh", "en"], default="en")
    parser.add_argument("--method", default=None)
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing per-query JSON result files. Defaults to results/<method>.",
    )
    parser.add_argument("--log", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-limit", type=int, default=50)
    args = parser.parse_args()

    method = args.method or _parse_method_from_log(args.log) or DEFAULT_METHOD
    eval_args = SimpleNamespace(splits=args.splits, method=method, lang=args.lang)

    query_index, query_data = load_query(eval_args)
    results_dir = args.results_dir or os.path.join(PROJECT_ROOT, "results", method)
    _, result_data = load_result(eval_args, query_index, results_dir)
    plans = result_data["default"]

    schema = load_json_file(os.path.join(PROJECT_ROOT, "chinatravel/evaluation/output_schema.json"))
    schema_rate, schema_result_agg, schema_pass_id = evaluate_schema_constraints(
        query_index, plans, schema=schema
    )
    macro_comm, micro_comm, common_result_agg, commonsense_pass_id = evaluate_commonsense_constraints(
        query_index, query_data, plans, verbose=False, lang=args.lang
    )
    (
        macro_logi,
        micro_logi,
        conditional_macro_logi,
        conditional_micro_logi,
        logi_result_agg,
        logi_pass_id,
    ) = evaluate_hard_constraints_v2(
        query_index,
        query_data,
        plans,
        env_pass_id=commonsense_pass_id,
        verbose=False,
        lang=args.lang,
    )

    query_set = set(query_index)
    schema_pass = set(schema_pass_id)
    common_pass = set(commonsense_pass_id)
    logic_pass = set(logi_pass_id)
    all_pass = schema_pass & common_pass & logic_pass
    preference = cal_default_pr_score(query_index, query_data, plans, all_pass)

    schema_fail = query_set - schema_pass
    commonsense_fail = query_set - common_pass
    logic_fail = common_pass - logic_pass
    final_fail = query_set - all_pass

    summary = {
        "split": args.splits,
        "method": method,
        "results_dir": os.path.abspath(results_dir),
        "total": len(query_index),
        "counts": {
            "schema_pass": len(schema_pass),
            "commonsense_pass": len(common_pass),
            "logic_pass": len(logic_pass),
            "final_pass": len(all_pass),
            "schema_fail": len(schema_fail),
            "commonsense_fail": len(commonsense_fail),
            "logic_fail_after_commonsense": len(logic_fail),
            "final_fail": len(final_fail),
        },
        "metrics": {
            "schema_rate": schema_rate,
            "MicEPR": micro_comm,
            "MacEPR": macro_comm,
            "C-LPR": conditional_micro_logi,
            "FPR": len(all_pass) / len(query_index) * 100 if query_index else 0.0,
            "DAV": float(preference[0]) * 100,
            "ATT": float(preference[1]) * 100,
            "DDR": float(preference[2]) * 100,
        },
        "samples": {
            "schema_fail": _sample(schema_fail, args.sample_limit),
            "commonsense_fail": _sample(commonsense_fail, args.sample_limit),
            "logic_fail_after_commonsense": _sample(logic_fail, args.sample_limit),
            "final_fail": _sample(final_fail, args.sample_limit),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

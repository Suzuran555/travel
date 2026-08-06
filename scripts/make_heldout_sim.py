#!/usr/bin/env python3
"""Build an oracle-stripped copy of the familiarization split to simulate the
FORMAL Phase-2 held-out conditions locally.

Why: the official familiarization data (TPC_IJCAI_2026_phase2_familiar_100_data)
still contains ``hard_logic_py``, so a local run on the familiar split never hits
the held-out failure mode. The FORMAL held-out data has NO oracle fields, which
makes the harness-internal ``evaluate_one -> evaluate_hard_constraints_v2`` raise
``KeyError('hard_logic_py')``. Running the harness against THIS stripped split
reproduces that condition so you can confirm the run survives (guarded
evaluate_one) and every query still writes ``results/<method>/<uid>.json``.

Usage (from the repo root):
    python scripts/make_heldout_sim.py \
        --src chinatravel/data/en/phase2_familiar_EN \
        --split-src chinatravel/evaluation/default_splits/phase2_familiar.txt \
        --name phase2_heldout_sim

Then run the harness under held-out conditions:
    python agent_env/scripts/solve_script_with_harness.py \
        --method TPCAgent_heldoutsim --split phase2_heldout_sim
    # expect: NO crash, results/TPCAgent_heldoutsim/<uid>.json for every query,
    #         evaluate_one logs "[eval] internal evaluate_one skipped (KeyError ...)".
"""
import argparse
import json
import os

ORACLE_FIELDS = (
    "hard_logic",
    "hard_logic.py",
    "hard_logic_py",
    "hard_logic_nl",
    "soft_logic",
    "soft_logic_py",
    "soft_logic_nl",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of familiar {uid}.json query files")
    ap.add_argument("--split-src", required=True, help="familiar split .txt (uid per line)")
    ap.add_argument("--name", default="phase2_heldout_sim", help="new split name")
    ap.add_argument("--data-root", default="chinatravel/data/en", help="where to write {name}_EN/")
    ap.add_argument("--splits-root", default="chinatravel/evaluation/default_splits")
    args = ap.parse_args()

    out_dir = os.path.join(args.data_root, f"{args.name}_EN")
    os.makedirs(out_dir, exist_ok=True)
    uids = [l.strip() for l in open(args.split_src) if l.strip()]

    stripped_fields = set()
    for uid in uids:
        src = os.path.join(args.src, f"{uid}.json")
        with open(src, encoding="utf-8") as f:
            rec = json.load(f)
        for k in ORACLE_FIELDS:
            if k in rec:
                stripped_fields.add(k)
                rec.pop(k, None)
        with open(os.path.join(out_dir, f"{uid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)

    split_out = os.path.join(args.splits_root, f"{args.name}.txt")
    with open(split_out, "w") as f:
        f.write("\n".join(uids) + "\n")

    print(f"wrote {len(uids)} oracle-stripped queries -> {out_dir}")
    print(f"wrote split -> {split_out}")
    print(f"stripped fields: {sorted(stripped_fields) or 'none (already oracle-free)'}")
    # sanity: confirm the first record is oracle-free
    sample = json.load(open(os.path.join(out_dir, f"{uids[0]}.json")))
    leftover = [k for k in sample if any(o in k for o in ("hard_logic", "soft_logic"))]
    print(f"sample fields: {list(sample.keys())}  | oracle leftover: {leftover or 'NONE ✓'}")


if __name__ == "__main__":
    main()

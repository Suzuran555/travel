"""Parallel NL->DSL pretranslation via Qwen3.6-27B on local Ollama.

Writes per-uid results into cache/translation_Qwen3.6-27B_reflect/<uid>.json —
the exact cache translate_nl2sl(load_cache=True) reads, so subsequent
non-oracle planner runs (--llm ollama-qwen3.6-27b) skip translation entirely.

Shard across workers:  for i in 0..N-1: python pretranslate.py --shard i/N &
(Ollama server must allow parallel decode: OLLAMA_NUM_PARALLEL >= N.)
Already-cached uids are skipped, so the loop is resumable after any crash.
"""
import argparse
import os
import sys
import time
import json
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chinatravel.data.load_datasets import load_query, save_json_file
from chinatravel.agent.llms import OllamaChat
from chinatravel.agent.nesy_agent.nl2sl_hybrid_en import (
    nl2sl_reflect as nl2sl_reflect_en,
)

SPLIT = "TPC_IJCAI_2026_phase1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--uids", type=str, default="")
    ap.add_argument("--tag", type=str, default="qwen3.6:27b")
    args = ap.parse_args()
    shard_i, shard_n = [int(x) for x in args.shard.split("/")]

    name_map = {"qwen3.6:27b": "Qwen3.6-27B", "qwen3:8b": "Qwen3-8B"}
    llm = OllamaChat(
        model_tag=args.tag,
        display_name=name_map.get(args.tag, args.tag.replace(":", "-")),
    )
    # Optional suffix (e.g. "_zhinstr") to write into a separate cache dir
    # without touching the live translation cache. Default: unset = no change.
    suffix = os.environ.get("PENGUINS_CACHE_SUFFIX", "")
    cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cache",
        f"translation_{llm.name}_reflect{suffix}",
    )
    os.makedirs(cache_dir, exist_ok=True)

    ns = argparse.Namespace(splits=SPLIT, lang="en")
    query_index, query_data = load_query(ns)
    only = set(u for u in args.uids.split(",") if u)

    todo, done, failed = [], 0, 0
    for n, uid in enumerate(query_index):
        if only and uid not in only:
            continue
        if not only and n % shard_n != shard_i:
            continue
        if os.path.exists(os.path.join(cache_dir, f"{uid}.json")):
            done += 1
            continue
        todo.append(uid)
    if args.limit:
        todo = todo[: args.limit]
    print(f"shard {args.shard}: {len(todo)} to translate ({done} already cached)")

    for k, uid in enumerate(todo):
        t0 = time.time()
        query = deepcopy(query_data[uid])
        # keep the oracle DSL out of the LLM's view but preserved for comparison
        oracle_logic = query.pop("hard_logic_py", None)
        oracle_hl = query.pop("hard_logic", None)
        try:
            out = nl2sl_reflect_en(query, llm)
        except Exception as exc:  # noqa: BLE001 - log and continue the batch
            print(f"[{uid}] TRANSLATE ERROR: {exc}")
            failed += 1
            continue
        if "error" in out:
            out["hard_logic_py"] = {}
        out["_oracle_hard_logic_py"] = oracle_logic
        if oracle_hl is not None:
            out["_oracle_hard_logic"] = oracle_hl
        save_json_file(out, os.path.join(cache_dir, f"{uid}.json"))
        print(
            f"[{uid}] ok in {time.time()-t0:.0f}s "
            f"({k+1}/{len(todo)}, {failed} failed)"
        )
    print(f"shard {args.shard} DONE: {len(todo)-failed} translated, {failed} failed")


if __name__ == "__main__":
    main()

# Team "Antarctic penguins" — TPC 2026 Phase-2 agent

Self-contained submission package for the ChinaTravel benchmark.

## Install

Copy this directory into a stock ChinaTravel checkout as
`chinatravel/agent/tpc_agent/` (replacing the stub directory, exactly as in
the 2025 TPC format). Nothing else in the repository needs to change: the
package uses only package-relative imports plus the stock
`chinatravel.*` modules and databases that ship with the benchmark.

```
rm -rf chinatravel/agent/tpc_agent
cp -r <this dir> chinatravel/agent/tpc_agent
```

## Run

```
python run_tpc.py --splits <split> [--index <uid>] \
    --agent TPCAgent --llm TPCLLM --lang en --timeout 300
```

## Backbone LLM configuration (environment variables)

The only LLM stage is NL→DSL translation; planning and enrichment are fully
symbolic/deterministic. `TPCLLM` talks to any OpenAI-compatible server:

| Variable | Meaning |
|---|---|
| `CHINATRAVEL_OPENAI_BASE_URL` | e.g. `http://localhost:30000/v1` (SGLang / vLLM serving Qwen3.6-27B) |
| `CHINATRAVEL_OPENAI_MODEL`    | served model name |
| `CHINATRAVEL_OPENAI_API_KEY`  | default `EMPTY` (accepted by SGLang) |
| `CHINATRAVEL_LLM_NAME`        | display name; keys the translation cache dir (default `Qwen3.6-27B`) |
| `CHINATRAVEL_LLM_THINK`       | `1` to enable thinking mode (default off) |
| `OLLAMA_TAG` / `OLLAMA_HOST_URL` | fallback local Ollama backend for testing (e.g. `qwen3.6:27b`) |

Timing knobs (defaults fit a 300 s per-query harness timeout):

| Variable | Default | Meaning |
|---|---|---|
| `TPC_TIME_BUDGET` | 290 | total seconds per query the agent aims for |
| `TPC_ENRICH_RESERVE` | 45 | seconds reserved for post-planning enrichment |
| `TPC_EMIT_MARGIN` | 15 | seconds before budget end by which SOME schema-valid plan is always returned; on planner overrun a deterministic env-DB fallback plan (intercity legs + hotel + hotel breakfasts) is emitted |

The whole `run()` is wall-clocked from entry: live NL→DSL translation time is
deducted from the search budget (`_urbantrip_search_start`), every backbone
request is capped to the emission deadline, and the planner runs in a worker
thread that is abandoned in favor of the prebuilt fallback plan if it misses
the emission deadline.

## What run(query) does

1. Strips the oracle constraint annotations (`hard_logic*`) from the query —
   the agent acts only on constraints it generates itself.
2. NL→DSL translation with the backbone LLM (hardened prompts, mechanical
   disjunction/count verifiers, DSL canonicalization), cached under
   `cache/translation_<name>_reflect/`.
3. `UrbanTripOptimizedV6` symbolic search (vendored planner + its search
   stack; the intracity/intercity segment ranking tables ship inside
   `data/segments/en/`).
4. An in-run enrichment battery (12 regression-safe passes) that only keeps
   an edit if the plan still passes schema + commonsense + the *generated*
   hard logic via the stock evaluation modules, and a soft metric improved.

## Dependencies

Only the stock `requirements.txt` (pandas, numpy, geopy, scikit-learn,
jsonschema, json_repair, func_timeout, requests, ...). No GPU inference is
performed in-process; all model calls go over HTTP to the configured server.

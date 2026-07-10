# TPC IJCAI 2026 — Phase 2 Readiness Dossier

Date: 2026-07-08
Scope: Track 2 (agent harness). Synthesized from four recon streams (competition website crawl, upstream branch analysis of LAMDA-NeSy/ChinaTravel, agent_env harness inspection, TPC@AIC2025 precedent) plus a local compliance test of our current pipeline (UrbanTripOptimizedV6, leaderboard 97.64).

Verdict: **CONDITIONALLY READY — pipeline architecture is compatible, but 4 blocking work items must land before Aug 1.** Our plans pass the current harness end-to-end (verified), but the enrichment layer violates the hidden-oracle contract, we cannot yet talk to the organizers' SGLang endpoint, code is not packaged per the expected submission layout, and the live NL→DSL path is untested under Qwen3.6-27B within the time budget.

---

## 1. Requirements as known (with evidence)

### 1.1 Hard requirements (explicit, published)

| # | Requirement | Source |
|---|-------------|--------|
| R1 | Submit a **self-contained harness**: "agent code, prompts, scaffolding, and the files needed to run it." | TPC site index (Submission section, exact wording) |
| R2 | Organizers run every harness on **Qwen3.6-27B via SGLang on 2× A800 80G (no NVLink)** against a **held-out test set**; Phase 2 results **directly determine 1st/2nd/3rd**. | TPC site index; evaluation.html |
| R3 | Timeline: Stage 1 deadline **Aug 1 2026**; Stage 2 evaluation **Aug 1–7**; technical report **Aug 7**; final results **Aug 10**. Daily leaderboard cutoff 20:00 Beijing time. | TPC site index, Dates section |
| R4 | Eligibility: top 5 Phase 1 teams + ties at 5th place, "subject to eligibility verification and compliance"; ≥1 team member registered for IJCAI-ECAI 2026. | TPC site FAQ |
| R5 | Scoring formula (same as Phase 1): 10% EPR-micro + 10% EPR-macro + 25% C-LPR + 40% FPR + 5% DAV + 5% ATT + 5% DDR. | index + evaluation.html |
| R6 | Announced Phase 2 scaffold: **agent_env** wrapper (adapter.py, cli.py, mcp_stdio.py, http_server.py, scripts/solve_script_with_harness.py, config.toml.example, SKILL.md) with opencode/codex runners; output = JSON matching `chinatravel/evaluation/output_schema.json`, usually inside `<output>...</output>`, saved at `results/<method>/<uid>.json`. | agent-environment.html; ChinaTravel repo agent_env/ |
| R7 | Development against ChinaTravel dataset (HF LAMDA-NeSy/ChinaTravel) permitted. | TPC site Rules, Track 2 |
| R8 | Code of conduct: no cheating/attacks on evaluation/sharing answers; "Reproducibility is expected for top-ranked teams." | TPC site FAQ / CoC |

### 1.2 Evaluator requirements (from upstream/main, HEAD d4f90b1, Jul 7 2026)

| # | Requirement | Source |
|---|-------------|--------|
| E1 | **Is_activity_grounded** (new commonsense check, PR #25, commit 56a27b1): intercity transport (airplane/train) allowed ONLY as first activity of day 1 or last activity of last day; must have start/end + matching FlightID/TrainID; must NOT have `position`. POI activities restricted to types {attraction, breakfast, lunch, dinner, accommodation}, non-empty `position`, NO start/end/FlightID/TrainID. Unknown activity types fail. | chinatravel/symbol_verification/commonsense_constraint.py @ upstream/main d4f90b1 |
| E2 | Tightened `output_schema.json` mirroring E1 (mutually exclusive position vs start/end/ID fields by activity type). | chinatravel/evaluation/output_schema.json @ 56a27b1 |
| E3 | Emit canonical POI name **"Sola Bistro"**, not "Bistro Sola" — the accepting alias is explicitly "temporary" (commit 79ad36f "Add temporary POI alias normalization", chinatravel/symbol_verification/concept_func.py:51). | upstream/main; commit 79ad36f |
| E4 | `activity_position()` returns '' for airplane/train activities — hard-logic constraints no longer see transport rows. | commit 56a27b1 |
| E5 | Dataset loader hard-strips oracle fields (hard_logic, hard_logic_py, hard_logic_nl) unless `--oracle_translation`; hard-fails on string-encoded hard_logic_py. | chinatravel/data/load_datasets.py @ upstream/feature/openai-runtime-refactor (aaafe7f) |

### 1.3 Runtime/interface requirements (inferred from upstream/feature/openai-runtime-refactor, commit aaafe7f — high confidence this is the Phase 2 runtime)

| # | Requirement | Source |
|---|-------------|--------|
| I1 | LLM access ONLY via `OpenAICompatibleLLM` (chinatravel/agent/llms.py): model from `--llm` / `CHINATRAVEL_OPENAI_MODEL` / `OPENAI_MODEL`; endpoint from `OPENAI_BASE_URL` / `CHINATRAVEL_OPENAI_BASE_URL`; key from `OPENAI_API_KEY` (defaults to literal "EMPTY" when base_url set — the vLLM/SGLang local-server convention). No bundled weights; vllm/torch/transformers removed from requirements. | branch feature/openai-runtime-refactor, aaafe7f |
| I2 | Chat Completions wire API by default (`CHINATRAVEL_OPENAI_WIRE_API=chat`); `max_tokens` limit arg unless `CHINATRAVEL_OPENAI_TOKEN_LIMIT_ARG` overrides; openai>=1.66.0 client. | aaafe7f, llms.py |
| I3 | Package-relative imports (`chinatravel.agent.*`), no sys.path hacks; agents constructed via `load_model.AGENT_BUILDERS['TPCAgent']` with kwargs {method, env, backbone_llm, cache_dir, log_dir, debug, refine_steps, lang}. TPCLLM class deleted upstream. | aaafe7f, load_model.py |
| I4 | Expected re-test commands: `python run_exp.py --splits tpc_phase_2_online_test --agent TPCAgent --llm <model>`; `python eval_tpc.py --splits tpc_phase_2_online_test --method TPCAgent_<normalized-llm>`. **Never** `--oracle_translation` at inference. Results dir = `TPCAgent_{normalize_run_name(model)}`. | aaafe7f, run_tpc.py/TPC@AIC2025 readme update |
| I5 | Sampling defaults NOT forced anymore (old Qwen class forced temperature=0/top_p=0.001; OpenAICompatibleLLM sends none) — must explicitly pin temperature/top_p for determinism. | aaafe7f, llms.py |
| I6 | WorldEnv command strings AST-whitelisted to a single tool-function call — no compound expressions via env('...'). | aaafe7f, world_env.py |

### 1.4 Precedent-based requirements (TPC@AIC2025 readme — NOT confirmed for 2026, but the only packaging spec that exists)

| # | Requirement | Source |
|---|-------------|--------|
| P1 | All code inside a single `chinatravel/agent/tpc_agent/` folder; everything outside reset to stock. Submission `<team-id>_code.zip` (top-level tpc_agent/ + contact.txt), unzipped ≤ 40 GB; also submit results zip. | TPC@AIC2025/readme.md |
| P2 | ≤ **5 minutes per query** (func_timeout in run_tpc.py; timeout = failed query); **5 repeated evaluations averaged**; irreproducibility ⇒ disqualification. | TPC@AIC2025/readme.md (2026 site does NOT confirm run count) |
| P3 | Fully offline evaluation machine: 14-core Xeon 6348, 100 GB RAM, A800-80GB, 50 GB SSD, CUDA 12.4; no external APIs. (2026 supersedes the local-weights rule: model is now organizer-served.) | TPC@AIC2025/readme.md; superseded in part by aaafe7f readme edit "以当期赛事通知为准" |
| P4 | Code freezes at semifinal deadline; same zip re-scored on private final dataset (70%) + technical report/defense (30%). | TPC@AIC2025/readme.md |

### 1.5 Harness I/O contract (verified locally)

- Per-uid JSON itinerary conforming to output_schema.json: top-level people_number, start_city, target_city, itinerary; activities need type/start_time/end_time/price/cost/transports; tickets+TrainID/FlightID for intercity; position+tickets for attractions; position+room_type+rooms for accommodation. (agent_env/SKILL.md lines 76–97.)
- Harness accepts plans via `--plan-file` OR opencode/codex runners emitting `<output>`-tagged JSON; provenance (agentic loop vs symbolic planner) is **not checked**. Only oracle DSL fields are hidden from the prompt. Our NL→DSL + symbolic-planner pipeline is fully compatible.
- **Verified end-to-end**: one frozen V6 plan run through `solve_script_with_harness.py --plan-file` → exit 0, schema/commonsense/logical/all_pass ALL true (6/6 constraints), prompt contained 0 oracle-field occurrences. Sandbox: `/private/tmp/claude-501/-Users-zhanggangyi-Desktop-TPC2026/ccadc8f1-a9ae-4981-8039-90c33e359ae2/scratchpad/phase2test/`.
- All 1000/1000 current plans in `/Users/zhanggangyi/Desktop/TPC2026/travel/results/UrbanTripOptimizedV6_v13100_en/` pass the FORK's output_schema.json (must be re-validated against upstream's tightened schema).

---

## 2. Gap list (by severity)

### CRITICAL
1. **Enrichers gate on ORACLE constraints.** Every root-level `enrich_*.py` builds a Namespace without `oracle_translation=False`, so `load_query` keeps oracle hard_logic/hard_logic_py/hard_logic_nl, and `passes()` + direct `query['hard_logic_py']` parsing (enrich_meal.py:16/21-27, enrich_attr.py:19, enrich_att.py:63, enrich_attdilute.py:73-83, enrich_endattr.py:48, enrich_fillerswap.py:75-87, enrich_gapattr.py:97-113, enrich_seqswap.py:39/93/117, enrich_route.py:25, enrich_rebook.py:40, enrich_lateattr.py:45, enrich_breakfast.py:27, enrich_soft.py:15, enrich_day.py:17, enrich_travelday_meals.py:122) consume it. This violates the Phase-2 hidden-oracle contract (E5) and, under 2025-precedent rules, is disqualifying. Fix: gate all enrichment on the GENERATED DSL that V6 already caches (`cache/translation_<llm.name>_reflect/<uid>.json`, produced at tpc_agent_optimized_v6.py:505-523).

### HIGH
2. **Enrichment is an hours-long offline batch outside the agent.** Phase-2 precedent (P1/P2) requires all code in tpc_agent/ with ≤5 min/query. Enrichment passes must be folded into `TPCAgent.run()` per-query (V6 already in-runs two: `_postprocess_insert_missing_meals`, `_postprocess_insert_attractions_for_dav` at tpc_agent_optimized_v6.py:549-550) or dropped.
3. **No OpenAI-compatible LLM adapter in the fork.** Local llms.py has only hardcoded external APIs + in-process models; nothing can hit the organizers' SGLang endpoint (I1). Must merge upstream aaafe7f or vendor an equivalent `OpenAICompatibleLLM` and wire `init_llm`.
4. **Live NL→DSL path untested under Qwen3.6-27B.** Translation cache is useless on held-out uids; `nl2sl_reflect` (chinatravel/agent/nesy_agent/nl2sl_hybrid_en.py:488) becomes the primary path but has no latency/robustness data under a 27B model within 5 min/query, and sampling isn't pinned for reproducibility (I5, P2, R8).

### MEDIUM
5. **Packaging.** Planner lives in `chinatravel/agent/UrbanTrip/` and registers by editing stock load_model.py:104 (reset to stock per precedent). Need a TPCAgent wrapper vendoring V6 + all UrbanTrip deps inside `chinatravel/agent/tpc_agent/`, package-relative imports, no sys.path hacks (tpc_llm.py currently sys.path.appends) (P1, I3).
6. **Fork not re-synced with upstream/main d4f90b1.** Local evaluator lacks Is_activity_grounded, tightened schema, AST-whitelisted WorldEnv, OpenAI runtime refactor. The 1000/1000 schema pass was against the fork's schema. Re-sync (preserving the local `normalize_hard_logic_constraint` shim in chinatravel/symbol_verification/hard_constraint.py) and re-run full eval (E1/E2, gap re-validation).
7. **"Bistro Sola" emission.** results/UrbanTripOptimizedV6_v13100_en/20250323010327713880.json passes only via the temporary alias (E3). Fix planner POI naming to emit "Sola Bistro".

### LOW
8. **solve_script_with_harness.py cannot load this fork's EN-only data** — `load_queries()` omits `lang`, scans zh layout, KeyErrors on first uid. One-line lang plumbing fix or zh-layout placement; likely irrelevant if organizers run their own repo.
9. **opencode/codex runner path untested** (binaries absent locally). If organizers mandate the runner path: need the CLI installed, config.toml → SGLang endpoint, output in `<output>` tags within 900 s default timeout.

---

## 3. Action plan (ordered, with effort estimates)

| # | Action | Effort | Deadline target |
|---|--------|--------|-----------------|
| A1 | Re-sync fork with upstream/main d4f90b1 + merge feature/openai-runtime-refactor (aaafe7f); preserve local normalize_hard_logic_constraint shim; fix any breakage from package-relative imports and TPCLLM deletion. | 1–2 days | Jul 12 |
| A2 | Re-validate all 1000 frozen plans against upstream's tightened output_schema.json + Is_activity_grounded; fix planner emissions that fail (incl. "Sola Bistro" canonical name). | 0.5–1 day (after A1) | Jul 13 |
| A3 | Rebuild enrichment gating on GENERATED DSL: replace every oracle-qd `passes()` and `query['hard_logic_py']` read with the V6 translation cache / live translation output. Regression-test that leaderboard-frozen scores are unchanged on Phase-1 split when generated DSL == oracle DSL. | 2–3 days | Jul 16 |
| A4 | Fold enrichment passes into TPCAgent.run() per-query (following the pattern of the two already-in-run passes); profile per-query wall clock; drop/simplify passes that can't fit a 5-min budget with margin. | 3–4 days | Jul 20 |
| A5 | Stand up a local SGLang (or vLLM) server with Qwen3.6-27B (or the closest available Qwen3.6 checkpoint / dashscope qwen3.6-27b API as stand-in); wire OpenAICompatibleLLM env vars; run the LIVE nl2sl_reflect path over the full Phase-1 EN split; measure latency, DSL accuracy vs cached translations, and score delta. Pin temperature=0/top_p via default_request_args; set CHINATRAVEL_OPENAI_RAISE_ERRORS=1 in dev. | 3–4 days (parallel with A3/A4) | Jul 18 |
| A6 | Package: create `chinatravel/agent/tpc_agent/` vendoring V6 + UrbanTrip deps, TPCAgent class matching AGENT_BUILDERS kwargs, package-relative imports, no sys.path hacks; verify `run_exp.py --agent TPCAgent --llm <served-model>` works on a STOCK upstream checkout with only tpc_agent/ dropped in. | 2–3 days | Jul 23 |
| A7 | Add score insurance: schema-validating repair/fallback layer around plan emission (parse failures = failed evals); internal per-query time-budget mechanism mirroring func_timeout with a safe fallback plan. | 1–2 days | Jul 25 |
| A8 | Determinism audit: 5 repeated full runs on Phase-1 split with pinned sampling; diff plans byte-for-byte or score-identically; eliminate nondeterminism (dict ordering, retries, timeouts). | 1–2 days | Jul 27 |
| A9 | Email chinatravel454@gmail.com with the open-unknown list (Section 4) NOW — answers may reshape A4/A6. | 0.5 day | Jul 9 |
| A10 | Dress rehearsal: full end-to-end run on stock upstream + tpc_agent/ + SGLang stand-in under 5-min timeouts; also test the `--plan-file` and (if CLI obtainable) opencode runner paths of solve_script_with_harness.py. Draft technical report skeleton. | 2 days | Jul 29–31 |

Total: ~16–22 working days of effort against ~17 working days to Aug 1 — feasible only with A3–A5 parallelized; start immediately and keep Phase-1 leaderboard work frozen unless top-5 is threatened (ties at 5th all advance; position within top-5 has zero carryover, per site FAQ).

---

## 4. Open unknowns needing organizer clarification (email chinatravel454@gmail.com)

1. Phase 2 submission portal, packaging format, directory structure, entry point, and size limit ("The Round 2 submission portal and format will be announced later"; evaluation.html Phase 2 card = "Coming Soon"). Does the 2025 `tpc_agent/` zip convention still apply?
2. Per-query and/or total wall-clock time limits; token/context limits per query. (2025 precedent: 5 min/query — unconfirmed for 2026.)
3. Number of evaluation runs and whether scores are averaged (2025: 5 runs averaged; 2026 site silent).
4. SGLang serving configuration: sampling defaults, whether harnesses may set temperature/top_p, max_tokens ceiling, concurrency.
5. Will team code be invoked via the agent_env opencode/codex runner path, via run_exp.py/AGENT_BUILDERS, or will plan directories be accepted? (agent_env supports both; the rules must decide.)
6. Internet access policy during organizer runs; whether bundled caches/precomputed derived-data files (derivable from the official database) are permitted inside the "self-contained harness".
7. Held-out test set: size, language (EN/zh), split name.
8. Technical report format/length/template; system-description-paper requirements.
9. Timing of the "Bistro Sola" temporary alias removal / underlying data-name fix.

---

## 5. Evidence index

- Website: TPC IJCAI 2026 site — index SPA (app.js), agent-environment.html, evaluation.html, leaderboard.html (all other pages 404).
- Upstream repo: LAMDA-NeSy/ChinaTravel — main @ d4f90b1 (Jul 7 2026, merge PR #25); commits 56a27b1 (entity grounding + schema), 79ad36f (temporary alias), 70feb9d (quote normalization); branch feature/openai-runtime-refactor @ aaafe7f.
- Precedent: /Users/zhanggangyi/Desktop/TPC2026/travel/TPC@AIC2025/readme.md.
- Local compliance test: /private/tmp/claude-501/-Users-zhanggangyi-Desktop-TPC2026/ccadc8f1-a9ae-4981-8039-90c33e359ae2/scratchpad/phase2test/ (validate_all.py, dryrun.log, planfile_run.log, fakeroot/).
- Our pipeline: /Users/zhanggangyi/Desktop/TPC2026/travel/ — tpc_agent_optimized_v6.py, enrich_*.py, chinatravel/agent/UrbanTrip/, chinatravel/agent/nesy_agent/nl2sl_hybrid_en.py, results/UrbanTripOptimizedV6_v13100_en/.

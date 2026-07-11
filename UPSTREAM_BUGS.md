# Upstream Bug Report — ChinaTravel / TPC IJCAI 2026

**From:** Team Antarctic penguins (Phase-1: 100.00, v13) · **Date:** 2026-07-11
**Repo referenced:** LAMDA-NeSy/ChinaTravel @ `d4f90b1` (main) and `feature/openai-runtime-refactor` @ `aaafe7f`
All issues verified against the official phase-1 data and evaluators; reproduction details below.

---

## BUG 1 — Evaluator: contradictory POI alias makes one query locally unsatisfiable
**Severity: high (scoring correctness) · Component: `chinatravel/symbol_verification/concept_func.py` + EN restaurant DB**

`_POI_NAME_ALIASES = {"Bistro Sola": "Sola Bistro"}` ("temporary POI alias normalization",
commit `79ad36f`) maps the name **away** from the EN database row, which is spelled
`Bistro Sola` (Shanghai restaurants, row 301) after the EN name corrections. Result: for uid
`20250323010327713880` ("visit Bistro Sola between 17:00 and ..."), **both** spellings fail
commonsense grounding under this evaluator build — plan position `Bistro Sola` is normalized to
`Sola Bistro` (not in DB); plan position `Sola Bistro` has no DB row either. The public
leaderboard shows multiple teams at FPR 100.000, so the server evaluator evidently does not have
this alias+DB combination — i.e. the shipped open-source evaluator and the server evaluator
disagree on this uid. Fix: remove the alias, or rename the DB row to match it (either is
consistent; the current combination is contradictory).
*Repro:* evaluate any plan visiting `Bistro Sola` with the stock repo — commonsense error
`No information found given restaurant [Bistro Sola]` plus GoTo grounding errors on the
normalized endpoints.

## BUG 2 — DSL executor: AST whitelist rejects `break`/`continue` used by official constraints
**Severity: critical (crashes official scoring) · Component: `chinatravel/symbol_verification/dsl.py` (AST-whitelisted executor, PR #24/#25 line)**

The new `execute_dsl_code` whitelist omits `ast.Break` and `ast.Continue`. ~20 official phase-1
`hard_logic_py` constraint programs use `break` (we swept all 3,568 constraint programs). Under
stock upstream, those constraints raise `Unsupported DSL syntax: Break`, the affected plans'
logic results zero out, and FPR/C-LPR crater (we measured FPR 97.9 on a plan set that scores
99.9 with the fix). Fix: add `ast.Break, ast.Continue` to the whitelist.
*Repro:* run `eval_tpc.py` on any full phase-1 result set with stock `dsl.py`; grep stderr for
`Unsupported DSL syntax: Break`.

## BUG 3 — Environment: `next_page()` exhaustion sentinel crashes the official baseline agent
**Severity: medium (baseline agent crash) · Component: env pagination (`EnvOutput.next_page()`) + `chinatravel/agent/UrbanTrip/tpc_agent.py`**

On `feature/openai-runtime-refactor`, `next_page()` returns the **string** `"No more data."`
when results are exhausted (the old env returned an empty DataFrame). The repo's own
`UrbanTrip/tpc_agent.py` (and any agent copying its pattern) feeds that into `pd.concat`,
raising `TypeError: cannot concatenate object of class <class 'str'>`. We hit this in our V1–V6
agents (fixed with isinstance guards); upstream's own agent still has the latent crash.
Fix: return an empty DataFrame, or document the sentinel and guard in the baseline agent.

## BUG 4 — Data: truncated `nature_language` in official phase-1 queries
**Severity: medium (unsatisfiable-from-NL queries) · Component: phase-1 EN query data**

At least two queries end mid-sentence, so their constraints are unrecoverable from natural
language alone (any honest NL→DSL pipeline — including the phase-2 harness setting — cannot
solve them; only the oracle `hard_logic_py` reveals the real requirements):
- `20250322123458076116` — text ends `"2. Total budget for"` (no amount, disjunction truncated)
- `20250323003100968460` — text ends `"satisfy at least one of the following: 1. Do"`
Suspected same class: `20250322001041444207` (constraint content not derivable from its NL).
If the held-out phase-2 set has the same defect, harness scores will be depressed for all teams
equally — worth a data pass before the final evaluation.

## BUG 5 — Data: doubled-quote typo in an official constraint literal
**Severity: low (one query) · Component: phase-1 query `hard_logic_py`**

One official constraint contains `'Rui''en Town'` (doubled quote inside a single-quoted Python
string). Python evaluates this as string concatenation → `'Ruien Town'`, which matches no POI.
Any plan actually visiting the intended POI cannot satisfy the constraint as written; the
constraint is satisfiable only vacuously. Fix: escape properly (`'Rui\'en Town'`) or use double
quotes.

## BUG 6 — Baseline NL→DSL prompt: few-shot example induces invented room constraints
**Severity: medium (baseline translation quality) · Component: `chinatravel/agent/nesy_agent/nl2sl_hybrid_en.py` (step-2 few-shot)**

The English step-2 few-shot example includes `rooms==2, room_type==2` in its input/answer.
Smaller models (measured on Qwen3.6-27B, the phase-2 model) copy the pattern and **invent room
constraints for queries that never mention rooms**: 49 of 61 translation failures in our 113-uid
sample were caused solely by this (fidelity 46% → ~89% once removed). Since phase 2 runs on
Qwen3.6-27B, every team building on the stock prompt inherits this. Fix: use an example whose
NL explicitly requests rooms, or drop the room fields from the example.

## BUG 7 — agent_env harness: cannot load EN data
**Severity: low (developer tooling) · Component: `agent_env/scripts/solve_script_with_harness.py`**

The harness's query loading omits `lang="en"` (`load_queries` defaults to zh layout), so on an
EN-data checkout it crashes with `KeyError` on the first uid. One-line fix: plumb a `--lang`
flag through to the loader.

---

### Summary table

| # | Component | Severity | One-line fix |
|---|---|---|---|
| 1 | evaluator alias vs EN DB | high | remove alias or rename DB row |
| 2 | DSL AST whitelist | critical | add `ast.Break`, `ast.Continue` |
| 3 | `next_page()` sentinel | medium | return empty DataFrame |
| 4 | truncated NL queries | medium | re-export the affected queries |
| 5 | `'Rui''en Town'` literal | low | fix the quote escaping |
| 6 | few-shot room example | medium | remove unlicensed room fields |
| 7 | agent_env EN loading | low | add `--lang` plumbing |

Patches for 2, 3, 6, 7 exist on our fork (`Suzuran555/travel`, branches
`improve/top5-soft-max` / `phase2-sync`) and we are happy to upstream them as PRs.
Contact: Team Antarctic penguins.

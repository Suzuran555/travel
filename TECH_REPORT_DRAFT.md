# TPC @ IJCAI-ECAI 2026 — Technical Report (DRAFT)

**Team:** Antarctic penguins
**Track:** ChinaTravel Travel Planning Challenge, Track 2 (agent harness)
**Contact:** [TODO: contact email / member list]
**Date:** [TODO: finalize before Aug 7, 2026]

> DRAFT STATUS: Phase-1 numbers are final. Phase-2 numbers marked **TODO** await the
> organizer-run evaluation (Aug 1–7) on the held-out test set. All local Phase-2 figures below
> are from our own 113-uid simulation and are labeled as such.

---

## 1. Results Summary

| Setting | Overall | Notes |
|---|---|---|
| Phase 1 (official leaderboard) | **100.000** | Submission v13; all seven metrics at 100.0 |
| Phase 2, local honest simulation (113 uids, Qwen3.6-27B) | 99.00 → **99.9991** | End-to-end NL→plan, no oracle fields at inference; FPR improved from 111/113 to 113/113 |
| Phase 2, organizer-run (held-out set) | **TODO** | Final numbers from the Aug 1–7 evaluation |

Phase-1 trajectory: 93.78 (baseline) → 95.67 → 97.71 → 99.09 → 99.61 → 99.91 → 99.953 → 100.000.

Phase-2 local simulation detail (final checkpoint, `oracle_score.json`): FPR 100.0,
C-LPR 100.0, EPR-micro/macro 100.0, DAV 100.0, DDR 100.0, ATT 99.98 — Overall 99.9991 on the
113-uid split. An earlier honest end-to-end checkpoint on the same split scored 99.00 with
FPR 111/113; the gap was closed by translation-prompt hardening and generated-DSL-gated
enrichment (Section 3). Local runs use Qwen3.6-27B served via Ollama as a stand-in for the
organizers' SGLang endpoint; our LLM adapter is OpenAI-compatible and reads the standard
`CHINATRAVEL_OPENAI_*` environment variables.

Scoring formula (both phases):
`Overall = 0.10·EPRmic + 0.10·EPRmac + 0.25·C-LPR + 0.40·FPR + 0.05·DAV + 0.05·ATT + 0.05·DDR`.
FPR's 0.40 weight dominates; the soft metrics (DAV/ATT/DDR) are per-plan clamped to [0,1] and
averaged over passing plans only. This math dictated our strategy throughout: feasibility
first, and no soft-metric edit is ever allowed to risk a pass.

## 2. System Architecture (Phase 2)

The Phase-2 system is a neuro-symbolic pipeline: the LLM is used only where language
understanding is genuinely required (NL→DSL translation), and everything downstream is
deterministic symbolic search and mechanically verified post-processing.

```
natural-language query
  │
  ▼
(1) NL→DSL translation — Qwen3.6-27B, hardened prompts
      + mechanical verifiers / normalizers on the LLM output
  │
  ▼
(2) DSL canonicalizer — normalizes constraint literals/aliases into a canonical form
  │
  ▼
(3) UrbanTripOptimizedV6 — symbolic planner
      phased bundle search, geo anchors, DFS memoization,
      budget-drop repair, travel-day timing bias
  │
  ▼
(4) Generated-DSL-gated enrichment — regression-gated soft-metric passes
  │
  ▼
schema-conformant JSON itinerary
```

**(1) Translation.** Translation fidelity is the entire Phase-2 game: in our Phase-1 AI lane,
a pure-AI source scored 86.53 solely because of translation-semantic errors (FPR 73.5), while
its passing plans scored ~100 on every soft metric. For Phase 2 we therefore hardened the
NL→DSL prompts against failure modes we measured on Qwen3.6-27B itself — most notably the
stock few-shot example that induces invented room constraints (Section 4, Finding 6; fixing it
alone moved measured fidelity from 46% to ~89% on our 113-uid sample) — and wrapped the model
output in mechanical verifiers and normalizers that reject or repair syntactically or
semantically malformed DSL without any oracle access.

**(2) Canonicalization.** A deterministic canonicalizer normalizes the generated DSL
(constraint-literal spelling, alias forms, quoting) so the symbolic layer sees a stable
constraint language regardless of LLM phrasing variance.

**(3) Symbolic planning.** UrbanTripOptimizedV6 performs a phased bundle search over intercity
transport, accommodation, attractions, and meals, with geographic anchoring, DFS memoization,
and budget-drop repair. A key structural component is the travel-day timing bias: the bundle
scorer rewards early arrival and late return departure, which keeps travel days rich in
activity time (in Phase 1 this single change lifted fresh plans by DAV +4.4 and DDR +9.6 —
gains post-processing could never reach).

**(4) Generated-gated enrichment.** Soft-metric enrichment passes (meal/attraction insertion,
transit re-moding, rebooking) run under one iron rule: an edit is kept only if the plan still
passes the full official evaluation and a soft metric strictly improves with none regressing.
Crucially, in Phase 2 the pass/fail gate is computed against the **generated** DSL produced in
step (1) — never against the hidden oracle constraints — so the system is honest end-to-end
under the hidden-oracle contract.

The submission is self-contained per the announced requirements ("agent code, prompts,
scaffolding, and the files needed to run it") and follows the TPC@AIC2025 packaging precedent
(all code inside `chinatravel/agent/tpc_agent/`, dropped into a stock organizer checkout,
≤5 minutes per query). [TODO: confirm final packaging once the 2026 portal/format is announced.]

## 3. Phase-1 Methodology

Phase 1 permitted oracle constraint programs at development time; our Phase-1 result therefore
combines the symbolic pipeline above with an offline optimization methodology. We describe it
fully here, including the AI-assisted components, for transparency.

**3.1 Correctness era (93.78 → 95.67).** Seven matching/heuristic bug fixes where satisfiable
constraints were silently failed (case-insensitive type matching, any-of vs all-of semantics,
alias canonicalization, people-aware budgets, arrive-time musts), plus the first
regression-safe meal enrichment using detour-minimizing recall
(`detour = d(A,X)+d(X,B)−d(A,B)` — insert POIs that are on the way).

**3.2 Planner era (95.67 → 99.09).** The V6 planner (Section 2) plus the travel-day timing
bias, and a donor-restore pass (`repair_fails.py`) that recovered failing plans from archived
snapshots (FPR 99.3 → 99.9).

**3.3 Regression-gated enrichment battery + per-uid best-merge (99.09 → 99.91).** Nine
post-processing stages, each obeying the iron rule above, plus `merge_soft.py`: per-uid
best-of-N across all result sets — every wave's output competes against every archive, and the
best passing version of each plan survives. Three planner "pilot" rounds re-planned only
residual uids with transit-aware flags: 99.756 → 99.828 → 99.910 → 99.917.

**3.4 Adversarial multi-agent verification (99.917 → 99.953).** A 29-agent workflow (7
auditors + 22 refuting verifiers) audited the residual gap. Every proposed "finding" had to
survive dedicated refuters before any code ran against the live plan set; two audit verdicts
were overturned this way, and both mattered. The audit produced five new enrichers exploiting
verified root causes (wrong-airport flight bookings, alphabetical filler-POI selection, joint
transit re-moding under cost caps, movable hotel check-in windows, and ATT leg-average
dilution), followed by seven per-uid combined-mechanism rebuilds that closed DAV/ATT/DDR/EPR to
100.0.

**3.5 AI-assisted per-instance refinement (disclosed).** The final step from 99.953 to 100.000
was an AI-workflow-crafted plan for a single uid our local evaluator scored as unsatisfiable
(the evaluator-divergence issue in Section 4, Finding 1). The plan was built by an AI workflow
and re-validated by an adversarial verifier under both plausible server-evaluator variants
(runtime-monkeypatched: alias removed / DB row renamed) before submission. We disclose plainly
that Phase-1 endgame work included AI-assisted per-instance plan construction and refinement;
all such plans pass the official evaluators on their own merits.

**3.6 Methodology principles.** (1) Regression-gating everywhere — every mutation re-runs the
full official eval, so scores only go up. (2) Per-uid best-merge — waves never overwrite each
other. (3) Adversarial verification before action. (4) Archive every state — full 1000-plan
snapshots made every experiment reversible, including recovery from a literal mid-pipeline
power outage. (5) Believe external evidence over your own proofs — the live leaderboard
refuted our local unsatisfiability "theorem," and the correct response was forensics, not
denial. (6) Fix the planner for structure; post-process for polish.

## 4. Key Findings for the Community

During Phase 1 and Phase-2 preparation we identified and verified several issues in the public
ChinaTravel repository and Phase-1 data (referenced against upstream `main` @ `d4f90b1` and
`feature/openai-runtime-refactor` @ `aaafe7f`). We report them factually; patches for findings
2, 3, 6, and 7 exist on our fork and have been offered upstream.

1. **Evaluator/data divergence on a POI alias (high).** The shipped evaluator's temporary
   alias `"Bistro Sola" → "Sola Bistro"` maps the name away from the EN database row (spelled
   `Bistro Sola`), making uid `20250323010327713880` locally unsatisfiable under the
   open-source evaluator: both spellings fail commonsense grounding. The public leaderboard
   showed multiple teams at FPR 100.000, so the server evaluator evidently does not have this
   alias+DB combination — i.e., the shipped evaluator and the server evaluator disagree on this
   uid. Fix: remove the alias or rename the DB row to match it.
2. **DSL executor AST whitelist rejects `break`/`continue` (critical).** The AST-whitelisted
   `execute_dsl_code` omits `ast.Break`/`ast.Continue`; ~20 official Phase-1 `hard_logic_py`
   programs use `break` (swept across all 3,568 constraint programs). Under stock upstream,
   those constraints raise `Unsupported DSL syntax: Break` and the affected plans' logic
   results zero out (we measured FPR 97.9 on a plan set that scores 99.9 with the fix).
3. **`next_page()` exhaustion sentinel crashes the official baseline agent (medium).** The
   refactored environment returns the string `"No more data."` on exhaustion; the repo's own
   `UrbanTrip/tpc_agent.py` feeds this into `pd.concat`, raising a `TypeError`.
4. **Truncated `nature_language` in official Phase-1 queries (medium).** At least two queries
   end mid-sentence (`20250322123458076116` ends "2. Total budget for";
   `20250323003100968460` ends "satisfy at least one of the following: 1. Do"), so their
   constraints are unrecoverable from natural language alone — no honest NL→DSL pipeline,
   including the Phase-2 harness setting, can solve them; only the oracle `hard_logic_py`
   reveals the real requirements. If the held-out Phase-2 set has the same defect, harness
   scores will be depressed for all teams equally; we suggested a data pass before the final
   evaluation.
5. **Doubled-quote typo in an official constraint literal (low).** `'Rui''en Town'` evaluates
   via Python string concatenation to `'Ruien Town'`, which matches no POI; the constraint as
   written is satisfiable only vacuously.
6. **Baseline NL→DSL few-shot example induces invented room constraints (medium).** The
   English step-2 few-shot example includes `rooms==2, room_type==2`; smaller models copy the
   pattern and invent room constraints for queries that never mention rooms. Measured on
   Qwen3.6-27B (the Phase-2 model): 49 of 61 translation failures in our 113-uid sample were
   caused solely by this; fidelity 46% → ~89% once removed. Every team building on the stock
   prompt inherits this.
7. **`agent_env` harness cannot load EN data (low).** `solve_script_with_harness.py` omits
   `lang="en"` when loading queries, so an EN-data checkout crashes with a `KeyError` on the
   first uid; one-line `--lang` plumbing fix.

## 5. Reproducibility

- **Phase 2.** The submission is a self-contained `chinatravel/agent/tpc_agent/` directory
  that runs on a stock organizer checkout against any OpenAI-compatible endpoint
  (configuration via `CHINATRAVEL_OPENAI_BASE_URL` / `CHINATRAVEL_OPENAI_MODEL` /
  `OPENAI_API_KEY`); sampling parameters are pinned for determinism. No oracle fields
  (`hard_logic`, `hard_logic_py`, `hard_logic_nl`) are read anywhere at inference time; the
  enrichment gate consumes only the DSL generated live by the model. [TODO: final determinism
  audit numbers (repeat-run variance) and packaging manifest.]
- **Phase 1.** Full pipeline reproduction: `run_full_pipeline.sh` (planner + enrichment
  battery from source) → pilot chains (`chain_pilot2/3.sh`) → wave chains
  (`chain_wave4/5.sh`) → endgame (`_ARCHIVE_V6_ceiling`) → v13 validation
  (`run_logs/crack713880_validate.py`). The score trajectory is documented in
  `README_PIPELINE.md`; complete 1000-plan snapshots exist at every milestone
  (`_ARCHIVE_V13_100` final, `_ARCHIVE_V6_ceiling` at 99.953, `_ARCHIVE_CLAUDE_*` for the AI
  lane).
- **Local simulation environment.** Qwen3.6-27B stood up locally via Ollama (with a
  GGUF-patched llama.cpp) as the SGLang stand-in; on Apple Metal the architecture gains
  nothing from batching (~13 tok/s), which shaped our evaluation cadence.

## 6. Limitations

- **Local Phase-2 numbers are a simulation, not the official setting.** The 99.9991 figure is
  from our own 113-uid split on locally served Qwen3.6-27B via Ollama, not the organizers'
  SGLang deployment on the held-out set. Serving-stack differences (sampling defaults,
  tokenization, concurrency) may shift translation outputs.
- **Sample size.** 113 uids is a fraction of a full evaluation split; per-uid variance on a
  larger held-out set may expose translation failure modes not present in our sample.
- **Irreducible NL defects.** Queries with truncated natural language (Finding 4) are
  unsolvable by any honest NL-only pipeline, ours included; our scores on such uids depend
  entirely on whether the held-out data shares the defect.
- **Evaluator divergence risk.** Our Phase-1 endgame explicitly navigated a divergence between
  the open-source and server evaluators (Finding 1). We chose the plan variant that passes
  under both plausible server configurations, but any further unshipped server-side changes
  remain a residual risk for both phases.
- **Phase-1 methodology is offline-heavy.** The per-uid best-merge, archive competition, and
  per-instance rebuilds that closed the last Phase-1 points are development-time techniques;
  they do not transfer to the organizer-run per-query harness and are not part of the Phase-2
  submission. Phase-2 performance rests on the live pipeline of Section 2 alone.
- **AI-assisted refinement.** As disclosed in Section 3.5, Phase-1 included AI-assisted
  per-instance plan construction. We believe this is within the rules as published; we flag it
  here so organizers can assess compliance with full information.

---

*Prepared by Team Antarctic penguins. Facts in this report are drawn from our internal
engineering records (`ROAD_TO_100.md`, `PHASE2_DOSSIER.md`, `UPSTREAM_BUGS.md`) and the public
ChinaTravel repository at the commits cited above.*

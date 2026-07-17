# TPC 2026 — Complete Campaign Report & Issue Register

**Team:** Antarctic penguins · **Period:** 2026-07-08 → 2026-07-17 · **Branch:** `improve/top5-soft-max` (85 commits, HEAD `350094e`)
**Status at stop:** Phase 1 **100.00 locked (#5)** · Familiarization **98.74 / FPR 99.0** · package feature-complete, round 8 pending

---

## Part I — All runs, chronologically

### Phase 1 (leaderboard, 2026-07-08 → 07-11)

| # | Run | Result | What it did |
|---|-----|--------|-------------|
| 1 | Pilot 3 recovery (power-loss) | 99.910 → **99.917** | resumed killed chain; variant-C transit flags on 158 uids |
| 2 | residual-100-audit (29 agents) | verdict: fixable | overturned "structural residual" belief; per-uid fix recipes |
| 3 | Wave 4 (rebook/fillerswap/seqswap/lateattr) | → **99.951** | audited fixers; DAV+DDR → 100 |
| 4 | attdilute + ATT endgame (7 rebuilds) | → **99.95299** | local-evaluator ceiling; all soft metrics 100 |
| 5 | crack-713880 (alias forensics) | → **100.000** | leaderboard refuted our "unsatisfiable" proof; dominant-spelling plan passes both server variants |
| 6 | **v13 submission** | **board-confirmed 100.00** | official organizer email, rank #5, tie rule secures Phase 2 |

### AI-lane / Phase-2 preparation (2026-07-10 → 07-17)

| # | Run | Set | Result | Meaning |
|---|-----|-----|--------|---------|
| 7 | Claude NL→DSL (40 agents, 14.5 min) | phase-1 1000 | fidelity 78.7% | fast AI source |
| 8 | Pure-Claude source | 1000 | **86.53** | first honest AI number; FPR 73.5 |
| 9 | Hybrid / hybrid-max | 1000 | 99.93 / **99.953** | AI + donor backstop; ties ceiling |
| 10 | Qwen3.6-27B baseline | 113 eval set | fidelity **46%** | Phase-2 model raw = weak |
| 11 | Hardening rounds 1–2 (prompts+normalizers) | 113 | fidelity **87.6%** | room-boilerplate + builtins killed |
| 12 | DSL canonicalizer + extractor widening | 113 e2e | **76.28 → 96.97** | extractor dialect-lock was the real binding constraint; planner p95 307→46 s |
| 13 | Round 3 (disjunction verifier) | 113 e2e | → **99.00** | 467/1000 queries carry disjunctions |
| 14 | A6 package + ceiling push | 113 e2e | → **99.9991** | `tpc_agent_penguins/` stock-verified |
| 15 | Dress rehearsal + fixes | 5 problem uids ×2 | 10/10 emit <300 s | budget accounting, fallback plan, determinism |
| 16 | zh-instruction A/B | 25 uids | en 22/25 = zh 22/25 | tie → bilingual query-routing (user decision) |
| 17 | Generalization probe (never-seen human split) | 60 | **82.16** | reality check; 9/18 fails benchmark-intrinsic |
| 18 | Round 4 (coverage/span-grounding/conventions) | 60 probe | → **89.08**; satisfiable-FPR 98% | generalization gap mostly closed |
| 19 | Full-1000 honest sweep | 1000 | **91.95** (FPR 86.8) | pre-round-5 baseline on complete distribution |
| 20 | Round 5 (12 fixes from 132-fail autopsy) | 1000 | → **98.97** (FPR 98.3) | budget-rescope was #1 killer |
| 21 | Fresh holdout (153 untouched uids) | holdout | raw 85.37 / cond. **87.09** | **overfit quantified** (see Issue O-1) |
| 22 | Familiarization 100 (organizer Phase-2 data) | famil | **57.81** (FPR 13) | style shock — see Part II |
| 23 | Round 6 (containment windows, real mode semantics, rooms) | famil | → **81.50** (FPR 58) | phase-1-fitted defenses were liabilities |
| 24 | Round 7 (un-vacuation + window store + must-dine priority) | famil | → **98.74** (FPR **99.0**) | 1 residual fail (timeout) |

**Verified interim states archived:** `_ARCHIVE_V13_100`, `_ARCHIVE_V6_ceiling`, `_ARCHIVE_CLAUDE_{pure,hybrid,hybridmax}`, plus ~25 phase-1 waypoints.

---

## Part II — Issue register (open + closed)

### O. Overfitting (the issue you asked about)

- **O-1 [MEASURED, PARTIALLY OPEN] Iterated-set overfit.** Full-1000 after round 5 scored 98.97, but the untouched fresh holdout scored 87.09 conditional — a **~12-point generalization gap** on hard human-split queries. Interpretation is mixed: part real overfit, part the human split's difficulty, part our own zh→en oracle materialization noise. Mitigations in place: fresh-holdout protocol (one-shot, never iterated), DB-gated fixes only, per-round zero-regression gates. **The famil trajectory (13→99 FPR in 2 rounds on organizer data) is the better generalization signal, but famil is now itself an iterated set (rounds 6–7 fitted to it) — the held-out test remains the only uncontaminated measurement, and our famil 98.74 should be read as an upper bound.**
- **O-2 [CLOSED, INSTRUCTIVE] Phase-1 conventions were Phase-2 liabilities.** Round 3's "covering" window template and round 2's canonical vacuous transport idiom were *correct for phase-1 oracle style* and *wrong for phase-2 style* — famil FPR 13 was largely our own hardening backfiring. Fixed in round 6. Lesson: fitting to any proxy carries transfer risk; only organizer-authored data settles conventions.
- **O-3 [CLOSED] Round-4 budget injection over-fired** on the wider phase-1 distribution (the #1 round-5 category, ~21 uids) — a fix tuned on 60 human queries misbehaving at scale. Fixed by enforcement-gating the ungrounded-cap detector.

### T. Translation layer (Qwen3.6-27B NL→DSL)

- **T-1 [CLOSED]** Room-constraint hallucination from the repo's own few-shot (49/61 misses) — round 1.
- **T-2 [CLOSED]** Sandbox builtins (`len`/`bool`) crash constraints — round 1 + mechanical check.
- **T-3 [CLOSED]** Count-boilerplate arithmetic (`taxi_cars!=2`) — round-2 normalizer.
- **T-4 [CLOSED]** Disjunction collapse/split ("at least one of") — round-3 verifier; en+zh markers.
- **T-5 [CLOSED]** Dropped implicit constraints (local cuisine, airfare→airplane, only-free) — round 4/5 coverage rules, DB-gated.
- **T-6 [CLOSED]** Vacuous transportation-guard dialect — round 7 un-vacuation (was blinding three gates simultaneously).
- **T-7 [OPEN, residual]** Model stochasticity: even at temperature 0, occasional dropped clauses on re-translation (observed once in rehearsal). Ensemble arbitration designed but not shipped.

### P. Planner (UrbanTripOptimizedV6)

- **P-1 [CLOSED]** Constraint extractor dialect-locked to oracle DSL style (the 76.28 crater) — canonicalizer + widened extractor.
- **P-2 [CLOSED]** Evening-window POIs never scheduled — constraint-driven evening repair (also caught a false "already implemented" claim in a prior commit message — see M-2).
- **P-3 [CLOSED]** Window store collided on same-POI multiple windows; windowed must-dine lost its slot to unwindowed musts — round 7.
- **P-4 [OPEN]** Slow class: 4–5 famil uids at 300–324 s raw planning (1-day flight trips + windows + cost caps). The packaged agent's 275 s emit-deadline + fallback plan bounds the damage (partial credit, never zero), but native completion is worth round 8.
- **P-5 [OPEN]** 2 phase-1 regressions from round 7 (`20250322120737483309`, `20250322123456665886`) — net-positive trade accepted, unfixed.

### G. Gating / enrichment

- **G-1 [CLOSED]** Enrichers gated on oracle constraints (Phase-2 disqualifying) — `constraint_gate.py`, poison-tested leak-free.
- **G-2 [CLOSED]** Enrichment inserted banned transport modes when the generated constraint was vacuous — fixed via T-6.
- **G-3 [OPEN, minor]** Famil soft metrics below phase-1 saturation (DAV 97.2, ATT 96.2, DDR 98.3 ≈ 0.3 Overall headroom) — enrichment not yet tuned for famil's denser plans.

### U. Upstream / benchmark issues (all documented in UPSTREAM_BUGS.md, 9 items)

- **U-1 [REPORTED? → user action]** Evaluator alias contradiction (Bistro Sola) — the bug that briefly made 100.00 look impossible.
- **U-2** AST whitelist rejects `break`/`continue` used by ~20 official constraints (craters FPR on stock upstream).
- **U-3** `people_number` NameError in the evaluator zeroes valid constraints for every team.
- **U-4** ~15% of human-split oracle constraints unsatisfiable against the DBs; famil has ≥1 intrinsic too.
- **U-5** Truncated `nature_language` queries (≥4 known) — unsolvable for any honest NL pipeline.
- Plus: next_page sentinel crash, room-inducing baseline few-shot, agent_env EN loading, quote-typo literal.
- **STATUS: the organizer email with all of this has NOT been sent yet — highest-value pending human action.**

### M. Meta / process issues

- **M-1 [SYSTEMIC, worked around]** Background harness tasks killed after ~minutes on this machine; API auth outages killed several workflow agents mid-run. Workaround: nohup-detached scripts + resumable per-uid caching everywhere; no data was ever lost.
- **M-2 [CAUGHT]** One workflow agent's commit message claimed a fix ("evening-window support") that was never implemented — caught by the integration agent's verification. Lesson: verify claims against diffs, not reports.
- **M-3 [ACCEPTED]** The famil oracle-scorer initially crashed on leftover template asserts (153 vs 100) — cosmetic, fixed inline; wasted one evening run due to a stale-path launch.

---

## Part III — State at stop & the runway

**Ready today:** `tpc_agent_penguins/` (stock-verified, bilingual, deterministic, deadline-bounded with fallback, famil FPR 99). Docs: ROAD_TO_100, PHASE2_DOSSIER, UPSTREAM_BUGS, TECH_REPORT_DRAFT, MATERIALS.

**Round 8 backlog (unstarted, ~2–3 days):** P-4 slow class, P-5 regressions, G-3 soft tuning, final package re-sync + stock re-verify, TECH_REPORT finalization.

**Human actions pending:** send UPSTREAM_BUGS.md to chinatravel454@gmail.com · IJCAI-ECAI 2026 registration · Round-2 portal submission when announced (before Aug 1) · tech report by Aug 7.

**Honest projection for the held-out set:** famil-style → high 90s (upper bound 98.7); human-split-style → high 80s–low 90s; the intrinsic-defect rate (U-3/U-4/U-5) caps everyone equally, so relative position — which is what decides 1st/2nd/3rd — favors the team that measured all of this first.

"""Single source of truth for WHICH hard_logic_py constraints gate plan edits.

Every root-level enricher (enrich_*.py, merge_soft.py, repair_fails.py) gates
its edits through a passes(uid, plan) helper that evaluates schema +
commonsense + hard-logic on the query dict returned by load_query().  In
phase 1 that dict carries the ORACLE hard_logic_py.  In phase 2 the harness
never sees oracle constraints, so gating must run on the GENERATED
constraints produced by the NL->DSL translation step (the translation cache,
e.g. cache/translation_Claude_reflect/<uid>.json).

Modes, selected by environment variable (read once at import):

  GATE_CONSTRAINTS unset / "oracle"  (DEFAULT)
      apply_gate() is a strict no-op.  All existing phase-1 chains and
      reproductions behave byte-identically.

  GATE_CONSTRAINTS=generated  +  GATE_TRANSLATION_CACHE=<dir>
      apply_gate(qd) replaces each query's hard_logic_py IN PLACE with the
      generated constraint list from <dir>/<uid>.json.  Because the swap
      happens on the shared query dict immediately after load_query(), EVERY
      consumer sees the generated constraints with no per-callsite changes:
        * passes() -> evaluate_hard_constraints_v2(qd, ...)
        * direct evaluate_constraints_py(qd[uid]["hard_logic_py"], ...) calls
          (enrich_seqswap cap probing)
        * pin / type / cap parsing that reads query["hard_logic_py"] directly
          (enrich_fillerswap, enrich_attdilute, enrich_att, enrich_gapattr,
           enrich_endattr, enrich_travelday_meals)
      That is semantically correct for phase 2: the harness pins/caps what its
      own translation demands.  Schema and commonsense checks are unchanged --
      they never read constraints.

Usage (right after load_query):

    from constraint_gate import apply_gate
    qi, qd = load_query(args)
    apply_gate(qd)
"""
import ast
import json
import os
import sys

GATE_MODE = (os.environ.get("GATE_CONSTRAINTS") or "oracle").strip().lower()
GATE_CACHE = (os.environ.get("GATE_TRANSLATION_CACHE") or "").strip()

if GATE_MODE not in ("oracle", "generated"):
    raise RuntimeError(
        f"GATE_CONSTRAINTS must be 'oracle' or 'generated', got {GATE_MODE!r}")


def generated_constraints(uid):
    """The generated hard_logic_py list for uid from the translation cache.

    The list is canonicalized exactly like the planner does at load time
    (UrbanTripOptimizedV6.run -> canonicalize_query_hard_logic), so gating
    sees the same effective constraints the planning pipeline enforced --
    including repairs of vacuous dialect patterns that would otherwise let
    an edit silently violate the intended constraint.
    """
    path = os.path.join(GATE_CACHE, f"{uid}.json")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    hl = data.get("hard_logic_py", [])
    if isinstance(hl, str):  # some caches store the list as a repr string
        hl = ast.literal_eval(hl)
    from chinatravel.agent.UrbanTrip.dsl_canonicalizer import (
        canonicalize_query_hard_logic,
    )
    data["hard_logic_py"] = list(hl)
    return canonicalize_query_hard_logic(data)["hard_logic_py"]


def constraints_for(uid, query):
    """Constraint list for uid honoring the mode (query = oracle-loaded dict)."""
    if GATE_MODE == "generated":
        return generated_constraints(uid)
    return query.get("hard_logic_py", []) or []


def apply_gate(qd):
    """Swap hard_logic_py on every query in qd according to the mode.

    Mutates qd in place and returns it.  No-op in oracle mode (the default),
    so phase-1 behavior is untouched.
    """
    if GATE_MODE != "generated":
        return qd
    if not GATE_CACHE or not os.path.isdir(GATE_CACHE):
        raise RuntimeError(
            "GATE_CONSTRAINTS=generated requires GATE_TRANSLATION_CACHE to "
            f"point at a translation cache directory (got {GATE_CACHE!r})")
    missing = []
    for uid, q in qd.items():
        try:
            q["hard_logic_py"] = generated_constraints(uid)
        except FileNotFoundError:
            # No translation for this uid -> the phase-2 harness would have no
            # hard constraints for it either; gate on none, but say so loudly.
            missing.append(uid)
            q["hard_logic_py"] = []
    print(f"[constraint_gate] generated mode: hard_logic_py for {len(qd)} "
          f"queries taken from {GATE_CACHE}"
          + (f" ({len(missing)} uids MISSING from cache -> empty constraint "
             f"list, e.g. {missing[:3]})" if missing else ""),
          file=sys.stderr)
    return qd

"""Custom-agent runner for ``solve_script_with_harness.py`` (harness = "tpcagent").

Runs the team's deterministic NeSy planner (TPCAgent from the installed
``chinatravel/agent/tpc_agent`` package) IN-PROCESS, instead of driving an
agentic LLM coding harness (OpenCode/Codex). The organizers invoke exactly:

    python agent_env/scripts/solve_script_with_harness.py

With ``[run].harness = "tpcagent"`` this module's ``run_tpc_agent()`` is called
once per query and returns a plan dict; ``solve_query`` then writes it to
``results/<method>/<uid>.json`` and evaluates it -- same path as the other
harnesses.

The planner translates the *visible* query (NL only; hard_logic* stripped) to
DSL via the SGLang-served Qwen3.6-27B and searches a valid plan, so ``oracle``
is never read. See PHASE2_HARNESS_GLUE.md for the two solve_script edits and the
``[tpcagent]`` config stanza.

Budget: the whole 100-query run must finish inside 5 hours. This module keeps a
global soft deadline (default 4h50m) and caps each query so one slow query
cannot exhaust the budget; on timeout/error it emits the deterministic DB-only
fallback plan so ``results/<uid>.json`` is never missing or empty.
"""
import os
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # <repo root> (…/agent_env/scripts/..)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chinatravel.environment.world_env import WorldEnv

# Defaults (overridable via the [tpcagent] config section).
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_MODEL = "Qwen3.6-27B"
DEFAULT_TOTAL_BUDGET = 17400   # 4h50m of the 5h/100 limit — leaves a safety margin
DEFAULT_PER_QUERY = 170        # 100 * 170s = 4h43m worst case
_FLOOR = 20                    # never cap below this (fallback still emits fast)
_RESERVE = 30                  # keep this much global slack in front of the deadline

_AGENT = {}                    # lang -> constructed agent (built once, reused)
_STATE = {"deadline": None}    # global run deadline, set on first call


def _package_imports():
    """Resolve (TPCAgent, TPCLLM, build_fallback_plan) from the REAL planner
    package. The submission installs it at ``chinatravel/agent/tpc_agent``; the
    in-repo dev copy is ``tpc_agent_penguins`` while ``tpc_agent`` is only the
    stock stub (no planner, empty plan). Gate on ``fallback_plan`` existing so
    the stub is skipped in dev and the installed package is used in the
    submission -- NOT via load_model.init_agent, which imports the stub."""
    import importlib

    last_exc = None
    for pkg in ("tpc_agent", "tpc_agent_penguins"):
        try:
            base = f"chinatravel.agent.{pkg}"
            bfp = importlib.import_module(f"{base}.fallback_plan").build_fallback_plan
            TPCLLM = importlib.import_module(f"{base}.tpc_llm").TPCLLM
            TPCAgent = importlib.import_module(f"{base}.tpc_agent").TPCAgent
            return TPCAgent, TPCLLM, bfp
        except Exception as exc:  # stub lacks fallback_plan -> try the next name
            last_exc = exc
    raise ImportError(f"No runnable tpc_agent package found: {last_exc!r}")


def _apply_sglang_env(cfg):
    provider = cfg.get("provider", {}) if isinstance(cfg.get("provider"), dict) else {}
    base_url = cfg.get("base_url") or provider.get("base_url") or DEFAULT_BASE_URL
    model = cfg.get("model") or DEFAULT_MODEL
    api_key = cfg.get("api_key") or os.environ.get(
        str(cfg.get("api_key_env") or ""), ""
    ) or "EMPTY"
    # setdefault: honour anything already exported by the eval environment.
    os.environ.setdefault("CHINATRAVEL_OPENAI_BASE_URL", str(base_url))
    os.environ.setdefault("CHINATRAVEL_OPENAI_MODEL", str(model))
    os.environ.setdefault("CHINATRAVEL_OPENAI_API_KEY", str(api_key))
    os.environ.setdefault("CHINATRAVEL_LLM_NAME", str(model))


def _get_agent(lang, cache_dir, log_dir):
    if lang not in _AGENT:
        TPCAgent, TPCLLM, _ = _package_imports()
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        # Construct directly (matches init_agent's TPCAgent(**kwargs)) so we always
        # build the REAL planner, never load_model's stub resolution.
        _AGENT[lang] = TPCAgent(
            method="TPCAgent",
            env=WorldEnv(lang=lang),
            backbone_llm=TPCLLM(),
            cache_dir=cache_dir,
            log_dir=log_dir,
            debug=False,
            lang=lang,
        )
    return _AGENT[lang]


def _fallback(agent, query):
    try:
        _, _, build_fallback_plan = _package_imports()
        return build_fallback_plan(query, agent.env)
    except Exception:
        traceback.print_exc()
        return {"itinerary": []}


def _self_check(uid, plan, translated, lang):
    """Honest in-run self-verification against the GENERATED constraints only
    (LLM-Modulo style: the model proposes, symbolic verifiers judge). Returns
    (full_pass, cs_pass, satisfied_count) — never touches oracle fields."""
    import io
    import json as _json
    from contextlib import redirect_stdout
    from copy import deepcopy

    if not isinstance(plan, dict) or not plan.get("itinerary"):
        return (False, False, -1)
    if not isinstance(translated, dict) or translated.get("uid") != uid:
        return (False, False, -1)
    gate_query = {
        k: v for k, v in translated.items() if not str(k).startswith("_urbantrip_")
    }
    cons = list(gate_query.get("hard_logic_py") or [])
    try:
        from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
        from chinatravel.evaluation.commonsense_constraint import (
            evaluate_commonsense_constraints,
        )
        from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
        from chinatravel.evaluation.utils import load_json_file
        from chinatravel.symbol_verification.concept_func import func_dict

        plan = _json.loads(_json.dumps(plan, ensure_ascii=False, default=str))
        buf = io.StringIO()
        with redirect_stdout(buf):
            schema = load_json_file("chinatravel/evaluation/output_schema.json")
            _, _, sp = evaluate_schema_constraints([uid], {uid: plan}, schema=schema)
            *_, cp = evaluate_commonsense_constraints(
                [uid], {uid: gate_query}, {uid: plan}, verbose=False, lang=lang
            )
            *_, lp = evaluate_hard_constraints_v2(
                [uid], {uid: gate_query}, {uid: plan},
                env_pass_id=list(cp), verbose=False, lang=lang,
            )
        n_sat = 0
        for c in cons:
            vd = deepcopy(func_dict)
            vd["plan"] = plan
            try:
                exec(c, {"__builtins__": {"set": set}}, vd)
                if bool(vd.get("result", False)):
                    n_sat += 1
            except Exception:
                pass
        return (uid in sp and uid in cp and uid in lp, uid in cp, n_sat)
    except Exception:
        return (False, False, -1)


def _purge_translation_cache(cache_dir, llm_name, uid):
    import glob as _glob

    for f in _glob.glob(
        os.path.join(str(cache_dir), f"translation_{llm_name}_reflect", f"{uid}.json")
    ):
        try:
            os.remove(f)
        except OSError:
            pass


RETRY_MIN_SLACK = int(os.environ.get("PENGUINS_RETRY_MIN_SLACK", "2400"))
RETRY_CAP = int(os.environ.get("PENGUINS_RETRY_CAP", "600"))


def run_tpc_agent(
    *,
    uid,
    query,
    lang="en",
    tpcagent_config=None,
    timeout=900,
    cache_dir="cache/tpcagent",
    log_dir="agent_env/runs/tpcagent",
):
    """Solve one query with the deterministic planner and return a plan dict.

    Self-verified retry loop: after each attempt the plan is checked against
    the GENERATED constraints (honest — the held-out data carries no oracle);
    while the attempt fails its own constraints and the shard has ample
    budget, the query is retried with a fresh translation (deterministically
    perturbed seed) and a fresh planner pass, and the best-scoring attempt is
    kept.
    """
    from func_timeout import func_timeout, FunctionTimedOut

    cfg = tpcagent_config or {}
    _apply_sglang_env(cfg)

    total_budget = int(cfg.get("total_budget_sec", DEFAULT_TOTAL_BUDGET))
    if _STATE["deadline"] is None:
        # a crash-restarted worker must inherit the ORIGINAL run clock, never
        # a fresh budget (the supervisor exports the run's start epoch)
        epoch = os.environ.get("PENGUINS_RUN_START_EPOCH")
        base = float(epoch) if epoch else time.time()
        _STATE["deadline"] = base + total_budget

    per_query = int(cfg.get("per_query_timeout", min(int(timeout), DEFAULT_PER_QUERY)))
    cap_env = os.environ.get("PENGUINS_PER_QUERY_CAP")
    if cap_env:
        # sharded parallel workers each own ~1/N of the split, so the
        # per-query share of the global budget grows accordingly
        per_query = max(per_query, int(cap_env))

    agent = _get_agent(lang, str(cache_dir), str(log_dir))

    if os.environ.get("PENGUINS_FORCE_FALLBACK_UID") == str(uid):
        # this query has repeatedly crashed its worker: never run the planner
        # again, emit the deterministic fallback immediately
        return _fallback(agent, query)

    def _attempt(cap_s):
        saved_stdout = sys.stdout  # agent.run redirects stdout to its log
        try:
            _succ, p = func_timeout(
                cap_s,
                agent.run,
                args=(query,),
                kwargs=dict(prob_idx=uid, oralce_translation=False),  # organizer's spelling
            )
            return p
        except FunctionTimedOut:
            return None
        except Exception:
            traceback.print_exc()
            return None
        finally:
            sys.stdout = saved_stdout

    max_attempts = int(
        os.environ.get("PENGUINS_MAX_ATTEMPTS", "3" if cap_env else "1")
    )
    llm = getattr(agent, "backbone_llm", None)
    base_seed = getattr(llm, "seed", None)

    best_plan, best_rank = None, (False, False, -2)
    try:
        for attempt in range(1, max_attempts + 1):
            remaining = _STATE["deadline"] - time.time()
            if attempt == 1:
                cap = max(_FLOOR, min(per_query, remaining - _RESERVE))
            else:
                if best_rank[0] or remaining < RETRY_MIN_SLACK:
                    break  # already passing, or shard budget too thin to retry
                # deterministic variation: fresh translation + perturbed seed
                _purge_translation_cache(
                    cache_dir, getattr(llm, "name", "Qwen3.6-27B"), uid
                )
                if llm is not None and base_seed is not None:
                    llm.seed = base_seed + 1000 * (attempt - 1)
                cap = max(_FLOOR, min(RETRY_CAP, remaining - _RESERVE))
            plan_i = _attempt(cap)
            translated = getattr(agent.planner, "query", None)
            rank_i = _self_check(uid, plan_i, translated, lang)
            if plan_i is not None and rank_i > best_rank:
                best_plan, best_rank = plan_i, rank_i
            if best_rank[0]:
                break  # generated-constraint full pass — done
            if attempt > 1:
                print(
                    f"[tpc_agent] retry {attempt}: rank={rank_i} best={best_rank}",
                    file=sys.stderr,
                )
    finally:
        if llm is not None and base_seed is not None:
            llm.seed = base_seed

    plan = best_plan
    if not isinstance(plan, dict) or not plan.get("itinerary"):
        plan = _fallback(agent, query)
    return plan

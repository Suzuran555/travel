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

from chinatravel.agent.load_model import init_agent
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
    """Import LLM + fallback from the installed package (submission: ``tpc_agent``),
    falling back to the in-repo dev package name (``tpc_agent_penguins``)."""
    try:
        from chinatravel.agent.tpc_agent.tpc_llm import TPCLLM
        from chinatravel.agent.tpc_agent.fallback_plan import build_fallback_plan
    except Exception:
        from chinatravel.agent.tpc_agent_penguins.tpc_llm import TPCLLM
        from chinatravel.agent.tpc_agent_penguins.fallback_plan import build_fallback_plan
    return TPCLLM, build_fallback_plan


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
        TPCLLM, _ = _package_imports()
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        _AGENT[lang] = init_agent(
            {
                "method": "TPCAgent",
                "env": WorldEnv(lang=lang),
                "backbone_llm": TPCLLM(),
                "cache_dir": cache_dir,
                "log_dir": log_dir,
                "debug": False,
                "lang": lang,
            }
        )
    return _AGENT[lang]


def _fallback(agent, query):
    try:
        _, build_fallback_plan = _package_imports()
        return build_fallback_plan(query, agent.env)
    except Exception:
        traceback.print_exc()
        return {"itinerary": []}


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
    """Solve one query with the deterministic planner and return a plan dict."""
    from func_timeout import func_timeout, FunctionTimedOut

    cfg = tpcagent_config or {}
    _apply_sglang_env(cfg)

    total_budget = int(cfg.get("total_budget_sec", DEFAULT_TOTAL_BUDGET))
    if _STATE["deadline"] is None:
        _STATE["deadline"] = time.time() + total_budget

    per_query = int(cfg.get("per_query_timeout", min(int(timeout), DEFAULT_PER_QUERY)))
    remaining = _STATE["deadline"] - time.time()
    cap = max(_FLOOR, min(per_query, remaining - _RESERVE))

    agent = _get_agent(lang, str(cache_dir), str(log_dir))
    saved_stdout = sys.stdout  # agent.run redirects stdout to its per-query log
    plan = None
    try:
        _succ, plan = func_timeout(
            cap,
            agent.run,
            args=(query,),
            kwargs=dict(prob_idx=uid, oralce_translation=False),  # keep organizer's spelling
        )
    except FunctionTimedOut:
        plan = _fallback(agent, query)
    except Exception:
        traceback.print_exc()
        plan = _fallback(agent, query)
    finally:
        sys.stdout = saved_stdout

    if not isinstance(plan, dict) or not plan.get("itinerary"):
        plan = _fallback(agent, query)
    return plan
